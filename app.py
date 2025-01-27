import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from htmltemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def chonky(text):
    text_splitter= CharacterTextSplitter(separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Try using Context for finding answer, but if the answer is not available in the context, reply with "Not enough information is available in the documents provided, but I can get an answer based on the Internet knowledge" and generate a response using Internet data.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    # Append to the session state to maintain chat history
    st.session_state["messages"].append({"role": "user", "content": user_question})
    st.session_state["messages"].append({"role": "assistant", "content": response["output_text"]})

def main():
    st.set_page_config(page_title="PAQ Bot", page_icon="🤖")
    
    # Initialize session state for messages and vector store flag
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    if "vector_store_ready" not in st.session_state:
        st.session_state["vector_store_ready"] = False

    st.header("🤖 PAQ Bot")

    # Display chat messages
    

    # Sidebar for document upload
    with st.sidebar:
        st.header("PAQ Bot")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Pick a PDF file", type="pdf", accept_multiple_files=True)
        if pdf_docs and st.session_state["vector_store_ready"] == False:
            with st.spinner("Processing uploaded documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = chonky(raw_text)
                get_vectorstore(text_chunks)
                st.session_state["vector_store_ready"] = True

        # if st.session_state["vector_store_ready"] == True:
        #     st.success("Documents processed! You can now ask queries.")

        if not st.session_state["vector_store_ready"]:
            st.info("Please upload a PDF file to start.")

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
    # Chat input box
    user_question = st.chat_input("Input your query here")
    if user_question:
        if st.session_state["vector_store_ready"]:
            # Process the query immediately if vector store is ready
            user_input(user_question)
        else:
            st.error("Please upload and process a PDF before asking questions.")


if __name__ == "__main__":
    main()
