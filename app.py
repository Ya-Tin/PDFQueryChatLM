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
from langchain.memory import ConversationBufferMemory
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

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "chat_history", "question"])

    memory = st.session_state["memory"]

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, memory=memory)
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

    st.session_state["memory"].save_context(
        {"question": user_question}, 
        {"output": response["output_text"]}
    )

    return response["output_text"]

def main():
    st.set_page_config(page_title="PAQ Bot", page_icon="🤖")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history", 
            input_key="question",
            return_messages=True 
        )
    st.header("🤖 PAQ Bot")

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    with st.sidebar:
        st.header("PAQ Bot")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Pick a PDF file", type="pdf", accept_multiple_files=True)

        if pdf_docs and st.button("Process Documents"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = chonky(raw_text)
                get_vectorstore(text_chunks)
                st.success("Done")

        if not pdf_docs:
            st.info("Please upload a PDF file to start.")

        st.write("Made with ❤️ by PEC ACM")
        "[View the source code](https://github.com/Ya-Tin/PDFQueryChatLM.git)"

    user_question = st.chat_input("Input your Query here and Press 'Process Query' button")
    
    if user_question:
        st.session_state["messages"].append({"role": "user", "content": user_question})
        st.chat_message("user").markdown(user_question)

        response = ""
        with st.spinner("Generating response..."):
            response = user_input(user_question)

        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

if __name__ == "__main__":
    main()
