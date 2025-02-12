import streamlit as st
import os 
import shutil
from dotenv import load_dotenv
from PyPDF2 import PdfReader
# from htmltemplates import css
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("""
    Try using Context for finding answer, but if the answer is not available in the context, reply with "Not enough information is available in the documents provided, but I can get an answer based on the Internet knowledge" and generate a response using Internet data.
    Context:
    {context}
    Question:
    {question}
    """)
    chain = create_stuff_documents_chain(model, prompt)
    return chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    st.session_state["chat_history"].append({"user:", user_question})

    response = chain.invoke(
        {"context": docs, "question": st.session_state["chat_history"]},
        return_only_outputs=True
    )

    st.session_state["chat_history"].append({"bot": response})

    return response

def delete_faiss_index():
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        st.success("Cleaned up the cache")
    else:
        st.warning("Cache file doesn't exist")

def main():
    st.set_page_config(page_title="PAQ Bot", page_icon="🤖")
    # st.write(css, unsafe_allow_html=True)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.header("🤖 PAQ Bot")

    # Display chat messages
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
    # Sidebar
    with st.sidebar:
        st.header("PAQ Bot")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Pick a PDF file", type="pdf", accept_multiple_files=True)
        if pdf_docs and st.button("Process Documents"):
            with st.spinner("Processing"):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # Get the text chunks
                text_chunks = chonky(raw_text)
                # Create the vector store
                vector_store = get_vectorstore(text_chunks)
                # Notify user
                st.success("Done")
        if not pdf_docs:
            st.info("Please upload a PDF file to start.")
        st.write("Made with ❤️ by PEC ACM")
        "[View the source code](https://github.com/Ya-Tin/PDFQueryChatLM.git)"
        if st.button("Reset Bot Memory"):
            delete_faiss_index()
    # Chat input box
    user_question = st.chat_input("Input your Query here and Press 'Process Query' button")
    if user_question:
        # Append user message first
        st.session_state["messages"].append({"role": "user", "content": user_question})        
        # Display user message immediately
        st.chat_message("user").markdown(user_question)
            
        # Generate response
        with st.spinner("Generating response..."):
                response = user_input(user_question)
            
            # Append assistant's response and display it
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

if __name__ == "__main__":
    main()