import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import pathlib
import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

def get_doc_vectorstore():
    if not os.path.exists("faiss_index"):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_query_vectorstore():
    if not os.path.exists("query_index"):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("query_index", embeddings, allow_dangerous_deserialization=True)

def save_query_embedding(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("query_index"):
        vector_store = FAISS.from_texts([query], embedding=embeddings)
    else:
        vector_store = get_query_vectorstore()
        vector_store.add_texts([query])
    vector_store.save_local("query_index")

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("""
     Try using Context or the Past Queries sent by User in this session for finding answer, but if the answer is not available in the context, reply with "Not enough information is available in the documents provided, but I can get an answer based on the Internet knowledge." and generate a response using Internet data.
    Context:
    {context}
    Past Queries sent by User in this session:
    {query}     
    Question:
    {question}                                 
    """)
    chain = create_stuff_documents_chain(model, prompt)
    return chain

def user_input(user_question):
    new_db = get_doc_vectorstore()
    query_db = get_query_vectorstore()

    docs = new_db.similarity_search(user_question) if new_db else []
    query = query_db.similarity_search(user_question) if query_db else []
    chain = get_conversational_chain() 
    # st.session_state["chat_history"].append({"user:", user_question})

    response = chain.invoke(
        {"context": docs, "query": query, "question": user_question}
    )
    # st.session_state["chat_history"].append({"bot": response})
    save_query_embedding(user_question)
    return response

def delete_faiss_index():
    if os.path.exists("faiss_index") or os.path.exists("query_index"):
        for root, dirs, files in os.walk("faiss_index", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        if os.path.exists("faiss_index"):
            os.rmdir("faiss_index")
        delete_query_index()
        st.success("Cleaned up the cache")
    else:
        st.warning("Cache file doesn't exist")

def delete_query_index():
    if os.path.exists("query_index"):
        for root, dirs, files in os.walk("query_index", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir("query_index")

# Main app
st.set_page_config(page_title="PAQ Bot", page_icon="🤖")
css_path = pathlib.Path("style.css")
load_css(css_path)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
    delete_query_index()
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
st.header("PAQ Bot", divider="red")
st.markdown('<div class="intro">Welcome to PAQ Bot! This bot can help you with your queries based on the documents you provide. Upload your PDF documents and ask your queries. The bot will try to answer your queries based on the content of the documents. Use the &#39;Reset Bot Memory&#39; button to clear the cache and &#39;Stop App button&#39; to stop the app.</div>', unsafe_allow_html=True)
# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])
# Sidebar
with st.sidebar:
    st.header("PAQ Bot", divider="red")
    st.subheader("Upload PDF Documents")
    pdf_docs = st.file_uploader("Pick a pdf file", type="pdf", accept_multiple_files=True)
    if pdf_docs and st.button("Process Documents", key="green"):
        with st.spinner("Processing", show_time=True):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = chonky(raw_text)
            vector_store = get_vectorstore(text_chunks)
            st.markdown('<div class="donepdf">Done</div>', unsafe_allow_html=True)
    if not pdf_docs:
        st.markdown('<div class="uppdf">Please upload a PDF file to start</div>', unsafe_allow_html=True)
    st.markdown('<div class="blanki"></div>', unsafe_allow_html=True)
    st.markdown('<div class="luvacm">Made with ❤️ by PEC ACM </div>', unsafe_allow_html=True)
    st.link_button("View the source code", "https://github.com/Ya-Tin/PDFQueryChatLM.git")
    if st.button("Reset Bot Memory", key="red"):
        delete_faiss_index()
    if st.button("Stop App", key="red2"):
        delete_query_index()
        os._exit(0)    
# Chat input box
user_question = st.chat_input("Input your Query here and Press 'Process Query' button")
if user_question:
    # Append user message first
    st.session_state["messages"].append({"role": "user", "content": user_question})        
    # Display user message immediately
    st.chat_message("user").markdown(user_question)
    with st.spinner("Generating response..."):
            response = user_input(user_question)
        
        # Append assistant's response and display it
    unwanted_line_pattern = r"Not enough information is available in the documents provided, but I can get an answer based on the Internet knowledge."
    response = re.sub(unwanted_line_pattern, "", response)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
