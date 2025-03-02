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
from transformers import pipeline

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("""
    Try using Context or the Past Queries sent by User in this session for finding answer, but if the answer is not available in the context, reply with "Not enough information is available in the documents provided, but I can get an answer based on the Internet knowledge" and generate a response using Internet data.
    Context:
    {context}
    Past Queries sent by User in this session:
    {query}     
    Question:
    {question}                                 
    """)
    chain = create_stuff_documents_chain(model, prompt)
    return chain

def get_huggingface_response(question, context, query):
    hf_model = "deepset/roberta-base-squad2"  
    qa_pipeline = pipeline("question-answering", model=hf_model)
    combined_context = " ".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in context])
    if not combined_context:
        combined_context = "No context available."
    inputs = {"question": question, "context": combined_context}
    result = qa_pipeline(inputs)
    return result["answer"]

def user_input(user_question, model_choice):
    new_db = get_doc_vectorstore()
    query_db = get_query_vectorstore()
    docs = new_db.similarity_search(user_question) if new_db else []
    query = query_db.similarity_search(user_question) if query_db else []
    if model_choice == "Google Generative AI":
        chain = get_conversational_chain()
        response = chain.invoke({"context": docs, "query": query, "question": user_question})
    else:
        response = get_huggingface_response(user_question, docs, query)
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
st.header("PAQ Bot", divider="red")
st.sidebar.header("PAQ Bot", divider="red")
st.sidebar.subheader("Upload PDF Documents")
pdf_docs = st.sidebar.file_uploader("Pick a pdf file", type="pdf", accept_multiple_files=True)
if pdf_docs and st.sidebar.button("Process Documents", key="green"):
    with st.spinner("Processing"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = chonky(raw_text)
        get_vectorstore(text_chunks)
        st.sidebar.success("Processing Done!")
model_choice = st.sidebar.radio("Choose Model", ["Google Generative AI", "Hugging Face QA"])
if st.sidebar.button("Reset Bot Memory", key="red"):
    delete_faiss_index()
if st.sidebar.button("Stop App", key="red2"):
    delete_query_index()
    os._exit(0)
user_question = st.chat_input("Input your Query here")
if user_question:
    st.session_state["messages"].append({"role": "user", "content": user_question})
    st.chat_message("user").markdown(user_question)
    with st.spinner("Generating response..."):
        response = user_input(user_question, model_choice)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)