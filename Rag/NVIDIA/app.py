import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# Initialize the LLM
llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct")

# Title
st.title("📄 NVIDIA with NIM - Document Q&A")

# Function to build vector database
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        split_docs = text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(split_docs, embedding=st.session_state.embeddings)
        st.write("✅ FAISS Vector DB created successfully!")

# Prompt Template for the model
template_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# UI Input
query_input = st.text_input("🔍 Enter your question based on the documents:")

# Button to create VDB
if st.button("📥 Embed Documents"):
    vector_embedding()

# Query Processing
if query_input and "vectors" in st.session_state:
    doc_chain = create_stuff_documents_chain(llm, template_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': query_input})
    elapsed = time.process_time() - start_time

    st.success(f"✅ Answer generated in {elapsed:.2f} seconds")
    st.write(response['answer'])

    # Show matching documents
    with st.expander("📄 Document Chunks Used"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write('-----------------------------')
elif query_input:
    st.warning("⚠️ Please embed the documents first.")
