import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize the LLM
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7, max_tokens=4096)

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the provided context only.
    Please provide the most accurate response based on the question.
    Be concise and accurate in your response.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

def create_vector_embedding():
    """Create vector embeddings for the documents and store in session state."""
    try:
        with st.spinner("Creating embeddings and vector store..."):
            # Initialize embeddings
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader("lks/") 
            st.session_state.docs = st.session_state.loader.load()
            
            # Split documents
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Increased chunk size for better context
                chunk_overlap=200
            )
            st.session_state.final_document = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:20]  # Processing first 20 documents for demo
            )
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_document,
                st.session_state.embeddings
            )
            
        st.success("Vector database is ready!")
        st.session_state.embedding_complete = True
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        st.session_state.embedding_complete = False

# Streamlit UI
st.title("Document Q&A with Groq and HuggingFace")
st.write("Upload PDFs to a 'data' folder and ask questions about their content.")

# Initialize session state variables
if 'embedding_complete' not in st.session_state:
    st.session_state.embedding_complete = False

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    if st.button("Create Vector Embeddings"):
        create_vector_embedding()

# Main content area
user_prompt = st.text_input("Enter your query about the research papers:")

if st.session_state.embedding_complete and user_prompt:
    try:
        with st.spinner("Searching for answer..."):
            # Create chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 1})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Process the query
            start_time = time.time()
            response = retrieval_chain.invoke({'input': user_prompt})
            response_time = time.time() - start_time
            
            # Display results
            st.subheader("Answer:")
            st.write(response['answer'])
            
            st.info(f"Response time: {response_time:.2f} seconds")
            
            # Show source documents
            with st.expander("View relevant document sections"):
                for i, doc in enumerate(response['context']):
                    st.markdown(f"**Document section {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"Error processing your query: {str(e)}")
elif user_prompt and not st.session_state.embedding_complete:
    st.warning("Please create vector embeddings first by clicking the button in the sidebar.")