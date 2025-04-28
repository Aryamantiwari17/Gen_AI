import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7, max_tokens=4096)

# Streamlit UI
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload a PDF and chat with its content.")

session_id = st.text_input("Session ID", value="default_session")

# Initialize session state storage
if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)

# Process uploaded file
if uploaded_file:
    temppdf = "./temp.pdf"
    
    # Save uploaded PDF
    with open(temppdf, "wb") as file:
        file.write(uploaded_file.read())

    loader = PyPDFLoader(temppdf)
    documents = loader.load()

    # Split text and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, which might reference context in the chat history, "
         "formulate a standalone question which can be understood without the chat history. "
         "Do NOT answer the question, just reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Question answering prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. Use the following retrieved context to answer "
         "the question. If you don't know the answer, say that you don't know. Use three sentences maximum and "
         "keep the answer concise.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to manage session chat history
    def get_session(session: str) -> BaseChatMessageHistory:##use session not the global variable session_id 
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Create conversational RAG with history
    conversation_rag = RunnableWithMessageHistory(
        rag_chain,
        get_session,
        input_message_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User input for queries
    user_input = st.text_input("Your Question")
    if user_input:
        session_history = get_session(session_id)
        st.write("Previous Chat History:")
        for msg in session_history.messages:
            st.write(f"**{msg.type.capitalize()}**: {msg.content}") 
        response = conversation_rag.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        
        # Display response and chat history
        st.success(f"Assistant: {response['answer']}")
        st.write("Chat History:", session_history.messages)
        



else:
    st.warning("Please upload a PDF file to start the conversation.")

'''
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []

    for idx, uploaded_file in enumerate(uploaded_files):
        temp_pdf = f"./temp_{idx}.pdf" 

        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.read()) 

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs) 
'''