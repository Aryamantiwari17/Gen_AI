import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
#Langsmith Tracking
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="pr-shadowy-jazz-89"
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


## prompt teplate

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant.Please respond to the the question asked"),
        ("user","Question:{question}")
    ]
)

##streamlit Framework
st.title("A RAG DEMO with  OLLaMa")
input_text=st.text_input("what question you have in mind?")

##ollama setup
llm=OllamaLLM(model="tinyllama")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
    

