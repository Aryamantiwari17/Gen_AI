import streamlit as st 
import os
import sqlite3
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
from db_config import configure_db 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
st.set_page_config(page_title="Chatbot using SQL Database")
st.title("Chat_BOT:chat With SQL_DB")

LOCALDB="USE_LOCALDB"
MYSQL="USE_MYSQL"

radio_opt=["Use SQLite 3 databse-student.db","Connect to you SQL Database"]

selected_opt=st.sidebar.radio(label="Choose the DB which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt)==1:
    db_url=MYSQL
    mysql_host=st.sidebar.text_input("provide my sql host")
    mysql_user=st.sidebar.text_input("MYSQL_User")
    mysql_pwd=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MYSQL Database")
    
else:
    db_url=LOCALDB

llm = ChatGroq(
    model_name="mistral-saba-24b",  # Updated to current Groq model name
    temperature=0.7,
    max_tokens=4096,
 
)
   
if not db_url:
    st.info("Please enter the database information and uri")
    
@st.cache_resource(ttl="2h")
def conifigure_db(db_url, mysql_host=None,mysql_user=None,mysql_pwd=None,mysql_db=None):
    
     if db_url==LOCALDB:
         dffilepath=(Path(__file__).parent/"student.db").absolute()
         print(dffilepath)
         creator=lambda:sqlite3.connect(f"file:{dffilepath}?mode=ro",uri=True)
         return SQLDatabase(create_engine("sqlite:///",creator=creator))
     
     elif db_url==MYSQL:
         if not (mysql_host and mysql_user and mysql_pwd and mysql_db):
             st.error("Please provide all MYSQL connection details")
             st.stop()
             
   
             return SQLDatabase(
              create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_pwd}@{mysql_host}/{mysql_db}"
        )
    )


db_url = "MYSQL"  

if db_url == "MYSQL":
    db = configure_db(db_url, mysql_host, mysql_user, mysql_pwd, mysql_db)
else:
    db = configure_db(db_url)


    
    
    