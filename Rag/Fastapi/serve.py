from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="mistral-saba-24b",  # Updated to current Groq model name
    temperature=0.7,
    max_tokens=4096,
 
)

# Create a template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

# Create chain
chain = prompt_template | llm | StrOutputParser()

# App definition
app = FastAPI(
    title="LangServe Server",
    version="1.0",
    description="A simple API using LangChain runnable interfaces"
)

# Adding chain routes
add_routes(
    app,
    chain,
    path="/translate",  # Changed to more descriptive path
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)