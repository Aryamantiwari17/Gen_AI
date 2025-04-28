import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define the LLM
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7, max_tokens=4096)

# Define tools
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250))
search_tool = DuckDuckGoSearchRun(name="search")

tools = [search_tool, arxiv_tool, wiki_tool]

# Streamlit UI
st.title("🤖 Mew mew → Chat with Tools, Agents and Search")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Handle user input
if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({'role': "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Agent setup and response
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.invoke(prompt, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.write(response)
