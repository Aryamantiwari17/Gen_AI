import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7, max_tokens=4096)

# Streamlit page setup
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("🧮 Text to Math Problem Solver using Mistral")

# Custom function to parse agent output
def parse_agent_output(output):
    try:
        # Clean up any unexpected formatting
        cleaned_output = re.sub(r'```.*?```', '', output, flags=re.DOTALL)
        cleaned_output = re.sub(r'`.*?`', '', cleaned_output)
        return cleaned_output.strip()
    except Exception as e:
        return f"Error parsing output: {str(e)}. Here is the raw output: {output}"

# Custom function to handle math expressions
def safe_math_chain(input_text):
    try:
        # Clean the input of any markdown code blocks or unnecessary formatting
        clean_input = re.sub(r'```.*?```', '', input_text, flags=re.DOTALL)
        clean_input = re.sub(r'`.*?`', '', clean_input)
        
        # Use the math chain
        result = math_chain.run(clean_input.strip())
        return result
    except Exception as e:
        return f"Error in calculation: {str(e)}. Please reformulate your math question."

# Wikipedia tool setup
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for retrieving information from Wikipedia to help with context for problems."
)

# Math calculator setup
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=safe_math_chain,
    description="A tool for solving mathematical calculations. Input should be a mathematical expression to evaluate."
)

# Enhanced prompt template for detailed logical reasoning
detailed_reasoning_prompt = """
You are a math teacher explaining concepts to students. For any math problem:

1. UNDERSTAND: First, identify what the problem is asking and what information is provided.
   
2. PLAN: Explain your approach before calculating anything. What mathematical concepts apply here?
   
3. REASON: Work through the solution step-by-step, explaining your reasoning at each point.
   - Show each transformation or calculation
   - Explain WHY you're taking each step
   - Write out intermediate steps clearly
   - Use clear mathematical notation
   
4. VERIFY: Check if your answer makes sense in the context of the original problem.

5. CONCLUDE: Summarize the solution process and what we learned.

DO NOT simply provide the final answer directly. The learning happens in the reasoning process.

Question: {question}

Detailed Reasoning:
"""

detailed_reasoning_template = PromptTemplate(
    input_variables=["question"],
    template=detailed_reasoning_prompt
)

reasoning_chain = LLMChain(llm=llm, prompt=detailed_reasoning_template)

reasoning_tool = Tool(
    name="Reasoning_tool",
    func=reasoning_chain.run,
    description="A tool for explaining mathematical problems step-by-step with detailed reasoning. Use this for any problem requiring explanation of the thought process."
)

# Initialize the agent
tools = [reasoning_tool, calculator, wikipedia_tool]  # Reasoning_tool is now first priority

assistant_agents = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # Set to True to see the agent's thought process
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": """You are a helpful math tutor. Your primary goal is to help users understand mathematical concepts by showing detailed reasoning, not just answers. 
        Always think about which tool is most appropriate for the question:
        - For any math problem where understanding is important, use the Reasoning_tool FIRST
        - Only use the Calculator tool for final verification or simple calculations
        - Use Wikipedia only when additional context about mathematical concepts is needed
        """
    }
)

# Session setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I am a math chatbot who can solve your math problems and answer logic-based questions. What can I help you with today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User interaction
question = st.text_area("Enter your question")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating a response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            
            try:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = assistant_agents.run(question, callbacks=[st_cb])
                
                # Clean up the response if needed
                cleaned_response = parse_agent_output(response)
                
                st.session_state.messages.append({'role': 'assistant', 'content': cleaned_response})
                st.write("### 📌 Response:")
                st.success(cleaned_response)
            except Exception as e:
                error_msg = f"Error: {str(e)}\n\nPlease try rephrasing your question or ask a different one."
                st.error(error_msg)
                st.session_state.messages.append({'role': 'assistant', 'content': error_msg})
    else:
        st.warning("Please enter a valid question.")