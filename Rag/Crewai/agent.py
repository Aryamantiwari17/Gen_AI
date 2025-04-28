from crewai import Agent
from tools import youtube_channel_tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
groq_llm = ChatGroq(
    temperature=0.7,
    model_name="mistral-saba-24b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Researcher Agent
blog_researcher = Agent(
    role='YouTube Content Researcher',
    goal='Extract key information from YouTube channel videos about {topic}',
    backstory="""Expert in analyzing and summarizing technical YouTube content, 
    with deep knowledge of AI, ML, and Data Science topics.""",
    llm=groq_llm,
    tools=[youtube_channel_tool],
    verbose=True,
    memory=True,
    allow_delegation=False
)

# Writer Agent
blog_writer = Agent(
    role='Technical Content Writer',
    goal='Create engaging blog posts from YouTube content about {topic}',
    backstory="""Skilled technical writer who transforms complex topics 
    into clear, engaging content for readers of all levels.""",
    llm=groq_llm,
    tools=[youtube_channel_tool],
    verbose=True,
    memory=True,
    allow_delegation=False
)