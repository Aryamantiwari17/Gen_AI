from crewai import Crew, Process
from agent import blog_researcher, blog_writer
from task import research_task, write_task
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Crew with Groq as manager LLM
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    manager_llm=ChatGroq(
        temperature=0.7,
        model_name="mistral-saba-24b",
        groq_api_key=os.getenv("GROQ_API_KEY")
    ),
    memory=True,
    verbose=True
)

# Execute the crew
result = crew.kickoff(inputs={'topic': 'AI VS ML VS DL VS Data Science'})
print("Final Output:")
print(result)