from crewai import Task
from agent import blog_researcher, blog_writer

# Research Task
research_task = Task(
    description="""Search the YouTube channel for videos about {topic}. 
    Identify key concepts, tutorials, and important explanations.
    Focus on:
    1. Core concepts explained
    2. Practical examples or demos
    3. Unique insights or perspectives""",
    expected_output="""A comprehensive report (3-5 paragraphs) summarizing:
    - Key concepts covered in the videos
    - Important demonstrations or examples
    - Unique insights from the channel""",
    agent=blog_researcher,
    output_file="research_report.md"
)

# Writing Task
write_task = Task(
    description="""Using the research findings, create an engaging blog post about {topic}.
    Structure should include:
    1. Introduction to the topic
    2. Key concepts explained simply
    3. Practical applications
    4. Conclusion with key takeaways""",
    expected_output="A well-structured 800-1000 word blog post in markdown format",
    agent=blog_writer,
    output_file="blog_post.md"
)