from crewai_tools import YoutubeChannelSearchTool
from langchain_community.embeddings import HuggingFaceEmbeddings

# Option 1: Using config dictionary with Groq and HuggingFace
youtube_channel_tool = YoutubeChannelSearchTool(
    youtube_channel_handle="@krishnaik",
    config={
        "llm": {
            "provider": "groq",
            "config": {
                "model": "mistral-saba-24b",
                "temperature": 0.7
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    }
)

