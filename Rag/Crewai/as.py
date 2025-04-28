from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model (using HuggingFace instead of OpenAI)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")