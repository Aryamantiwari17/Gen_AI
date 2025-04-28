import streamlit as st
import validators
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7, max_tokens=4096)

# Prompt template
prompt_template = """Provide a concise summary (max 300 words) of the following content:
{text}"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Streamlit UI
st.set_page_config(page_title="LangChain: Summarize YT or Web Content")
st.title("🎥📄 LangChain: Summarize YouTube or Web URL")
st.subheader("Paste any YouTube or Website URL below")

generic_url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize Content"):
    if not generic_url.strip():
        st.error("⚠️ Please provide a URL.")
    elif not validators.url(generic_url):
        st.error("❌ Invalid URL. Please enter a valid one.")
    else:
        try:
            with st.spinner("⏳ Processing..."):
                # Load documents based on source
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False,  
                        language=["en"]
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                            "Accept-Language": "en-US,en;q=0.9"
                        }
                    )

                docs = loader.load()
                
                # Create summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success("✅ Summary Generated:")
                st.write(summary)

        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.error("Note: Some YouTube videos may not have available transcripts or require age verification.")

