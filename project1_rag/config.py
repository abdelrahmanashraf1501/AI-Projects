from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

# Define paths
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE_DIR, "data")
DB_DIR=os.path.join(BASE_DIR, "chroma_db")

# RAG configurations
CHUNK_SIZE=800
CHUNK_OVERLAP=150
TOP_K=4

# LLM model + Embedding model
CHAT_MODEL=ChatOpenAI(model_name=os.getenv("CHAT_MODEL_NAME"), temperature=0, base_url=os.getenv("INCEPTION_BASE_URL"), api_key=os.getenv("INCEPTION_API_KEY"))
EMBED_MODEL=OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL_NAME"), base_url=os.getenv("BASE_URL"))