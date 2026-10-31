import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
FEEDBACK_FILE = BASE_DIR / "feedback.jsonl"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 8
CORPUS_DESCRIPTION = "the FastAPI Python web framework tutorial"
MAX_RETRIES = 2        # query rewrite + re-retrieve attempts
MAX_GEN_ATTEMPTS = 2   # generation attempts (hallucination check)
