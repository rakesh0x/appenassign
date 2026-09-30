"""Settings loaded from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-ebook")
EMBEDDING_DIMENSIONS = 384

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def validate() -> None:
    missing = [
        name
        for name, value in {
            "PINECONE_API_KEY": PINECONE_API_KEY,
            "GROQ_API_KEY": GROQ_API_KEY,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
