"""Central configuration for the Local Wikipedia RAG system.

All constants live here so other modules import from a single source of truth.
Override any value via environment variables of the same name.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SQLITE_PATH = DATA_DIR / "rag.db"
CHROMA_DIR = Path(os.environ.get("RAG_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.environ.get("RAG_CHROMA_COLLECTION", "wiki_rag")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "ollama")

CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "150"))
MIN_CHUNK_SIZE = int(os.environ.get("RAG_MIN_CHUNK_SIZE", "120"))

TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
LLM_TEMPERATURE = float(os.environ.get("RAG_LLM_TEMPERATURE", "0.2"))
LLM_NUM_CTX = int(os.environ.get("RAG_LLM_NUM_CTX", "4096"))
LLM_TIMEOUT = int(os.environ.get("RAG_LLM_TIMEOUT", "180"))

WIKI_LANG = os.environ.get("RAG_WIKI_LANG", "en")
WIKI_USER_AGENT = os.environ.get(
    "RAG_WIKI_USER_AGENT",
    "BLG483E-HW3-RAG/1.0 (educational; contact: student@itu.edu.tr)",
)
WIKI_TIMEOUT = int(os.environ.get("RAG_WIKI_TIMEOUT", "30"))


def ensure_dirs() -> None:
    """Create runtime directories if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
