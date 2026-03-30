
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Embedding model ────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Chunking ───────────────────────────────────────────
CHUNK_STRATEGY = "recursive"   # "fixed", "recursive", "semantic"
CHUNK_SIZE = 512               # Number of characters per chunk
CHUNK_OVERLAP = 64             # Overlap between chunks to preserve context

# ── Retrieval ──────────────────────────────────────────
TOP_K = 5                      # Number of chunks to retrieve
BM25_WEIGHT = 0.3              # Weight for sparse (BM25) retrieval
DENSE_WEIGHT = 0.7             # Weight for dense (vector) retrieval

# ── LLM ────────────────────────────────────────────────
LLM_MODEL = "mistral"          # Or "llama3" if you pull that instead
LLM_TEMPERATURE = 0.0          # 0 = deterministic, better for factual Q&A
LLM_MAX_TOKENS = 512

# ── Evaluation ─────────────────────────────────────────
EVAL_SAMPLE_SIZE = 30          # Number of Q&A pairs in our test set
RAGAS_LLM = "mistral"          # LLM used by RAGAS to judge answers

# ── Collection name (ChromaDB) ─────────────────────────
CHROMA_COLLECTION = "finsight_docs"
VECTOR_STORE_TYPE = "faiss"