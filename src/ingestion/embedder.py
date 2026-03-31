from langchain_huggingface import HuggingFaceEmbeddings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.settings import CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return a HuggingFaceEmbeddings instance.
    This will be used for both semantic chunking and dense retrieval.
    """
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True, "batch_size": 32})
    
    print("Embedding model loaded.")
    return embedder