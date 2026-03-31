from pathlib import Path
import pickle
from typing import List
from langchain_community.vectorstores import FAISS
# from langchain_core.schema import Document
from langchain.schema import Document
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import FAISS_DIR, FAISS_COLLECTION
from src.ingestion.embedder import get_embedding_model


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Embed all chunks and store them in FAISS index.
    """

    embeddings=get_embedding_model()
    print(f"\nBuilding FAISS vector store with {len(chunks)} chunks...")
    print(f"Persisting to: {FAISS_DIR}")
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"Vector store built. Collection: {FAISS_COLLECTION}")
    return vectorstore

def load_vectorstore() -> FAISS:
    """
    Load the vectorstore from disk. If it doesn't exist, return None. Call this after the first build — no re-embedding needed.
    """
    embeddings = get_embedding_model()

    if not FAISS_DIR.exists():
        print(f"Vector store directory {FAISS_DIR} not found. Please build the vector store first.")
        return None
    vectorstore=FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)

    print(f"Vector store loaded. Collection: {FAISS_DIR}")

    return vectorstore