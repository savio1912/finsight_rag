from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.settings import TOP_K
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def build_reranker(base_retriever, top_n: int = TOP_K):
    """
      Query → hybrid retriever (top-20) → cross-encoder → top-5
    """
    print(f"Loading reranker model: {RERANKER_MODEL}...")
    cross_encoder=HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
    compressor =CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    reranking_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    print(f"Reranker built with model: {RERANKER_MODEL}, top_n: {top_n}")
    return reranking_retriever