from pathlib import Path
from typing import List
from langchain.schema import Document   
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import TOP_K, BM25_WEIGHT, DENSE_WEIGHT


def build_dense_retriever(vectorstore: Chroma, k: int = TOP_K):
    """
    Dense retriever using cosine similarity over ChromaDB vectors.

    search_type="mmr" uses Maximal Marginal Relevance — instead of
    returning the top-K most similar chunks (which might all say the
    same thing), MMR balances relevance with diversity.
    """

    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k*3, "lambda_mult": 0.7})

    return retriever

def build_bm25_retriever(chunks: List[Document], k: int = TOP_K):
    """
    BM25 improves on simple TF-IDF by:
    - Applying diminishing returns to term frequency
      (mentioning "revenue" 10 times isn't 10x more relevant than 1 time)
    - Normalizing for document length
      (a short chunk with "revenue" once is more focused than a long
       chunk with "revenue" once buried in other content)
"""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever

def build_hybrid_retriever(vectorstore: Chroma, chunks: List[Document], k: int = TOP_K, bm25_weight: float = BM25_WEIGHT, dense_weight: float = DENSE_WEIGHT)-> EnsembleRetriever:
    """
    Combine BM25 and dense retrievers using EnsembleRetriever. The weights control the balance between the two.
    """
    bm25_retriever = build_bm25_retriever(chunks, k)
    dense_retriever = build_dense_retriever(vectorstore, k)

    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[bm25_weight, dense_weight])

    print(f"Hybrid retriever built with BM25 weight: {bm25_weight}, Dense weight: {dense_weight}, top_k: {k}")
    return hybrid_retriever

def retrieve(retriever: EnsembleRetriever, query: str) -> List[Document]:
    """
    Retrieve relevant chunks for the query using the provided retriever.
    """
    results = retriever.invoke(query)
    print(f"Retrieved {len(results)} documents for query: '{query}'")
    for i, doc in enumerate(results):
        meta=doc.metadata
        print(f"  [{i+1}] {meta.get('ticker','?')} "
              f"p.{meta.get('page','?')} | "
              f"{doc.page_content[:100].strip()}...")
    return results