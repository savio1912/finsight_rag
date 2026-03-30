from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter   
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.settings import CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL  

#Strategy 1: Fixed-size chunks

def fixed_size_chunker(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Split documents into fixed-size chunks with overlap.
    This is the simplest strategy but can cut off sentences awkwardly.
    """
    splitter=CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks=splitter.split_documents(documents)
    return _tag_chunks(chunks, strategy='fixed')

#Strategy 2: Recursive chunking

def recursive_chunker(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", ""," "], chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, keep_separator=True)
    chunks=splitter.split_documents(documents)
    return _tag_chunks(chunks, strategy='recursive')

#Strategy 3: Semantic chunking

def semantic_chunker(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, embedding_model: str = EMBEDDING_MODEL) -> List[Document]:
    """
    Use semantic similarity to create chunks that are more coherent.
    This is more complex and computationally expensive, but can yield better results.
    """
    embeddings=HuggingFaceEmbeddings(model_name=embedding_model)
    splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70,
    )
    chunks=splitter.split_documents(documents)
    return _tag_chunks(chunks, strategy='semantic')

def chunk_documents(documents: List[Document], strategy: str=CHUNK_STRATEGY)-> List[Document]:

    strategies={
        "fixed": fixed_size_chunker,
        "recursive": recursive_chunker,
        "semantic": semantic_chunker
    }
    if strategy not in strategies:
        raise ValueError(f"Invalid chunking strategy: {strategy}. Choose from {list(strategies.keys())}")
    
    print(f"\nChunking {len(documents)} pages using '{strategy}' strategy...")
    chunks=strategies[strategy](documents)
    print(f"Generated {len(chunks)} chunks. avg {sum(len(c.page_content) for c in chunks)/len(chunks):.1f} chars/chunk")
    return chunks

def _tag_chunks(chunks: List[Document], strategy: str) -> List[Document]:
    """
    Add metadata to each chunk to indicate which chunking strategy was used.
    This can be useful for analysis later.
    """
    for chunk in chunks:
        chunk.metadata["chunk_strategy"]=strategy
    return chunks