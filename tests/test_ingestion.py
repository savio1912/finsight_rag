import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_documents

# Load
pages = load_pdf("../data/raw/AAPL_2023_10K.pdf")
print(f"\nSample page metadata: {pages[0].metadata}")
print(f"Sample text (first 300 chars):\n{pages[0].page_content[:300]}")

# Chunk with all 3 strategies
for strategy in ["fixed", "recursive", "semantic"]:
    chunks = chunk_documents(pages, strategy=strategy)
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"\n[{strategy}] {len(chunks)} chunks, avg {avg_len} chars")
    print(f"  Sample: {chunks[0].page_content[:200]}")