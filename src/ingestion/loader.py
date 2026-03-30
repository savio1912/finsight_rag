from pathlib import Path
from typing import List
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document

def extract_metadata_from_filename(file_name: str) -> dict:
    """
    Extract company and year from a standardized filename.
    Expected format: TICKER_YEAR_FORMTYPE.pdf
    Example:         AAPL_2023_10K.pdf
    """
    stem=Path(file_name).stem  # Get filename without extension
    parts=stem.split('_')
    return {
        "ticker": parts[0] if len(parts) > 0 else "UNKNOWN",
        "year": parts[1] if len(parts) > 1 else "UNKNOWN",
        "form": parts[2] if len(parts) > 2 else "UNKNOWN"
    }

def clean_text(text: str) -> str:
    """
    Clean raw PDF text. SEC filings have a lot of noise:
    - Page headers/footers repeated on every page
    - Multiple consecutive whitespace characters
    - Null bytes and weird unicode from table extraction
    """
    # Remove null bytes
    text = text.replace("\x00", "")
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def load_pdf(file_path: str | Path) -> List[Document]:
    """
    Load a single PDF and return a list of LangChain Documents,
    one per page, with metadata attached to each.

    Why per-page Documents?
    We preserve page-level metadata here. The chunker will later
    split these further, but will inherit this metadata so we
    always know which page a chunk came from.
    """

    file_path=Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"Unsupported file type: {file_path.suffix}")  
    loader=PyMuPDFLoader(str(file_path))
    pages=loader.load()
    # Extract metadata from filename and attach to every page
    file_metadata=extract_metadata_from_filename(file_path.name)
    cleaned_pages=[]
    for page in pages:
        cleaned_text=clean_text(page.page_content)
        if len(cleaned_text) <100:
            continue  # Skip very short pages (likely noise)
        # Attach metadata to the cleaned page
        page.metadata.update({
            'ticker': file_metadata["ticker"],
            'year': file_metadata["year"],
            'form_type': file_metadata["form"],
            'source': file_path.name
        })
        page.page_content=cleaned_text
        cleaned_pages.append(page)
        print(f"Loaded {len(cleaned_pages)} pages from {file_path.name}", end="\r")
        return cleaned_pages
