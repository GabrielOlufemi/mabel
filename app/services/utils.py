from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# regex shit for cleaning
import re
logger = logging.getLogger(__name__)


def chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Splits text into overlapping chunks for context

    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking - textblock is empty")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )

    chunks = splitter.split_text(text)

    min_chunk_size = min(50, chunk_size // 10)  # Dynamic minimum
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > min_chunk_size]

    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")

    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Steps:
    - Remove extra whitespace
    - Remove special characters that don't add meaning
    - Normalize punctuation
    - Convert to lowercase for consistency (optional)
    """
    # Remove extra whitespace
    cleaned = " ".join(text.split())
    
    # Remove multiple punctuation (e.g., "???" -> "?")
    cleaned = re.sub(r'([?.!])\1+', r'\1', cleaned)
    
    # Remove special characters but keep basic punctuation
    # Keep: letters, numbers, spaces, ?.!,'-
    cleaned = re.sub(r'[^a-zA-Z0-9\s?.!,\'-]', '', cleaned)
    
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    
    logger.info(f"Cleaned text: '{text}' -> '{cleaned}'")
    return cleaned    
