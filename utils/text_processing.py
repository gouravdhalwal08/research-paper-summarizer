import re

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Fix spacing issues
    - Remove references section
    """
    # Replace multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove 'References' section if present
    text = re.split(r'References|REFERENCES|Bibliography', text)[0]

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks for LLM input.
    
    Args:
        text (str): Cleaned text
        chunk_size (int): Words per chunk
        overlap (int): Overlapping words to maintain context
    
    Returns:
        list: List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks
