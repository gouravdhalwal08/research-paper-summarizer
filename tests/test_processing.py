import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_reader import extract_text_from_pdf
from utils.text_processing import clean_text, chunk_text

# Step 1: PDF se text extract
raw_text = extract_text_from_pdf("data/sample.pdf")

# Step 2: Clean text
cleaned = clean_text(raw_text)
print("\n--- Cleaned Text (First 500 chars) ---\n")
print(cleaned[:500])

# Step 3: Chunking
chunks = chunk_text(cleaned, chunk_size=200, overlap=30)
print("\n--- Total Chunks:", len(chunks))
print("First Chunk:\n", chunks[0][:300])
