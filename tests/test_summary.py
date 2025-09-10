import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_reader import extract_text_from_pdf
from utils.text_processing import clean_text, chunk_text
from utils.summarizer import generate_summary

# Step 1: Extract text
raw_text = extract_text_from_pdf("data/sample.pdf")

# Step 2: Clean text
cleaned = clean_text(raw_text)

# Step 3: Chunk text
chunks = chunk_text(cleaned, chunk_size=150, overlap=30)

# Step 4: Summarize first 2 chunks
print("\n--- Summary of First Chunk ---\n")
print(generate_summary(chunks[0], max_words=60))

if len(chunks) > 1:
    print("\n--- Summary of Second Chunk ---\n")
    print(generate_summary(chunks[1], max_words=60))
