import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_reader import extract_text_from_pdf

text = extract_text_from_pdf("data/sample.pdf")
print(text[:1000])
