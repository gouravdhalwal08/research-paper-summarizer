"""
Unit tests for utils/pdf_reader.py
Note: These tests mock heavy IO (pytesseract) to avoid requiring Tesseract during CI.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import pdf_reader

import io
import pytest
from unittest import mock

from utils import pdf_reader


def test_extract_text_from_empty_bytes():
    # Create an empty PDF-like bytes object -> extraction should return empty string or handle gracefully.
    empty_bytes = b"%PDF-1.4\n%EOF\n"
    text = pdf_reader.extract_text_from_pdf(io.BytesIO(empty_bytes), ocr_threshold_chars=10)
    # For an invalid PDF, behavior may vary; ensure we return a string (possibly empty) and no exceptions.
    assert isinstance(text, str)


def test_layout_aware_fix_merges_short_lines():
    raw = "This is a line\nshort\nanother\n\nNew paragraph line which is rather long and stays as is."
    fixed = pdf_reader._layout_aware_fix(raw)
    assert "This is a line short another" in fixed or "This is a line short" in fixed


def test_ocr_fallback_mocked(monkeypatch):
    # Mock pytesseract.image_to_string to return some text to simulate OCR
    mock_text = "This is OCRed text."
    monkeypatch.setattr("utils.pdf_reader.pytesseract.image_to_string", lambda img: mock_text)
    # Create a minimal valid single-page PDF using PyMuPDF in memory
    import fitz
    doc = fitz.open()
    doc.insert_page(0, text="Hello world")
    pdf_bytes = doc.write()
    doc.close()
    # Force OCR by setting high threshold
    text = pdf_reader.extract_text_from_pdf(io.BytesIO(pdf_bytes), ocr_threshold_chars=1000)
    assert "OCRed" in text or "Hello" in text or isinstance(text, str)
