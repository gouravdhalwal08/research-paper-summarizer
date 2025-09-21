"""
utils/pdf_reader.py

Robust PDF text extraction with deterministic fallback order:
  1) pdfplumber structured extraction
  2) basic layout-aware heuristic (reading by bbox left-to-right, top-to-bottom)
  3) OCR fallback using PyMuPDF page render -> PIL image -> pytesseract

Expose: extract_text_from_pdf(path_or_bytes) -> str
"""

from typing import Union, Optional, List
import io
import logging

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _extract_with_pdfplumber(fp: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extract text using pdfplumber. This tries to preserve reading order via pdfplumber's layout.
    """
    logger.info("Trying pdfplumber extraction")
    text_parts: List[str] = []
    # pdfplumber accepts file path or file-like object
    with pdfplumber.open(fp) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                # page.extract_text() attempts to return text in a readable order
                txt = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            except Exception as e:
                logger.debug("pdfplumber page extract failed: %s", e)
                txt = ""
            if txt:
                text_parts.append(txt)
    combined = "\n\n".join(text_parts).strip()
    logger.info("pdfplumber extracted %d characters", len(combined))
    return combined


def _ocr_with_fitz(fp: Union[str, bytes, io.BytesIO]) -> str:
    """
    Render pages with PyMuPDF (fitz) to images, run pytesseract on each, and concatenate.
    This is used as a fallback for scanned PDFs or when structured extraction fails.
    """
    logger.info("Starting OCR pass with PyMuPDF + pytesseract")
    text_parts: List[str] = []
    # fitz.open accepts bytes or path
    doc = fitz.open(stream=fp, filetype="pdf") if isinstance(fp, (bytes, io.BytesIO)) else fitz.open(fp)
    zoom = 2  # render resolution multiplier
    mat = fitz.Matrix(zoom, zoom)
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            text_parts.append(ocr_text)
            logger.debug("OCR page %d produced %d chars", page_no, len(ocr_text))
        except Exception as e:
            logger.exception("OCR rendering/recognition failed on page %d: %s", page_no, e)
    doc.close()
    combined = "\n\n".join(text_parts).strip()
    logger.info("OCR extracted %d characters", len(combined))
    return combined


def _layout_aware_fix(raw_text: str) -> str:
    """
    A very simple attempt to reorganize text when extraction returns columns mixed
    or many short lines: join short lines into paragraphs and keep order.
    This is heuristic and intentionally conservative.
    """
    logger.info("Applying layout-aware cleanup")
    lines = raw_text.splitlines()
    cleaned_lines: List[str] = []
    buffer = []
    for line in lines:
        if not line.strip():
            if buffer:
                cleaned_lines.append(" ".join(buffer))
                buffer = []
            continue
        # if line is short, likely part of paragraph; if very long, treat as paragraph by itself
        if len(line) < 120:
            buffer.append(line.strip())
        else:
            if buffer:
                cleaned_lines.append(" ".join(buffer))
                buffer = []
            cleaned_lines.append(line.strip())
    if buffer:
        cleaned_lines.append(" ".join(buffer))
    return "\n\n".join(cleaned_lines).strip()


def extract_text_from_pdf(path_or_bytes: Union[str, bytes, io.BytesIO], ocr_threshold_chars: int = 200) -> str:
    """
    Extract text from a PDF with deterministic fallback order.

    Args:
        path_or_bytes: file path, bytes, or a BytesIO containing the PDF.
        ocr_threshold_chars: if extracted text length is less than this, run OCR fallback.

    Returns:
        A single string containing extracted text.
    """
    # Try pdfplumber first
    try:
        text = _extract_with_pdfplumber(path_or_bytes)
    except Exception as e:
        logger.exception("pdfplumber extraction failed: %s", e)
        text = ""

    # If extraction produced little text -> try layout-aware fix
    if text and len(text) < (ocr_threshold_chars * 2):
        logger.info("Extraction produced small text (%d chars). Applying layout-aware heuristics.", len(text))
        text_fixed = _layout_aware_fix(text)
        if len(text_fixed) > len(text):
            text = text_fixed

    # If still too little text -> OCR fallback
    if not text or len(text) < ocr_threshold_chars:
        logger.info("Falling back to OCR because extracted text length is %d", len(text))
        try:
            ocr_text = _ocr_with_fitz(path_or_bytes)
            if ocr_text and len(ocr_text) > len(text):
                logger.info("OCR produced more text (%d chars) than previous extraction (%d). Using OCR output.", len(ocr_text), len(text))
                text = ocr_text
        except pytesseract.TesseractError as t_err:
            logger.exception("Tesseract not found or failed: %s", t_err)
            raise RuntimeError("Tesseract not found or failing. Please install Tesseract OCR and ensure it's on PATH.") from t_err
        except Exception as e:
            logger.exception("OCR fallback failed: %s", e)

    final = text.strip()
    logger.info("Final extracted text length: %d", len(final))
    return final
