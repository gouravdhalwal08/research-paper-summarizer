"""
utils/summarizer.py

Hierarchical summarization and simple wrapper for HuggingFace transformers.
Provides:
- Summarizer class for chunked summarization
- chunk_text() function
- optional summarize_text() simple function
"""

from typing import List, Optional, Any
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def chunk_text(text: str, tokenizer: Any, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Chunk text into sliding windows of tokens using tokenizer.
    """
    if chunk_size <= 0 or overlap >= chunk_size or overlap < 0:
        raise ValueError("Invalid chunk_size or overlap")

    enc = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(enc)
    if total_tokens == 0:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        token_ids = enc[start:end]
        decoded = tokenizer.decode(token_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        chunks.append(decoded.strip())
        if end == total_tokens:
            break
        start += step
    return chunks


class Summarizer:
    """
    Summarizer class: lazy loads model, summarization of chunks, final aggregation.
    """

    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def load_model(self, model_name: str = "t5-small", device: int = -1, use_fast_tokenizer: bool = True) -> None:
        """
        Lazy load tokenizer, model, and pipeline.
        """
        if self.model_name == model_name and self.pipeline is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipeline = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=device)
        self.model_name = model_name

    def summarize_chunks(self, chunks: List[str], batch_size: int = 4, max_length: int = 150, min_length: int = 30) -> List[str]:
        """
        Summarize list of chunks in batches.
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        summaries = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            with torch.no_grad():
                results = self.pipeline(batch, max_length=max_length, min_length=min_length, truncation=True)
            for r in results:
                summaries.append(r.get("summary_text", "").strip())
        return summaries

    def aggregate_summaries(self, chunk_summaries: List[str], final_summary_max_length: int = 200, final_summary_min_length: int = 50) -> str:
        """
        Aggregate chunk summaries into a final summary.
        """
        if not chunk_summaries:
            return ""
        if not self.pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        concat = "\n\n".join(chunk_summaries)
        if len(concat.split()) <= final_summary_min_length:
            return concat.strip()

        with torch.no_grad():
            result = self.pipeline(concat, max_length=final_summary_max_length, min_length=final_summary_min_length, truncation=True)
        return result[0].get("summary_text", "").strip()


def summarize_text(text: str, max_words: int = 200) -> str:
    """
    Optional simple function for one-shot summarization (short text).
    """
    if not text or len(text.strip()) == 0:
        return "⚠️ No text provided."
    summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer_pipeline(text, max_length=max_words, min_length=int(max_words*0.3), do_sample=False)
    return result[0]["summary_text"]
