"""
Unit tests for utils/summarizer.py

These tests use a dummy tokenizer and monkeypatching to avoid downloading models in CI.
"""

import pytest
from typing import List

from utils import summarizer


class DummyTokenizer:
    """A very small dummy tokenizer for tests: splits on spaces and returns/accepts token ids as ints"""

    def encode(self, text: str, add_special_tokens: bool = False):
        # map each word to an integer id (word length + index)
        words = text.split()
        ids = [len(w) for w in words]
        return ids

    def decode(self, token_ids: List[int], clean_up_tokenization_spaces: bool = True, skip_special_tokens: bool = True):
        # return a placeholder text with count of tokens
        return " ".join(["[t]" for _ in token_ids])


class DummySummarizer(summarizer.Summarizer):
    def load_model(self, model_name: str = "dummy", device: int = -1, use_fast_tokenizer: bool = True):
        # override to avoid model downloads
        self.model_name = "dummy"
        self.tokenizer = DummyTokenizer()
        self.summarizer_pipeline = lambda texts, **kwargs: [{"summary_text": (t[:100] if isinstance(t, str) else str(t))} for t in (texts if isinstance(texts, list) else [texts])]


def test_chunk_text_token_counts():
    dummy = DummyTokenizer()
    text = "a " * 1200  # 1200 words
    chunks = summarizer.chunk_text(text, dummy, chunk_size=256, overlap=32)
    # Expect multiple chunks and overlap effect
    assert len(chunks) > 1
    # each decoded chunk should be non-empty
    assert all(isinstance(c, str) and c for c in chunks)


def test_summarizer_on_dummy(monkeypatch):
    s = DummySummarizer()
    s.load_model()
    long_text = "word " * 1000
    chunks = summarizer.chunk_text(long_text, s.tokenizer, chunk_size=200, overlap=20)
    chunk_summaries = s.summarize_chunks(chunks, batch_size=2, max_length=50, min_length=5)
    assert len(chunk_summaries) == len(chunks)
    final = s.aggregate_summaries(chunk_summaries, final_summary_max_length=100)
    assert isinstance(final, str)
    assert final  # non-empty
