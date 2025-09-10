from transformers import pipeline

# Lightweight model (safe for 8GB RAM)
summarizer = pipeline(
    "summarization",
    model="t5-small",
    device=-1   # Force CPU mode
)

def generate_summary(text: str, max_words: int = 60) -> str:
    """
    Generate abstractive summary using T5-small model.
    """
    if len(text.strip()) == 0:
        return "No text to summarize."

    max_tokens = min(150, max_words * 2)
    min_tokens = int(max_tokens / 2)

    summary = summarizer(
        text,
        max_new_tokens=max_tokens,   # ✅ only this
        do_sample=False
    )

    return summary[0]['summary_text']
