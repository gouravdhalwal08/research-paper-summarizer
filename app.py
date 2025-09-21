"""
app.py - Streamlit UI for Research Paper Summarizer
"""

from typing import Optional
import io
import logging
import time
import streamlit as st
from utils.pdf_reader import extract_text_from_pdf
from utils.summarizer import Summarizer, chunk_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Research Paper Summarizer", layout="wide")


@st.cache_resource
def get_summarizer(model_name: str):
    s = Summarizer()
    s.load_model(model_name=model_name, device=-1)
    return s


def main() -> None:
    st.title("📄 Research Paper Summarizer (Local, CPU-only)")

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Model", ["t5-small", "sshleifer/distilbart-cnn-12-6"])
        chunk_size = st.number_input("Chunk size (tokens)", 128, 2048, 512, 64)
        overlap = st.number_input("Overlap (tokens)", 0, chunk_size-1, 64, 8)
        chunk_summary_max_length = st.number_input("Chunk summary max length", 16, 400, 130, 10)
        final_summary_max_length = st.number_input("Final summary max length", 50, 600, 200, 10)
        batch_size = st.number_input("Batch size", 1, 16, 4)
        run_button = st.button("Start Summarization")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        raw_bytes = uploaded_file.read()
        st.info(f"Uploaded: {uploaded_file.name} ({len(raw_bytes)} bytes)")

        try:
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_pdf(io.BytesIO(raw_bytes))
            if not extracted_text:
                st.warning("No text extracted.")
            else:
                st.success(f"Extracted text length: {len(extracted_text)}")
                if st.checkbox("Show extracted text preview (first 5000 chars)"):
                    st.text_area("Preview", value=extracted_text[:5000], height=300)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            return

        if run_button:
            summarizer = get_summarizer(model_name)
            tokenizer = summarizer.tokenizer

            with st.spinner("Creating chunks..."):
                chunks = chunk_text(extracted_text, tokenizer, chunk_size=chunk_size, overlap=overlap)
            st.info(f"{len(chunks)} chunks created.")

            all_chunk_summaries = []
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total = len(chunks)

            try:
                for i in range(0, total, batch_size):
                    batch = chunks[i:i + batch_size]
                    progress_text.text(f"Summarizing chunks {i+1}-{min(i+batch_size,total)} of {total}...")
                    summaries = summarizer.summarize_chunks(batch, batch_size=1, max_length=chunk_summary_max_length)
                    all_chunk_summaries.extend(summaries)
                    progress_bar.progress(min(1.0, (i+len(batch))/total))
                    time.sleep(0.05)
            except Exception as e:
                st.error(f"Chunk summarization failed: {e}")
                return

            st.success("Chunk summarization completed.")
            st.write("### Chunk summaries (preview)")
            for idx, cs in enumerate(all_chunk_summaries[:10], 1):
                st.markdown(f"**Chunk {idx}**: {cs}")

            with st.spinner("Aggregating summaries..."):
                final_summary = summarizer.aggregate_summaries(all_chunk_summaries, final_summary_max_length)
            st.write("## Final Summary")
            st.write(final_summary)
            st.download_button("Download summary", final_summary, file_name=f"{uploaded_file.name}_summary.txt")
    else:
        st.info("Upload a PDF to begin.")


if __name__ == "__main__":
    main()
