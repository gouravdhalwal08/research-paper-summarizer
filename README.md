# 📄 Research Paper Summarizer  

An AI-powered web application built with **Streamlit** and **Hugging Face Transformers** that automatically extracts text from research papers (PDFs) and generates concise summaries using state-of-the-art NLP models.  

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)  

---

## ✨ Features  
- 📤 Upload any PDF research paper.  
- 🔎 Automatic **text extraction** from PDFs.  
- 🧩 **Chunk-based hierarchical summarization** (handles long papers).  
- ⚡ Supports **CPU & GPU** (auto-detects CUDA for faster inference).  
- 🎛️ Interactive **Streamlit UI** with sidebar settings.  
- 💾 Download the generated summary as `.txt`.  

---

## 📂 Project Structure  

research-paper-summarizer/
├── app.py # Streamlit app (main entrypoint)
├── utils/
│ ├── pdf_reader.py # Extracts text from PDFs
│ ├── summarizer.py # Summarization logic (GPU/CPU auto-detect)
├── artifacts/ # (optional) store processed data
├── sample_papers/ # (optional) store sample PDFs
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚡ Installation  

### 1️⃣ Clone the repo  
```bash
git clone https://github.com/yourusername/research-paper-summarizer.git
cd research-paper-summarizer
2️⃣ Create virtual environment
bash
Copy code
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows (PowerShell)
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ (Optional) Enable GPU Support
Install PyTorch with CUDA inside your venv:
👉 Find the right command here

Example (for CUDA 12.1):

bash
Copy code
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Verify GPU:

bash
Copy code
python -c "import torch; print(torch.cuda.is_available())"
If True, the app will automatically use GPU.

🚀 Usage
Start the Streamlit app:

bash
Copy code
streamlit run app.py
Workflow:
Open the web app in browser (default: http://localhost:8501).

Upload a PDF research paper.

Adjust settings (model, chunk size, summary length, etc.) in the sidebar.

Click Start Summarization.

View chunk-level summaries & final summary.

Download the final summary as .txt.

⚙️ Sidebar Settings
Model: Choose summarization model (t5-small, distilbart-cnn, etc.)

Chunk size: Token count per chunk (default: 512).

Overlap: Tokens carried over between chunks (default: 64).

Chunk summary max length: Controls summary size per chunk.

Final summary max length: Controls size of final aggregated summary.

Batch size: Number of chunks processed in one go.

👉 Larger chunk size = fewer chunks but may truncate text.
👉 Higher batch size = faster summarization (but needs more GPU memory).

🖥️ Example Output
Input:
A 20-page research paper on Diabetic Retinopathy detection.

Output:

15 chunk-level summaries.

Final concise summary (~200 tokens).

Downloadable .txt summary file.

📦 Requirements
Python 3.8+

PyTorch (CPU or CUDA)

Hugging Face Transformers

Streamlit

pdfplumber / PyMuPDF (for text extraction)

Install via:

bash
Copy code
pip install -r requirements.txt
👨‍💻 Contributor
Gourav Dhalwal (Developer)

