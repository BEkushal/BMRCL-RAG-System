# RAG-based PDF Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** system using:

- PDF text extraction and chunking
- Local embeddings with sentence-transformers
- External LLM API call to your API at `api.euron.one` (model: gpt-5.3-instant)
- FastAPI backend to serve question answering requests

---

## Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/BEkushal/BMRCL-RAG-System.git
cd Assignment_Aarav_Solutions
```

2. **Install dependencies** (preferably in a venv):  
```bash
pip install -r requirements.txt
```

3. **Run the app:**  
```bash
uvicorn app.main:app --reload
```
