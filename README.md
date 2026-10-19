# LocalDocuShield 🛡️

Privacy-preserving medical/legal document summarizer and intake assistant that runs fully local using Streamlit + Ollama + TabPFN.

## Features

- Local document intake (paste text or upload PDF/TXT)
- Local LLM summarization with Gemma via Ollama
- Risk flag + action item extraction (JSON-structured)
- Local semantic search over prior summaries using embeddings
- TabPFN-based tabular anomaly scan for ledgers/vitals/billing
- Local SQLite vault (no cloud dependency)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

## Quickstart

1. Pull local models:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run app:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Notes

- All inference calls are sent to `http://localhost:11434` (Ollama local host).
- `localdocu_vault.db` is created in the project root.
- For strict compliance deployments, combine this with full-disk encryption and endpoint hardening.

## Optional Enhancements

- Add OCR for scanned PDFs (`pytesseract` + `pdf2image`)
- Replace SQLite vector logic with PostgreSQL + `pgvector`
- Add role-based access and local audit logging
