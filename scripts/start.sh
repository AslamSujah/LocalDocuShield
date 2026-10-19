#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Install from https://ollama.com/download"
  exit 1
fi

ollama serve >/tmp/ollama.log 2>&1 &
sleep 2

ollama pull gemma3:4b
ollama pull nomic-embed-text

exec streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
