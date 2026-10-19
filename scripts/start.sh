#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking Ollama..."

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Ensure common install paths are available
export PATH="/usr/local/bin:/usr/bin:$PATH"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama binary still not found in PATH."
  exit 1
fi

echo "==> Ollama version:"
ollama --version || { echo "ERROR: ollama cannot execute."; exit 1; }

echo "==> Starting Ollama server..."
pkill -f "ollama serve" >/dev/null 2>&1 || true
nohup ollama serve >/tmp/ollama.log 2>&1 &

# Wait until API is healthy
for i in {1..30}; do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama API is ready."
    break
  fi
  sleep 1
done

echo "==> Pulling models..."
ollama pull gemma3:4b
ollama pull nomic-embed-text

echo "==> Starting Streamlit..."
exec streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"