#!/usr/bin/env bash
# Chay moi lan Codespace khoi dong lai: dam bao Ollama + webchat dang chay nen.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT/.local-ai"
mkdir -p "$AI_DIR"
cd "$ROOT"

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Khoi dong Ollama..."
  nohup ollama serve >>"$AI_DIR/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 2
  done
fi

if ! curl -fsS http://127.0.0.1:7860 >/dev/null 2>&1; then
  PY="$AI_DIR/venv/bin/python"
  if [ ! -x "$PY" ]; then PY="python3"; fi
  echo "Khoi dong webchat..."
  nohup "$PY" "$ROOT/webchat/app.py" >>"$AI_DIR/webchat.log" 2>&1 &
fi

echo "AI Gia Su Toan webchat: http://localhost:7860"
