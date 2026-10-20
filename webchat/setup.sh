#!/usr/bin/env bash
# Chay mot lan khi tao GitHub Codespace: cai Ollama, tai GGUF, tao model, cai webchat.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_DIR="$ROOT/.local-ai"
MODEL_NAME="gia-su-toan-qwen2.5-3b-q4_k_m.gguf"
MODEL_PATH="$AI_DIR/$MODEL_NAME"
REPO="tiensdattnek-ai/math-ai"
RELEASE_URL="https://github.com/$REPO/releases/download/gguf-v1/$MODEL_NAME"

mkdir -p "$AI_DIR"
cd "$ROOT"

echo "==> [1/5] Cai Ollama (neu chua co)"
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
command -v ollama >/dev/null

model_size() { stat -c%s "$MODEL_PATH" 2>/dev/null || echo 0; }
if [ "$(model_size)" -lt 1000000000 ]; then
  echo "==> [2/5] Tai file GGUF (~1.93 GB)"
  rm -f "$MODEL_PATH" "$MODEL_PATH.tmp"
  # Uu tien link Release truc tiep neu sau nay repo co ban phat hanh gguf-v1.
  if curl -fL --retry 3 -o "$MODEL_PATH.tmp" "$RELEASE_URL"; then
    mv "$MODEL_PATH.tmp" "$MODEL_PATH"
  else
    rm -f "$MODEL_PATH.tmp"
    echo "    Chua co Release truc tiep - tai tu Actions Artifact bang GitHub CLI..."
    command -v gh >/dev/null
    RUN_ID="$(gh run list --repo "$REPO" --workflow export-gguf.yml --status success --limit 1 --json databaseId --jq '.[0].databaseId')"
    if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
      echo "Khong tim thay lan Export GGUF thanh cong nao trong Actions." >&2
      exit 1
    fi
    TMP_DIR="$(mktemp -d)"
    gh run download "$RUN_ID" --repo "$REPO" --name gia-su-toan-gguf --dir "$TMP_DIR"
    FOUND="$(find "$TMP_DIR" -name "$MODEL_NAME" -type f | head -n 1)"
    if [ -z "$FOUND" ]; then
      echo "Artifact khong chua file $MODEL_NAME" >&2
      exit 1
    fi
    cp "$FOUND" "$MODEL_PATH"
    rm -rf "$TMP_DIR"
  fi
fi

echo "==> [3/5] Kiem tra dinh dang GGUF"
if [ "$(head -c 4 "$MODEL_PATH")" != "GGUF" ]; then
  echo "File tai ve khong phai GGUF hop le: $MODEL_PATH" >&2
  exit 1
fi
ls -lh "$MODEL_PATH"

echo "==> [4/5] Khoi dong Ollama va tao model gia-su-toan"
cp "$ROOT/Modelfile" "$AI_DIR/Modelfile"
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  nohup ollama serve >"$AI_DIR/ollama.log" 2>&1 &
fi
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null
(cd "$AI_DIR" && ollama create gia-su-toan -f Modelfile)

echo "==> [5/5] Cai webchat Gradio trong moi truong rieng"
if [ ! -x "$AI_DIR/venv/bin/python" ]; then
  python3 -m venv "$AI_DIR/venv"
fi
"$AI_DIR/venv/bin/pip" install -q --upgrade pip
"$AI_DIR/venv/bin/pip" install -q "gradio>=4.44,<6" requests

touch "$AI_DIR/.setup-done"
echo "SETUP HOAN TAT. Webchat se tu mo o cong 7860."
