#!/bin/bash
# ──────────────────────────────────────────────────────────────
# setup_and_run.sh  —  LSTM FDC-AI  —  macOS Apple Silicon
# Usage: bash setup_and_run.sh
# ──────────────────────────────────────────────────────────────
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   LSTM FDC-AI — Trace Anomaly Detection Setup   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

command -v python3 &>/dev/null || { echo "❌ python3 not found. brew install python"; exit 1; }
echo "✅  $(python3 --version)"

[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

echo "📥  Installing packages (PyTorch first run is large ~500 MB)..."
pip install --quiet --upgrade pip
pip install --quiet -r "$DIR/requirements.txt"
echo "✅  Packages installed"

mkdir -p "$DIR/models" "$DIR/data" "$DIR/static"

echo ""
echo "🔧  Training LSTM Autoencoder (~5–8 min on M4 CPU)..."
cd "$DIR"
python train.py

echo ""
echo "🚀  Starting server..."
echo "    Dashboard:  http://localhost:8000"
echo "    Swagger:    http://localhost:8000/docs"
echo "    Ctrl+C to stop"
echo ""
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
