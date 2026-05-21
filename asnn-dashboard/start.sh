#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  ASNN Detection Dashboard — Setup & Start Script
# ═══════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       ASNN Detection Dashboard Setup             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Fix ownership if directory is owned by root ───────────
OWNER=$(stat -c '%U' "$SCRIPT_DIR" 2>/dev/null || echo "unknown")
CURRENT=$(whoami)
if [ "$OWNER" != "$CURRENT" ] && [ "$OWNER" != "unknown" ]; then
    echo "[!] Directory owned by '$OWNER', you are '$CURRENT'"
    echo "[*] Fixing ownership with sudo..."
    sudo chown -R "$CURRENT":"$CURRENT" "$SCRIPT_DIR"
    echo "[✓] Ownership fixed"
fi

# ── Check Node.js ─────────────────────────────────────────
if ! command -v node &>/dev/null; then
    echo "[!] Node.js not found. Install it first:"
    echo "    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
    echo "    sudo apt-get install -y nodejs"
    exit 1
fi

NODE_VER=$(node --version)
echo "[✓] Node.js: $NODE_VER"

# ── Install npm dependencies ──────────────────────────────
if [ ! -d "node_modules" ]; then
    echo "[*] Installing npm dependencies..."
    npm config set cache "$HOME/.npm-cache" 2>/dev/null || true
    npm install
    echo "[✓] Dependencies installed"
else
    echo "[✓] Dependencies already installed"
fi

# ── Create required dirs ──────────────────────────────────
mkdir -p models uploads public

if [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo ""
    echo "[!] No models found in ./models/  Add one like:"
    echo "      mkdir -p models/car"
    echo "      cp car.nb libnn_car.so data.yaml models/car/"
    echo ""
fi

# ── Config ────────────────────────────────────────────────
PORT=${PORT:-8080}
MODELS_DIR=${MODELS_DIR:-"$SCRIPT_DIR/models"}
DETECT_SCRIPT=${DETECT_SCRIPT:-"$SCRIPT_DIR/detect.py"}

# ── Network info ──────────────────────────────────────────
echo ""
echo "Access dashboard from any device on your network:"
hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^$' | while read ip; do
    echo "   → http://$ip:$PORT"
done || echo "   → http://localhost:$PORT"
echo ""

# ── Start ─────────────────────────────────────────────────
echo "[*] Starting on port $PORT..."
echo "[*] Models : $MODELS_DIR"
echo "[*] Script : $DETECT_SCRIPT"
echo ""

PORT="$PORT" \
MODELS_DIR="$MODELS_DIR" \
DETECT_SCRIPT="$DETECT_SCRIPT" \
node server.js
