#!/usr/bin/env bash
# Set up the quantized_ft virtual environment.
#
# Usage:
#   bash install.sh
#
# Prerequisites:
#   - Python 3.11 on PATH (or set PYTHON=/path/to/python3.11)
#   - CT-CLIP repository at ../CT-CLIP  (or export CT_CLIP_ROOT=/path/to/CT-CLIP)
#   - CUDA 12.8 drivers (for the cu128 torch build)
#     On a CUDA 11.x or 12.x system substitute the matching torch index URL below.

set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/venv"

echo "=== quantized_ft install ==="
echo "Python : $("${PYTHON}" --version 2>&1)"
echo "Venv   : ${VENV_DIR}"
echo ""

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON}" -m venv "${VENV_DIR}"
    echo "Created venv at ${VENV_DIR}"
else
    echo "Venv already exists — skipping creation"
fi

source "${VENV_DIR}/bin/activate"

# ── 2. Install PyTorch (CUDA 12.8 build) ─────────────────────────────────────
# If your cluster has a different CUDA version, change the index URL:
#   CUDA 11.8 → https://download.pytorch.org/whl/cu118
#   CUDA 12.1 → https://download.pytorch.org/whl/cu121
#   CUDA 12.4 → https://download.pytorch.org/whl/cu124
echo "Installing PyTorch (cu128)..."
pip install --quiet \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── 3. Install remaining dependencies ────────────────────────────────────────
echo "Installing requirements..."
pip install --quiet -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

echo ""
echo "=== Done ==="
echo ""
echo "Activate the venv with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Set CT_CLIP_ROOT if CT-CLIP is not at ../CT-CLIP:"
echo "  export CT_CLIP_ROOT=/path/to/CT-CLIP"
