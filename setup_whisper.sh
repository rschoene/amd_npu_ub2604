#!/usr/bin/env bash
#
# setup_whisper.sh — Install the Python dependencies for the Whisper pipeline.
#
# This is separate from setup_npu.sh on purpose: it only adds the ML packages
# (openai-whisper, torch, onnx, onnxruntime) into the existing .venv. It does
# NOT touch the NPU driver / XRT stack.
#
# Usage:
#   ./setup_whisper.sh install
#   ./setup_whisper.sh status
#
# Note: torch + openai-whisper are large downloads (~2 GB for CPU torch).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"

cmd="${1:-install}"

if [[ ! -x "${PY}" ]]; then
    echo "ERROR: ${PY} not found. Run ./setup_npu.sh install first to create the venv." >&2
    exit 1
fi

case "${cmd}" in
    install)
        echo "Installing Whisper dependencies into ${VENV} ..."
        # CPU-only torch to keep the download small (NPU work is via AIE, not CUDA).
        "${PY}" -m pip install --upgrade pip
        "${PY}" -m pip install \
            torch --index-url https://download.pytorch.org/whl/cpu
        "${PY}" -m pip install openai-whisper onnx onnxruntime
        echo
        echo "Done. Verify with:  ${PY} whisper/status.py"
        ;;
    status)
        "${PY}" "${SCRIPT_DIR}/status.py"
        ;;
    *)
        echo "Usage: $0 {install|status}" >&2
        exit 1
        ;;
esac
