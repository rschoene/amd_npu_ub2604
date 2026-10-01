#!/usr/bin/env bash
#
# run_npu.sh — Run the compiled Whisper encoder on the NPU via xrt-runner.
#
# Mirrors the GEMM benchmark flow in the main README: xrt-runner executes the
# compiled AIE binary (.xclbin + .elf) described by a recipe/profile JSON.
#
# Prerequisites:
#   1. ./setup_npu.sh install   (driver, XRT, memlock)
#   2. scripts/whisper/export_onnx.py
#   3. scripts/whisper/compile_aie.sh
#
# Usage:
#   scripts/whisper/run_npu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AIE_DIR="${REPO_ROOT}/artifacts/whisper/aie"

if ! command -v xrt-runner >/dev/null 2>&1; then
    echo "ERROR: 'xrt-runner' not found. Run ./setup_npu.sh install first." >&2
    exit 1
fi

if [[ ! -d "${AIE_DIR}" || -z "$(ls -A "${AIE_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: No compiled AIE artifacts in ${AIE_DIR}." >&2
    echo "       Run scripts/whisper/compile_aie.sh first." >&2
    exit 1
fi

cd "${AIE_DIR}"

# The recipe/profile JSON are produced by the AIE compiler. If they are not
# present, list the directory so you can point xrt-runner at the right files.
if [[ ! -f recipe_whisper.json ]]; then
    echo "No recipe_whisper.json found. Contents of ${AIE_DIR}:"
    ls -la
    echo
    echo "Run xrt-runner with the recipe/profile the compiler emitted, e.g.:"
    echo "  xrt-runner --recipe <recipe.json> --profile <profile.json> --dir . --report -"
    exit 1
fi

echo "Running Whisper encoder on the NPU ..."
xrt-runner --recipe recipe_whisper.json --profile profile_whisper.json \
           --dir . --report -
