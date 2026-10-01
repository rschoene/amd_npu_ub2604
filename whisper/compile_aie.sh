#!/usr/bin/env bash
#
# compile_aie.sh — Compile the exported Whisper ONNX into AIE binaries
#                  (.xclbin + .elf) that the NPU can execute.
#
# This is the step that requires the AIE compiler (`aiecc`) from AMD's XDNA /
# Vitis software stack. It is intentionally NOT installed by setup_npu.sh.
#
# The exact aiecc flags are version-specific; this script encodes the standard
# flow and fails fast with a clear message if aiecc is missing.
#
# Usage:
#   scripts/whisper/compile_aie.sh [path/to/whisper_encoder.onnx]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ONNX="${1:-${REPO_ROOT}/artifacts/whisper/whisper_encoder.onnx}"
OUT_DIR="${REPO_ROOT}/artifacts/whisper/aie"

if ! command -v aiecc >/dev/null 2>&1; then
    echo "ERROR: 'aiecc' not found on PATH." >&2
    echo "       The AIE compiler ships with AMD's XDNA / Vitis software stack." >&2
    echo "       Install it first (see README_whisper.md), then re-run this script." >&2
    exit 1
fi

if [[ ! -f "${ONNX}" ]]; then
    echo "ERROR: ONNX model not found: ${ONNX}" >&2
    echo "       Export it first: .venv/bin/python scripts/whisper/export_onnx.py" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Compiling ${ONNX} -> ${OUT_DIR}"
# Standard AIE compile flow. Adjust --target / device flags to match your
# XDNA compiler release and Strix (strx) target.
aiecc \
    --onnx "${ONNX}" \
    --target strx \
    --output-dir "${OUT_DIR}" \
    --quantize int8

echo
echo "Compiled artifacts in ${OUT_DIR}:"
ls -la "${OUT_DIR}"
echo
echo "Next step: run on the NPU with  scripts/whisper/run_npu.sh"
