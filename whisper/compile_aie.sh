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

# Locate aiecc: check PATH first, then the local tools/ironenv/ venv.
AIECC=""
if command -v aiecc >/dev/null 2>&1; then
    AIECC="aiecc"
elif [[ -x "${REPO_ROOT}/tools/ironenv/bin/aiecc" ]]; then
    AIECC="${REPO_ROOT}/tools/ironenv/bin/aiecc"
else
    # Try to find it via the venv's site-packages layout
    for sp in "${REPO_ROOT}"/tools/ironenv/lib/python3.*/site-packages/mlir_aie/bin/aiecc; do
        if [[ -x "${sp}" ]]; then
            AIECC="${sp}"
            break
        fi
    done
fi

if [[ -z "${AIECC}" ]]; then
    echo "ERROR: 'aiecc' not found." >&2
    echo "       Install it with:  ./setup_whisper.sh aiecc" >&2
    echo "       (installs into tools/ironenv/, ~500 MB download)" >&2
    exit 1
fi

# Ensure aiecc's shared libs are findable
AIECC_DIR="$(dirname "${AIECC}")"
export LD_LIBRARY_PATH="${AIECC_DIR}/../lib:${LD_LIBRARY_PATH:-}"

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
