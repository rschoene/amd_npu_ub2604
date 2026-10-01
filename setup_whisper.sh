#!/usr/bin/env bash
#
# setup_whisper.sh — Install the Python dependencies for the Whisper pipeline.
#
# This is separate from setup_npu.sh on purpose: it only adds the ML packages
# (openai-whisper, torch, onnx, onnxruntime) into the existing .venv. It does
# NOT touch the NPU driver / XRT stack.
#
# Usage:
#   ./setup_whisper.sh install       # CPU Whisper deps (torch, openai-whisper, onnx)
#   ./setup_whisper.sh aiecc         # AIE compiler (mlir-aie + Peano) in tools/ironenv/
#   ./setup_whisper.sh status        # Show pipeline readiness
#
# Note: torch + openai-whisper are large downloads (~2 GB for CPU torch).
#       aiecc pulls ~500 MB of wheels into a local tools/ironenv/ venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
TOOLS_DIR="${REPO_ROOT}/tools"
IRON_VENV="${TOOLS_DIR}/ironenv"

# mlir-aie release to install (cp314 wheel available since v1.4.3)
MLIR_AIE_VERSION="v1.4.3"
MLIR_AIE_WHEEL_URL="https://github.com/Xilinx/mlir-aie/releases/expanded_assets/${MLIR_AIE_VERSION}"
PEANO_INDEX="https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
PEANO_SPEC="llvm-aie==22.0.0.2026090701+3e93bf7b"

cmd="${1:-install}"

case "${cmd}" in
    install)
        if [[ ! -x "${PY}" ]]; then
            echo "ERROR: ${PY} not found. Run ./setup_npu.sh install first." >&2
            exit 1
        fi
        echo "Installing Whisper dependencies into ${VENV} ..."
        # CPU-only torch to keep the download small (NPU work is via AIE, not CUDA).
        "${PY}" -m pip install --upgrade pip
        "${PY}" -m pip install \
            torch --index-url https://download.pytorch.org/whl/cpu
        "${PY}" -m pip install openai-whisper onnx onnxruntime
        echo
        echo "Done. Verify with:  ${PY} whisper/status.py"
        ;;

    aiecc)
        # --- Install the AIE compiler (aiecc) into a local tools/ironenv/ venv ---
        #
        # This bypasses the upstream env_install.sh (which hard-requires
        # python3.12) and instead uses the existing Python (3.14) with the
        # matching cp314 wheels. Everything stays local under tools/.
        #
        # Known caveat: the upstream env_setup.sh is not used; instead we
        # locate aiecc directly. If you later need the full IRON Python API
        # (aie.iron), source the env manually:
        #   source tools/ironenv/bin/activate
        #   export MLIR_AIE_INSTALL_DIR="$(python -c 'import mlir_aie; print(mlir_aie.__path__[0])')"
        #   export PATH="${MLIR_AIE_INSTALL_DIR}/bin:${PATH}"
        #   export PYTHONPATH="${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}"
        #   export LD_LIBRARY_PATH="${MLIR_AIE_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

        # Find a suitable Python (prefer 3.14, fall back to 3.13/3.12)
        aie_py=""
        for candidate in python3.14 python3.13 python3.12 "${PY}"; do
            if command -v "${candidate}" >/dev/null 2>&1; then
                aie_py="${candidate}"
                break
            fi
        done
        if [[ -z "${aie_py}" ]]; then
            echo "ERROR: No suitable Python found (need 3.12+)." >&2
            exit 1
        fi
        aie_py_ver="$("${aie_py}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        echo "Using ${aie_py} (Python ${aie_py_ver}) for the AIE toolchain."

        # Create the local venv
        mkdir -p "${TOOLS_DIR}"
        if [[ ! -d "${IRON_VENV}" ]]; then
            echo "Creating venv at ${IRON_VENV} ..."
            "${aie_py}" -m venv "${IRON_VENV}"
        fi
        aie_pip="${IRON_VENV}/bin/pip"
        aie_venv_py="${IRON_VENV}/bin/python"

        echo "Upgrading pip ..."
        "${aie_venv_py}" -m pip install --upgrade pip

        # Install Peano (llvm-aie) — the per-core RISC-V compiler
        echo "Installing Peano (llvm-aie) ..."
        "${aie_pip}" install -U "${PEANO_SPEC}" -f "${PEANO_INDEX}"

        # Install mlir_aie (contains aiecc) — cp314 wheel
        echo "Installing mlir_aie ${MLIR_AIE_VERSION} (contains aiecc) ..."
        "${aie_pip}" install -U "mlir_aie" -f "${MLIR_AIE_WHEEL_URL}"

        # Locate aiecc
        aiecc_dir="$("${aie_venv_py}" -c 'import mlir_aie; print(mlir_aie.__path__[0])' 2>/dev/null || true)"
        aiecc_bin="${aiecc_dir}/bin/aiecc"

        if [[ -x "${aiecc_bin}" ]]; then
            echo
            echo "aiecc installed at: ${aiecc_bin}"
            echo
            echo "To use it, either add to PATH:"
            echo "  export PATH=\"${aiecc_dir}/bin:\$PATH\""
            echo "  export LD_LIBRARY_PATH=\"${aiecc_dir}/lib:\$LD_LIBRARY_PATH\""
            echo
            echo "Or run whisper/compile_aie.sh which auto-detects it."
        else
            echo
            echo "WARNING: aiecc not found at expected location."
            echo "         The wheel layout may differ; check:"
            echo "           ${IRON_VENV}/lib/python${aie_py_ver}/site-packages/mlir_aie/"
            exit 1
        fi
        ;;

    status)
        if [[ -x "${PY}" ]]; then
            "${PY}" "${SCRIPT_DIR}/status.py"
        else
            echo "ERROR: ${PY} not found. Run ./setup_npu.sh install first." >&2
            exit 1
        fi
        ;;

    *)
        echo "Usage: $0 {install|aiecc|status}" >&2
        exit 1
        ;;
esac
