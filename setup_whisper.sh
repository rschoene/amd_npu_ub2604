#!/usr/bin/env bash
#
# setup_whisper.sh — Install the dependencies for the Whisper pipeline.
#
# This is separate from setup_npu.sh on purpose: it adds the ML packages and
# (optionally) the Ryzen AI SDK. It does NOT touch the NPU driver / XRT stack
# that setup_npu.sh manages.
#
# Usage:
#   ./setup_whisper.sh install    # CPU Whisper deps (torch, openai-whisper, onnx)
#   ./setup_whisper.sh rai        # Ryzen AI SDK (Miniforge + RAI) in tools/
#   ./setup_whisper.sh status     # Show pipeline readiness
#
# Note:
#   - 'install' pulls ~2 GB (CPU torch) into the existing .venv.
#   - 'rai' installs a local Miniforge + the Ryzen AI SDK into tools/. The RAI
#     SDK provides the onnxruntime build with the VitisAIExecutionProvider that
#     actually runs Whisper on the NPU. The RAI installer itself must be
#     downloaded from the AMD account portal (see README_whisper.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
TOOLS_DIR="${REPO_ROOT}/tools"
MINIFORGE="${TOOLS_DIR}/miniforge3"
RAI_ENV_NAME="ryzen-ai"

cmd="${1:-install}"

case "${cmd}" in
    install)
        if [[ ! -x "${PY}" ]]; then
            echo "ERROR: ${PY} not found. Run ./setup_npu.sh install first." >&2
            exit 1
        fi
        echo "Installing Whisper dependencies into ${VENV} ..."
        # CPU-only torch to keep the download small (NPU work is via VitisEP, not CUDA).
        "${PY}" -m pip install --upgrade pip
        "${PY}" -m pip install \
            torch --index-url https://download.pytorch.org/whl/cpu
        "${PY}" -m pip install openai-whisper onnx onnxruntime
        echo
        echo "Done. Verify with:  ${PY} whisper/status.py"
        ;;

    rai)
        # --- Install the Ryzen AI SDK (RAI) into a local tools/ tree ---------
        #
        # The RAI SDK ships a custom onnxruntime build that includes the
        # VitisAIExecutionProvider (VitisEP). That EP is what compiles the
        # Whisper ONNX subgraphs to AIE and runs them on the NPU. There is no
        # public conda/pip package for it — it comes from the RAI installer.
        #
        # This script does the parts that can be automated locally:
        #   1. Install Miniforge into tools/miniforge3 (no sudo, no ~/.bashrc)
        #   2. Create a 'ryzen-ai' conda env
        #   3. Install the demo's Python deps into that env
        #
        # It then STOPS and tells you to run the RAI installer (which needs an
        # AMD account download) into that env. See README_whisper.md.

        mkdir -p "${TOOLS_DIR}"

        # 1. Miniforge (local, no system modification)
        if [[ ! -x "${MINIFORGE}/bin/conda" ]]; then
            echo "Installing Miniforge into ${MINIFORGE} ..."
            installer="${TOOLS_DIR}/miniforge.sh"
            curl -L -o "${installer}" \
                "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
            bash "${installer}" -b -p "${MINIFORGE}"
            rm -f "${installer}"
        else
            echo "Miniforge already present at ${MINIFORGE}"
        fi

        conda="${MINIFORGE}/bin/conda"

        # 2. Create the RAI conda env (idempotent)
        echo "Ensuring conda env '${RAI_ENV_NAME}' ..."
        "${conda}" env list | awk '{print $1}' | grep -qx "${RAI_ENV_NAME}" \
            || "${conda}" create -y -n "${RAI_ENV_NAME}" python=3.12

        # 3. Install the Whisper demo's Python deps into the env
        echo "Installing Whisper demo dependencies into '${RAI_ENV_NAME}' ..."
        "${conda}" run -n "${RAI_ENV_NAME}" pip install --upgrade pip
        "${conda}" run -n "${RAI_ENV_NAME}" pip install \
            torch torchaudio transformers onnxruntime \
            huggingface_hub jiwer sounddevice soundfile

        echo
        echo "=============================================================="
        echo " Local setup done. Next, install the Ryzen AI SDK itself:"
        echo
        echo "  1. Download the RAI Linux installer from the AMD portal:"
        echo "     https://ryzenai.docs.amd.com/en/latest/inst.html"
        echo "     (RAI 1.7.1+ ships a Linux installer)"
        echo
        echo "  2. Run it, pointing it at the '${RAI_ENV_NAME}' conda env:"
        echo "     ${conda} env list   # confirm the env exists"
        echo
        echo "  3. Verify the VitisEP is available:"
        echo "     ${conda} run -n ${RAI_ENV_NAME} python -c \\"
        echo "       'import onnxruntime as ort; print(ort.get_available_providers())'"
        echo "     -> should list 'VitisAIExecutionProvider'"
        echo
        echo "  4. Run Whisper on the NPU:"
        echo "     ${conda} run -n ${RAI_ENV_NAME} python whisper/run_npu.py \\"
        echo "       --model-type whisper-base --device npu --input audio.wav"
        echo "=============================================================="
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
        echo "Usage: $0 {install|rai|status}" >&2
        exit 1
        ;;
esac
