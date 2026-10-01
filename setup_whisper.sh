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
#   ./setup_whisper.sh rai        # Install the Ryzen AI SDK (RAI) into tools/
#   ./setup_whisper.sh status     # Show pipeline readiness
#
# Note:
#   - 'install' pulls ~2 GB (CPU torch) into the existing .venv.
#   - 'rai' installs the Ryzen AI SDK, which provides the onnxruntime build with
#     the VitisAIExecutionProvider that actually runs Whisper on the NPU.
#     On Linux, RAI is a .tgz package (ryzen_ai-<ver>.tgz) that creates its own
#     Python venv. The .tgz must be downloaded from the AMD account portal
#     (SSO-gated) and placed in whisper/ before running 'rai'.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
TOOLS_DIR="${REPO_ROOT}/tools"

# Ryzen AI SDK (RAI) — Linux package
RAI_VERSION="1.8.0"
RAI_TGZ="${SCRIPT_DIR}/whisper/ryzen_ai-${RAI_VERSION}.tgz"
RAI_DL_URL="https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen_ai-${RAI_VERSION}.tgz"
RAI_WORK="${TOOLS_DIR}/ryzen_ai-${RAI_VERSION}"     # extraction dir
RAI_VENV="${TOOLS_DIR}/ryzen_ai/venv"               # final RAI venv

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
        # --- Install the Ryzen AI SDK (RAI) on Linux -------------------------
        #
        # On Linux, RAI is a .tgz package (not a conda installer). It creates
        # its own Python venv and provides the onnxruntime build with the
        # VitisAIExecutionProvider (VitisEP) that compiles the Whisper ONNX
        # subgraphs to AIE and runs them on the NPU.
        #
        # The .tgz is SSO-gated behind the AMD account portal, so it cannot be
        # fetched anonymously. This script:
        #   1. Checks for the .tgz in whisper/ (download it first if missing)
        #   2. Extracts it and runs install_ryzen_ai.sh non-interactively
        #   3. Verifies the VitisEP is present in the resulting venv

        # 1. RAI requires Python 3.12.x (it builds its own venv from it).
        if ! command -v python3.12 >/dev/null 2>&1; then
            echo "ERROR: RAI requires Python 3.12.x, but python3.12 was not found." >&2
            echo "       Install it first:" >&2
            echo "         sudo apt update && sudo apt install -y python3.12 python3.12-venv" >&2
            exit 1
        fi

        # 2. The .tgz must be present (SSO-gated download).
        if [[ ! -f "${RAI_TGZ}" ]]; then
            echo "=============================================================="
            echo " RAI ${RAI_VERSION} package not found at:"
            echo "   ${RAI_TGZ}"
            echo
            echo " Download it from the AMD account portal (requires an AMD"
            echo " account login), then place it in whisper/:"
            echo
            echo "   ${RAI_DL_URL}"
            echo
            echo " Docs: https://ryzenai.docs.amd.com/en/latest/inst.html"
            echo " (Linux section — the package is 'ryzen_ai-${RAI_VERSION}.tgz')"
            echo
            echo " Once it is in place, re-run:  ./setup_whisper.sh rai"
            echo "=============================================================="
            exit 1
        fi

        mkdir -p "${TOOLS_DIR}"

        # 3. Extract the package (idempotent).
        if [[ ! -d "${RAI_WORK}" ]]; then
            echo "Extracting RAI ${RAI_VERSION} into ${RAI_WORK} ..."
            mkdir -p "${RAI_WORK}"
            tar -xzf "${RAI_TGZ}" -C "${RAI_WORK}"
        else
            echo "RAI already extracted at ${RAI_WORK}"
        fi

        # 4. Locate the installer (top level of the extraction, or a subdir).
        installer="$(find "${RAI_WORK}" -maxdepth 2 -name 'install_ryzen_ai.sh' -print -quit 2>/dev/null || true)"
        if [[ -z "${installer}" ]]; then
            echo "ERROR: install_ryzen_ai.sh not found under ${RAI_WORK}." >&2
            echo "       The package layout may have changed; inspect ${RAI_WORK} manually." >&2
            exit 1
        fi
        installer_dir="$(cd "$(dirname "${installer}")" && pwd)"

        # 5. Install into a local venv (non-interactive: -a yes accepts the EULA).
        if [[ ! -x "${RAI_VENV}/bin/python" ]]; then
            echo "Installing RAI ${RAI_VERSION} into ${RAI_VENV} ..."
            ( cd "${installer_dir}" && ./install_ryzen_ai.sh -a yes -p "${RAI_VENV}" )
        else
            echo "RAI venv already present at ${RAI_VENV}"
        fi

        # 6. Verify the VitisEP is available.
        echo
        if "${RAI_VENV}/bin/python" -c \
            "import onnxruntime as ort; import sys; sys.exit(0 if 'VitisAIExecutionProvider' in ort.get_available_providers() else 1)" 2>/dev/null; then
            echo "OK: VitisAIExecutionProvider is available in ${RAI_VENV}."
        else
            echo "WARNING: VitisAIExecutionProvider not detected in ${RAI_VENV}." >&2
            echo "         Check the install output above." >&2
        fi

        echo
        echo "=============================================================="
        echo " RAI ${RAI_VERSION} installed. To run Whisper on the NPU:"
        echo
        echo "   source ${RAI_VENV}/bin/activate"
        echo "   source /opt/xilinx/xrt/setup.sh   # if present (XRT utils)"
        echo "   python whisper/run_npu.py --model-type whisper-small \\"
        echo "       --device npu --input audio.wav"
        echo
        echo " (First NPU run compiles the model for ~15 min.)"
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
