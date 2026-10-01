#!/usr/bin/env python3
"""
status.py — Report which stages of the Whisper-on-NPU pipeline are ready.

The NPU path uses the Ryzen AI SDK (RAI): its onnxruntime build includes the
VitisAIExecutionProvider, which compiles the Whisper ONNX subgraphs to AIE at
runtime and runs them on the NPU. There is no manual AIE-compile step.

Stages:
  1. CPU deps      : openai-whisper + torch in .venv  (CPU transcription)
  2. NPU runtime   : xrt-smi / pyxrt  (driver + XRT, from setup_npu.sh)
  3. RAI SDK       : Miniforge + 'ryzen-ai' conda env with VitisAIExecutionProvider
  4. NPU models    : pre-quantized Whisper ONNX (auto-downloaded on first run)

This script only *inspects* the environment; it installs nothing.

Usage:
  .venv/bin/python whisper/status.py
"""

import importlib.util
import os
import shutil
import subprocess
import sys


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def have_bin(name: str) -> bool:
    return shutil.which(name) is not None


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def check_vitisep() -> bool:
    """Check whether the RAI conda env's onnxruntime has the VitisAI EP."""
    py = os.path.join(repo_root(), "tools", "miniforge3", "envs", "ryzen-ai", "bin", "python")
    if not os.path.isfile(py):
        return False
    try:
        out = subprocess.run(
            [py, "-c",
             "import onnxruntime as ort; "
             "print('VitisAIExecutionProvider' in ort.get_available_providers())"],
            capture_output=True, text=True, timeout=120,
        )
        return out.stdout.strip().endswith("True")
    except Exception:
        return False


def main() -> int:
    print("=" * 60)
    print("  Whisper-on-NPU pipeline status")
    print("=" * 60)

    # --- Stage 1: CPU dependencies ----------------------------------------
    print("\n[1] CPU dependencies (.venv)")
    for label, mod in (("openai-whisper", "whisper"), ("torch", "torch")):
        print(f"  [{'OK ' if have_module(mod) else 'MISS'}] {label}")

    # --- Stage 2: NPU runtime ---------------------------------------------
    print("\n[2] NPU runtime (driver + XRT)")
    for tool in ("xrt-smi", "xrt-runner"):
        print(f"  [{'OK ' if have_bin(tool) else 'MISS'}] {tool}")
    print(f"  [{'OK ' if have_module('pyxrt') else 'MISS'}] pyxrt")

    # --- Stage 3: RAI SDK / VitisEP ---------------------------------------
    print("\n[3] Ryzen AI SDK (VitisAIExecutionProvider)")
    miniforge = os.path.join(repo_root(), "tools", "miniforge3", "bin", "conda")
    if os.path.isfile(miniforge):
        print(f"  [OK ] Miniforge at {os.path.dirname(os.path.dirname(miniforge))}")
    else:
        print("  [MISS] Miniforge (tools/miniforge3)")
        print("         Run: ./setup_whisper.sh rai")
    if check_vitisep():
        print("  [OK ] VitisAIExecutionProvider available in 'ryzen-ai' env")
    else:
        print("  [MISS] VitisAIExecutionProvider not detected")
        print("         Install the RAI SDK into the 'ryzen-ai' env (see README_whisper.md)")

    # --- Stage 4: NPU models ----------------------------------------------
    print("\n[4] NPU Whisper ONNX models")
    print("  (auto-downloaded from HuggingFace on first run — nothing to check)")

    # --- Verdict ----------------------------------------------------------
    print("\n" + "=" * 60)
    cpu_ready = have_module("whisper") and have_module("torch")
    npu_ready = check_vitisep() and have_bin("xrt-smi")
    if cpu_ready:
        print("  CPU transcription: READY")
        print("    .venv/bin/python whisper/transcribe.py <audio.wav>")
    else:
        print("  CPU transcription: NOT READY (run ./setup_whisper.sh install)")
    if npu_ready:
        print("  NPU inference:     READY")
        print("    tools/miniforge3/bin/conda run -n ryzen-ai python whisper/run_npu.py \\")
        print("        --model-type whisper-small --device npu --input <audio.wav>")
    else:
        print("  NPU inference:     NOT READY (needs RAI SDK with VitisEP + NPU runtime)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
