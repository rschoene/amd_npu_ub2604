#!/usr/bin/env python3
"""
status.py — Report which stages of the Whisper-on-NPU pipeline are ready.

The pipeline has four stages:

  1. Python deps   : openai-whisper, torch, onnx, onnxruntime
  2. Export        : Whisper -> ONNX  (scripts/whisper/export_onnx.py)
  3. AIE compile   : ONNX -> .xclbin/.elf  (needs `aiecc`, scripts/whisper/compile_aie.sh)
  4. NPU run       : xrt-runner executes the compiled binary

This script only *inspects* the environment; it installs nothing.

Usage:
  .venv/bin/python scripts/whisper/status.py
"""

import importlib.util
import os
import shutil
import sys


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def have_bin(name: str) -> bool:
    return shutil.which(name) is not None


def main() -> int:
    print("=" * 60)
    print("  Whisper-on-NPU pipeline status")
    print("=" * 60)

    # --- Stage 1: Python dependencies -------------------------------------
    print("\n[1] Python dependencies")
    deps = {
        "openai-whisper": "whisper",
        "torch": "torch",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
    }
    for label, mod in deps.items():
        ok = have_module(mod)
        print(f"  [{'OK ' if ok else 'MISS'}] {label}")

    # --- Stage 2: exported ONNX ------------------------------------------
    print("\n[2] Exported ONNX model")
    onnx_path = os.path.join(os.path.dirname(__file__), "..",
                             "artifacts", "whisper", "whisper_encoder.onnx")
    onnx_path = os.path.abspath(onnx_path)
    if os.path.isfile(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  [OK ] {onnx_path} ({size_mb:.1f} MB)")
    else:
        print(f"  [MISS] {onnx_path}")
        print("         Run: .venv/bin/python whisper/export_onnx.py")

    # --- Stage 3: AIE compiler -------------------------------------------
    print("\n[3] AIE compiler (aiecc)")
    aiecc_path = shutil.which("aiecc")
    if aiecc_path is None:
        # Check the local tools/ironenv/ venv
        import glob
        for pattern in ("../tools/ironenv/bin/aiecc",
                        "../tools/ironenv/lib/python3.*/site-packages/mlir_aie/bin/aiecc"):
            matches = glob.glob(os.path.join(os.path.dirname(__file__), pattern))
            if matches:
                aiecc_path = matches[0]
                break
    if aiecc_path:
        print(f"  [OK ] aiecc at {aiecc_path}")
    else:
        print("  [MISS] aiecc not found")
        print("         Install with:  ./setup_whisper.sh aiecc")
        print("         (installs mlir-aie + Peano into tools/ironenv/, ~500 MB)")

    # --- Stage 4: NPU runtime --------------------------------------------
    print("\n[4] NPU runtime")
    for tool in ("xrt-runner", "xrt-smi"):
        ok = have_bin(tool)
        print(f"  [{'OK ' if ok else 'MISS'}] {tool}")
    if have_module("pyxrt"):
        print("  [OK ] pyxrt")
    else:
        print("  [MISS] pyxrt")

    # --- Verdict ----------------------------------------------------------
    print("\n" + "=" * 60)
    cpu_ready = have_module("whisper") and have_module("torch")
    npu_ready = (have_bin("aiecc") and os.path.isfile(onnx_path)
                 and have_bin("xrt-runner"))
    if cpu_ready:
        print("  CPU transcription: READY  ->  .venv/bin/python scripts/whisper/transcribe.py <audio>")
    else:
        print("  CPU transcription: NOT READY (run ./setup_whisper.sh install)")
    if npu_ready:
        print("  NPU inference:     READY  ->  scripts/whisper/run_npu.sh")
    else:
        print("  NPU inference:     NOT READY (needs aiecc + exported ONNX)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
