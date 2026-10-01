#!/usr/bin/env python3
"""
export_onnx.py — Export the Whisper *encoder* to ONNX.

Why only the encoder?
  Whisper = encoder (fixed-shape transformer over the 30 s mel spectrogram)
           + decoder (autoregressive, one token at a time).
  The encoder is the bulk of the compute and has a static shape, which is what
  the AIE compiler (`aiecc`) can target. The autoregressive decoder is normally
  kept on CPU. This script exports the encoder; the decoder stays in PyTorch.

Input : log-mel spectrogram, shape (batch, 80, 3000)  [30 s @ 16 kHz]
Output: encoder features, shape (batch, 1500, d_model)

Usage:
  .venv/bin/python scripts/whisper/export_onnx.py --model base
  .venv/bin/python scripts/whisper/export_onnx.py --model tiny --out-dir artifacts/whisper
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the Whisper encoder to ONNX")
    ap.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model size (default: base)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: artifacts/whisper)")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version (default: 17)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed. Run ./setup_whisper.sh install first.")
        return 1
    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper is not installed. Run ./setup_whisper.sh install first.")
        return 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = args.out_dir or os.path.join(repo_root, "artifacts", "whisper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "whisper_encoder.onnx")

    print(f"Loading Whisper '{args.model}' ...")
    model = whisper.load_model(args.model)
    model.eval()

    # Dummy input: 1 batch, 80 mel bins, 3000 timesteps (30 s).
    dummy = torch.zeros(1, 80, 3000)
    with torch.no_grad():
        out = model.encoder(dummy)
    print(f"Encoder output shape: {tuple(out.shape)}")

    print(f"Exporting to ONNX (opset {args.opset}) ...")
    torch.onnx.export(
        model.encoder,
        (dummy,),
        out_path,
        input_names=["log_mels"],
        output_names=["encoder_out"],
        dynamic_axes={
            "log_mels": {0: "batch"},
            "encoder_out": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.1f} MB)")
    print("\nNext step: compile for the NPU with")
    print("  scripts/whisper/compile_aie.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
