#!/usr/bin/env python3
"""
transcribe.py — Transcribe audio with Whisper on the CPU (works today).

This is the immediately-usable path: it runs the full Whisper model in
PyTorch on the CPU. The NPU path (encoder offload) is a separate pipeline —
see README_whisper.md.

Usage:
  .venv/bin/python scripts/whisper/transcribe.py audio.wav
  .venv/bin/python scripts/whisper/transcribe.py audio.mp3 --model base --language en
"""

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe audio with Whisper (CPU)")
    ap.add_argument("audio", help="Path to an audio file (wav/mp3/m4a/...)")
    ap.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model size (default: base)")
    ap.add_argument("--language", default=None,
                    help="Force a language code (e.g. 'en'); default: auto-detect")
    ap.add_argument("--task", default="transcribe",
                    choices=["transcribe", "translate"],
                    help="transcribe (default) or translate-to-English")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write the .txt/.srt/.vtt outputs")
    args = ap.parse_args()

    if not os.path.isfile(args.audio):
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper is not installed. Run ./setup_whisper.sh install first.",
              file=sys.stderr)
        return 1

    print(f"Loading Whisper '{args.model}' ...")
    model = whisper.load_model(args.model)

    print(f"Transcribing {args.audio} ...")
    t0 = time.time()
    result = model.transcribe(
        args.audio,
        language=args.language,
        task=args.task,
        fp16=False,  # CPU: disable fp16
    )
    dt = time.time() - t0

    print(f"\nDetected language: {result.get('language', '?')}")
    print(f"Elapsed: {dt:.1f} s")
    print("\n--- Transcript ---")
    print(result["text"].strip())

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_results(result, audio_path=args.audio, output_dir=args.output_dir)
        print(f"\nSaved results to {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
