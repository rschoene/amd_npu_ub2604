# Running Whisper on the AMD NPU (XDNA)

This directory holds the pipeline for running **OpenAI Whisper** on the AMD
Ryzen AI NPU. It builds on the base NPU stack from `setup_npu.sh` (driver,
XRT, memlock) and adds the model side.

## The honest picture

The NPU executes **compiled AIE binaries** (`.xclbin` + `.elf`), not raw ONNX
or PyTorch. So getting Whisper onto the NPU is a real pipeline, not a one-liner:

```
Whisper (PyTorch)  →  ONNX  →  aiecc (AIE compiler)  →  .xclbin/.elf  →  xrt-runner
     encoder            encoder        int8 quantize          NPU executes
```

Two things are **not** present out of the box on a fresh `setup_npu.sh` install:

1. **`aiecc`** — the AIE compiler. It ships with AMD's XDNA / Vitis software
   stack and is intentionally *not* installed by `setup_npu.sh` (large download).
2. **A prebuilt Whisper binary** — AMD's [VTD](https://github.com/Xilinx/VTD)
   archive only ships `gemm`, `resnet50`, and microbenchmarks, no Whisper.

So there are **two paths**, and you can use them independently:

| Path | What runs where | Needs | Works today? |
|------|-----------------|-------|--------------|
| **CPU** | Full Whisper in PyTorch on CPU | `openai-whisper` + `torch` | Yes, after `./setup_whisper.sh install` |
| **NPU** | Whisper *encoder* offloaded to NPU | `aiecc` + exported ONNX | After installing the AIE compiler |

## Files

| Path | Purpose |
|------|---------|
| `setup_whisper.sh` | Install the ML Python deps into `.venv` |
| `whisper/status.py` | Report which pipeline stages are ready |
| `whisper/transcribe.py` | **CPU** transcription (works immediately) |
| `whisper/export_onnx.py` | Export the Whisper encoder to ONNX |
| `whisper/compile_aie.sh` | Compile ONNX → AIE binaries (needs `aiecc`) |
| `whisper/run_npu.sh` | Run the compiled encoder on the NPU via `xrt-runner` |

## Quick start — CPU transcription (works now)

```bash
./setup_whisper.sh install                 # ~2 GB download (CPU torch)
.venv/bin/python whisper/transcribe.py audio.wav --model base
.venv/bin/python whisper/status.py         # see pipeline readiness
```

## The NPU path

```bash
# 1. Export the encoder to ONNX
.venv/bin/python whisper/export_onnx.py --model base

# 2. Install the AIE compiler (aiecc) from AMD's XDNA / Vitis stack, then:
whisper/compile_aie.sh

# 3. Run the compiled encoder on the NPU
whisper/run_npu.sh
```

### Why only the encoder?

Whisper = **encoder** (fixed-shape transformer over the 30 s mel spectrogram)
+ **decoder** (autoregressive, one token at a time). The encoder is the bulk of
the compute and has a static shape — exactly what `aiecc` can target. The
autoregressive decoder is normally kept on CPU. This is the same split used by
other NPU/edge Whisper deployments.

### Installing `aiecc`

The AIE compiler is part of AMD's XDNA / Vitis AI software stack. It is a large,
version-specific download and is deliberately left out of `setup_npu.sh`. See
AMD's XDNA documentation for the current release and the exact `aiecc` flags
for your Strix (strx) target. `whisper/compile_aie.sh` encodes the standard
flow and fails fast with a clear message if `aiecc` is missing.

## Notes

- `artifacts/whisper/` is git-ignored (large model + compiled binaries).
- The CPU path uses `fp16=False` (CPU has no fp16).
- Model sizes: `tiny`/`base` are the practical choices for on-device work.
