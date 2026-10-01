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

# 2. Install the AIE compiler (aiecc) into a local tools/ironenv/ venv
./setup_whisper.sh aiecc

# 3. Compile the ONNX for the NPU
whisper/compile_aie.sh

# 4. Run the compiled encoder on the NPU
whisper/run_npu.sh
```

### Why only the encoder?

Whisper = **encoder** (fixed-shape transformer over the 30 s mel spectrogram)
+ **decoder** (autoregressive, one token at a time). The encoder is the bulk of
the compute and has a static shape — exactly what `aiecc` can target. The
autoregressive decoder is normally kept on CPU. This is the same split used by
other NPU/edge Whisper deployments.

### Installing `aiecc`

```
./setup_whisper.sh aiecc
```

This installs the **mlir-aie** (IRON) toolchain + **Peano** (llvm-aie) into a
local `tools/ironenv/` venv using your existing Python (3.14). The `aiecc`
binary ends up at `tools/ironenv/lib/python3.14/site-packages/mlir_aie/bin/aiecc`.

- **~500 MB** download (two wheels: `mlir_aie` + `llvm-aie`)
- Everything stays local under `tools/` (git-ignored)
- No `sudo` needed
- `whisper/compile_aie.sh` auto-detects `aiecc` from `tools/ironenv/`

> **Caveat:** The upstream `env_install.sh` hard-requires Python 3.12, but the
> release wheels ship cp314 variants. This script bypasses that check and
> installs the cp314 wheel directly. If you hit ABI issues, fall back to
> `sudo apt install python3.12` and re-run.

## Notes

- `artifacts/whisper/` is git-ignored (large model + compiled binaries).
- `tools/` is git-ignored (AIE compiler toolchain, ~500 MB).
- The CPU path uses `fp16=False` (CPU has no fp16).
- Model sizes: `tiny`/`base` are the practical choices for on-device work.
- The `aiecc` install uses Python 3.14 (cp314 wheels). If you encounter
  compatibility issues, install `python3.12` via apt and re-run.
