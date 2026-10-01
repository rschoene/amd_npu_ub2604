# Running Whisper on the AMD NPU (XDNA)

This directory holds the pipeline for running **OpenAI Whisper** on the AMD
Ryzen AI NPU. It builds on the base NPU stack from `setup_npu.sh` (driver,
XRT, memlock) and adds the model side.

## How it works

The NPU path uses AMD's **Ryzen AI Software (RAI)** stack. The key piece is the
**Vitis AI Execution Provider (VitisEP)** — a custom `onnxruntime` build that
ships with RAI. You feed it the pre-quantized Whisper ONNX models; the VitisEP
partitions the graph, compiles the NPU-supported subgraphs to AIE *at runtime*,
and runs them on the NPU. **There is no manual AIE-compile step.**

```
Pre-quantized Whisper ONNX (amd/whisper-*-onnx-npu on HuggingFace)
    → onnxruntime + VitisAIExecutionProvider  (from the RAI SDK)
        → VitisEP compiles subgraphs to AIE at runtime (~15 min first run)
            → NPU executes: encoder 100%, decoder ~93%
```

This is the same flow as AMD's official
[Whisper demo](https://github.com/amd/RyzenAI-SW/blob/main/Demos/ASR/Whisper).

> **Note on an earlier approach:** a first draft of this repo tried to export
> Whisper to ONNX and compile it with `aiecc` (the mlir-aie/IRON compiler).
> That was the wrong tool — `aiecc` takes `.mlir` designs for *custom* AIE
> kernels, not standard ML models. The RAI/VitisEP path above is the supported
> way to run Whisper on the NPU.

## Two paths

| Path | What runs where | Needs | Works today? |
|------|-----------------|-------|--------------|
| **CPU** | Full Whisper in PyTorch on CPU | `openai-whisper` + `torch` | Yes, after `./setup_whisper.sh install` |
| **NPU** | Whisper encoder+decoder via VitisEP | RAI SDK (VitisEP) + NPU runtime | After `./setup_whisper.sh rai` |

## Files

| Path | Purpose |
|------|---------|
| `setup_whisper.sh` | `install` (CPU deps) / `rai` (RAI SDK) / `status` |
| `whisper/status.py` | Report which pipeline stages are ready |
| `whisper/transcribe.py` | **CPU** transcription (works immediately) |
| `whisper/run_npu.py` | **NPU** transcription via the VitisAI EP |
| `whisper/ryzen_ai-1.8.0.tgz` | RAI package (downloaded from AMD portal, git-ignored) |

## Quick start — CPU transcription (works now)

```bash
./setup_whisper.sh install                 # ~2 GB download (CPU torch)
.venv/bin/python whisper/transcribe.py audio.wav --model base
.venv/bin/python whisper/status.py         # see pipeline readiness
```

## The NPU path

```bash
# 1. Download the RAI package (AMD account login required) into whisper/
#    See "Installing the Ryzen AI SDK" below.

# 2. Install the RAI SDK (extracts the .tgz, builds its own venv in tools/)
./setup_whisper.sh rai

# 3. Run Whisper on the NPU (first run compiles for ~15 min)
source tools/ryzen_ai/venv/bin/activate
python whisper/run_npu.py --model-type whisper-small --device npu --input audio.wav
```

### Installing the Ryzen AI SDK

On **Linux**, RAI is a **`.tgz` package** (`ryzen_ai-1.8.0.tgz`) — not a conda
installer. It creates its **own Python venv** (no Miniforge/conda needed).

1. **Download** `ryzen_ai-1.8.0.tgz` from the AMD account portal (SSO-gated,
   so it can't be fetched anonymously) and place it in `whisper/`:
   - Portal: <https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen_ai-1.8.0.tgz>
   - Docs (Linux section): <https://ryzenai.docs.amd.com/en/latest/inst.html>

2. **Install** it locally:
   ```bash
   ./setup_whisper.sh rai
   ```
   This extracts the package into `tools/ryzen_ai-1.8.0/` and runs
   `install_ryzen_ai.sh -a yes -p tools/ryzen_ai/venv` (non-interactive), then
   verifies the VitisEP is present.

**Prerequisite:** RAI requires **Python 3.12.x** (it builds its venv from it).
If `python3.12` is missing:

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
```

Everything stays under `tools/` and `whisper/` (both git-ignored).

### Models

`run_npu.py` auto-downloads the NPU-optimized ONNX models from HuggingFace:

| `--model-type` | HuggingFace repo |
|----------------|------------------|
| `whisper-small` | `amd/whisper-small-onnx-npu` |
| `whisper-medium` | `amd/whisper-medium-onnx-npu` |
| `whisper-large-v3-turbo` | `amd/whisper-large-turbo-onnx-npu` |

For other sizes (e.g. `whisper-base`), pass `--encoder`/`--decoder` paths
explicitly.

### Whisper-medium note

If `whisper-medium` fails to compile on the NPU, add these flags to the
encoder's VitisEP config JSON and pass it via `--encoder-config`:

```json
{ "vaiml_config": { "optimize_level": 3, "aiecompiler_args": "--system-stack-size=512" } }
```

## Notes

- `artifacts/whisper/` is git-ignored (downloaded models + VitisEP cache).
- `tools/` is git-ignored (RAI SDK + its venv, several GB).
- `whisper/ryzen_ai-*.tgz` is git-ignored (the downloaded RAI package).
- The CPU path uses `fp16=False` (CPU has no fp16).
- First NPU run compiles the model (~15 min); later runs load from the
  VitisEP cache in `artifacts/whisper/cache/`.
- The RAI venv uses Python 3.12 (RAI's supported version), independent of the
  system Python 3.14 used by the CPU `.venv`.
