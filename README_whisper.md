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
| **NPU** | Whisper encoder+decoder via VitisEP | RAI SDK (VitisEP) + NPU runtime | After `./setup_whisper.sh rai` + RAI installer |

## Files

| Path | Purpose |
|------|---------|
| `setup_whisper.sh` | `install` (CPU deps) / `rai` (Miniforge + RAI) / `status` |
| `whisper/status.py` | Report which pipeline stages are ready |
| `whisper/transcribe.py` | **CPU** transcription (works immediately) |
| `whisper/run_npu.py` | **NPU** transcription via the VitisAI EP |

## Quick start — CPU transcription (works now)

```bash
./setup_whisper.sh install                 # ~2 GB download (CPU torch)
.venv/bin/python whisper/transcribe.py audio.wav --model base
.venv/bin/python whisper/status.py         # see pipeline readiness
```

## The NPU path

```bash
# 1. Local setup: Miniforge + 'ryzen-ai' conda env + demo deps
./setup_whisper.sh rai

# 2. Install the Ryzen AI SDK itself (AMD account download — see below)
#    Point it at the 'ryzen-ai' conda env.

# 3. Verify the VitisEP is present
tools/miniforge3/bin/conda run -n ryzen-ai python -c \
    "import onnxruntime as ort; print(ort.get_available_providers())"
#    -> should list 'VitisAIExecutionProvider'

# 4. Run Whisper on the NPU (first run compiles for ~15 min)
tools/miniforge3/bin/conda run -n ryzen-ai python whisper/run_npu.py \
    --model-type whisper-small --device npu --input audio.wav
```

### Installing the Ryzen AI SDK

`./setup_whisper.sh rai` does everything that can be automated locally:

1. Installs **Miniforge** into `tools/miniforge3` (no `sudo`, no `~/.bashrc`).
2. Creates a **`ryzen-ai`** conda env (Python 3.12).
3. Installs the demo's Python deps (`torch`, `torchaudio`, `transformers`,
   `onnxruntime`, `huggingface_hub`, `jiwer`, `sounddevice`, `soundfile`).

The **RAI SDK itself** (which provides the VitisEP) is **not** on a public
conda/pip channel — it comes from the RAI installer, downloaded from the AMD
portal:

- Docs: <https://ryzenai.docs.amd.com/en/latest/inst.html>
- RAI **1.7.1+** ships a **Linux** installer (earlier releases were Windows-only).

Run the installer so it installs into the `ryzen-ai` conda env created above.
Everything stays under `tools/` (git-ignored).

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
- `tools/` is git-ignored (Miniforge + RAI SDK, several GB).
- The CPU path uses `fp16=False` (CPU has no fp16).
- First NPU run compiles the model (~15 min); later runs load from the
  VitisEP cache in `artifacts/whisper/cache/`.
- The RAI conda env uses Python 3.12 (RAI's supported version), independent of
  the system Python 3.14 used by the CPU `.venv`.
