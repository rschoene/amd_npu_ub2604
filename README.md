# AMD NPU (XDNA) on Ubuntu 26.04

A minimal, non-intrusive setup for using the **AMD Ryzen AI NPU** (XDNA / AIE
accelerator) on a Strix Point laptop. This is *not* an Intel GNA — the software
stack is AMD's **XRT (Xilinx Runtime)** + the in-kernel `amdxdna` driver.

Verified on: **AMD Ryzen AI 7 PRO 350** (Strix Point), Ubuntu 26.04.1 LTS,
kernel 7.0.0-34-generic. Measured **51 TOPS** (INT8 GEMM) via `xrt-smi validate`.

## What this repo contains

| Path | Purpose |
|------|---------|
| `setup_npu.sh` | Install / uninstall / status for the whole stack |
| `scripts/test_npu.py` | Diagnostic: memlock, device node, XRT device open |
| `artifacts/gemm/` | Prebuilt GEMM benchmark (downloaded on demand, git-ignored) |
| `whisper/` + `setup_whisper.sh` | Run Whisper (CPU now, NPU via Ryzen AI SDK / VitisEP) — see `README_whisper.md` |

Everything runs **as your user** at runtime — no `sudo` needed to use the NPU.
`sudo` is only required once, during install.

## Prerequisites

- A Strix Point (or compatible) AMD Ryzen AI machine with an NPU.
- Ubuntu 26.04 (the `amdxdna` driver ships in the mainline kernel).

Confirm the hardware is present:

```bash
lspci | grep -i "neural processing"
```

## Install

```bash
./setup_npu.sh install
```

This does, in order:

1. Loads the `amdxdna` kernel driver (and its `gpu_sched` / `amd_pmf` deps).
2. Ensures NPU firmware is present (`linux-firmware-amd-misc`).
3. Installs the XRT userspace packages from Ubuntu:
   `libxrt2`, `libxrt-npu2`, `libxrt-utils-npu`, `python3-xrt`, `libxrt-dev`.
4. Raises the **memlock** limit to 1 GB. The NPU driver `mmap`s a 64 MB region
   with `MAP_LOCKED`; the default 8 MB limit makes it fail with
   `EAGAIN`. This is set in **both** places because on modern Ubuntu systemd
   overrides PAM:
   - `/etc/security/limits.d/99-npu-memlock.conf` (PAM fallback)
   - `/etc/systemd/system/user@.service.d/99-npu-memlock.conf` (systemd, the one
     that actually applies to a GDM login)
5. Creates a local `.venv` (with `--system-site-packages`, so it can see the
   system `pyxrt` module) and installs `onnxruntime`.

> **Reboot** (or log out and back in) after install so the memlock limit applies
> to your session. Verify with `ulimit -l` → should print `1048576`.

## Verify

```bash
./setup_npu.sh status          # full stack overview
.venv/bin/python scripts/test_npu.py   # memlock + device + XRT open
```

## Run a benchmark (no compiler needed)

AMD publishes prebuilt AIE binaries for Strix in the
[Xilinx/VTD](https://github.com/Xilinx/VTD) repo. The GEMM test measures INT8
TOPS.

```bash
# 1. Fetch the GEMM artifacts (~2.5 MB)
mkdir -p artifacts/gemm && cd artifacts/gemm
base=https://github.com/Xilinx/VTD/raw/refs/heads/main/archive/strx/gemm
curl -sSL -O "$base/gemm.xclbin" -O "$base/gemm.elf" \
     -O "$base/recipe_gemm.json" -O "$base/profile_gemm.json"

# 2. Run it (as your user)
xrt-runner --recipe recipe_gemm.json --profile profile_gemm.json \
           --dir . --report -
```

Or use the higher-level validator. It wants a static archive in your home dir:

```bash
mkdir -p ~/.local/share/xrt/2.21.75/amdxdna/bins
curl -sSL -o ~/.local/share/xrt/2.21.75/amdxdna/bins/xrt_smi_strx.a \
     "https://raw.githubusercontent.com/Xilinx/VTD/2.21.75/archive/strx/xrt_smi_strx.a"

xrt-smi validate -r gemm        # → TOPS: 51.0  [PASSED]
xrt-smi validate -r latency
xrt-smi validate -r throughput
```

## Running your own models

For standard ML models (CNNs, transformers, Whisper, LLMs), the supported path
is the **Ryzen AI Software (RAI)** stack: export your model to ONNX, then run it
with the RAI's `onnxruntime` build, whose **Vitis AI Execution Provider**
compiles the NPU-supported subgraphs to AIE at runtime. No manual compiler step.
See `README_whisper.md` for a worked example (Whisper).

The lower-level `aiecc` (mlir-aie / IRON) compiler is for writing *custom* AIE
kernels in MLIR — not for deploying standard models — and is not installed here.

## Uninstall

```bash
./setup_npu.sh uninstall
```

Removes the memlock configs, the `.venv`, and (optionally) the XRT apt
packages. It does **not** touch the kernel driver or firmware.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `mmap ... failed (err=-11): Resource temporarily unavailable` | memlock too low. Reboot after install; check `ulimit -l`. |
| `/dev/accel/accel0` missing | Driver not bound. `sudo modprobe amdxdna`, then check `dmesg \| grep xdna`. |
| `pyxrt not available` in venv | Recreate venv with `--system-site-packages` (the script does this). |
| `xrt-smi` says "Archive not found" | Download the `.a` archive as shown above. |

## Sources

- Kernel driver: `amdxdna` (in-tree, `modinfo amdxdna`)
- Firmware: `linux-firmware-amd-misc` (`/lib/firmware/amdnpu/`)
- Userspace runtime: Ubuntu `xrt` packages (XRT 2.21.75)
- Prebuilt benchmark binaries: [Xilinx/VTD](https://github.com/Xilinx/VTD)
- AIE compiler / model deployment: AMD XDNA documentation (upstream)
