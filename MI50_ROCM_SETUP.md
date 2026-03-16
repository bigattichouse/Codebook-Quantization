# AMD Radeon Instinct MI50 32G — Setup Guide

**Hardware**: Radeon Instinct MI50, 32 GB HBM2, gfx906 (Vega20)
**Short answer**: Yes, this project works on the MI50.  The 32 GB VRAM is
large enough to run Qwen3-7B and likely Qwen2.5-14B fully compressed.

---

## Why it works

Our CUDA kernels use only standard CUDA primitives that HIP/ROCm translates
automatically via the `hipify` toolchain:

| Construct used | HIP equivalent | Notes |
|----------------|---------------|-------|
| `__global__`, `__device__` | identical | core CUDA vocabulary |
| `blockIdx`, `threadIdx`, `blockDim` | identical | grid/block indexing |
| `__syncthreads()` | identical | barrier sync |
| `#include <cuda.h>` | translated to `hip/hip_runtime.h` | done by hipify |
| `torch.utils.cpp_extension.load_inline` | works on ROCm builds | PyTorch handles hipification |

PyTorch's ROCm build presents the same `torch.cuda.*` API, so:
- `torch.cuda.is_available()` returns `True`
- `tensor.to('cuda')` works
- `torch.cuda.memory_allocated()` works
- Our JIT CUDA extension compiles via `hipcc` transparently

---

## ROCm version compatibility

| ROCm | gfx906 (MI50) support | PyTorch wheel |
|------|----------------------|---------------|
| 5.7  | ✅ supported          | `rocm5.7`     |
| 6.0  | ✅ supported          | `rocm6.0`     |
| 6.1  | ✅ supported          | `rocm6.1`     |
| 6.2+ | check AMD docs       | `rocm6.2`     |

MI50 (gfx906/Vega20) is a mature target; any ROCm ≥ 5.4 supports it.

---

## Installation

### 1. Install ROCm

```bash
# Ubuntu 22.04 — follow AMD official docs for your distro:
# https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

wget https://repo.radeon.com/amdgpu-install/6.1.3/ubuntu/jammy/amdgpu-install_6.1.60103-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60103-1_all.deb
sudo amdgpu-install --usecase=rocm

# Add user to render/video groups
sudo usermod -a -G render,video $USER
# Log out and back in, then verify:
rocm-smi
```

### 2. Install PyTorch with ROCm support

```bash
# Create venv (same structure as the CUDA machine)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch ROCm wheel (replace rocm6.1 with your version)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.1

# Verify
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
# Expected: True  6.1.xxxxx
```

### 3. Install project dependencies

```bash
pip install transformers accelerate safetensors numpy scipy
# Same requirements.txt as the CUDA machine — no changes needed.
pip install -r requirements.txt  # if present
```

### 4. Verify CUDA extension compiles

```bash
cd /path/to/compress
source venv/bin/activate
python3 -c "
from proofofconcept.src.gpu_accelerated_functions import GPUAcceleratedLinear
print('GPU extension loaded OK')
"
```

If this fails with a compilation error, see **Troubleshooting** below.

---

## Extra CUDA compiler flags for ROCm

The JIT extension in `gpu_accelerated_functions.py` currently passes:
```python
extra_cuda_cflags=["-O3", "--use_fast_math"]
```
`--use_fast_math` is an NVCC flag.  On ROCm it may be silently ignored or
cause a warning.  If you see compilation errors, edit the relevant line in
`proofofconcept/src/gpu_accelerated_functions.py`:

```python
# Before (NVCC-only flag may warn on ROCm):
extra_cuda_cflags=["-O3", "--use_fast_math"],

# After (safe on both NVCC and hipcc):
extra_cuda_cflags=["-O3"],
```

The performance impact is negligible for codebook lookup kernels.

---

## Memory budget on MI50 32 GB

With 32 GB HBM2 you can run models that don't fit on the P2200 (5 GB):

| Model | Packed indices | Fits in 32 GB? |
|-------|---------------|----------------|
| Qwen3.5-0.8B | ~1.6 GB | ✅ easily |
| Qwen3-1.7B   | ~3.1 GB | ✅ |
| Qwen2.5-3B   | ~4.7 GB | ✅ |
| Qwen3-7B     | ~10 GB  | ✅ with headroom |
| Qwen2.5-14B  | ~20 GB  | ✅ likely |
| Qwen3-30B    | ~45 GB  | ❌ too large |

The always-compressed representation means the packed uint8 indices are
the dominant VRAM cost, not the full fp16 weights.  Use the memory budget
test to get an exact estimate before loading:

```bash
pytest proofofconcept/tests/test_memory_budget.py -v \
    --model ~/workspace/model/Qwen3-7B -s
```

---

## Running inference

Exactly the same commands as on the P2200:

```bash
source venv/bin/activate

# Interactive chat
python proofofconcept/chat.py ~/workspace/model/Qwen3-1.7B \
    --device cuda --mode lossless

# Single prompt
python proofofconcept/chat.py ~/workspace/model/Qwen3-7B \
    --device cuda --mode balanced \
    --prompt "Write a haiku about compression" \
    --max-tokens 60
```

The device flag `--device cuda` is correct on ROCm — `torch.cuda` is the
unified API for both NVIDIA CUDA and AMD HIP in PyTorch.

---

## Running tests

```bash
# Phase 1: cache integrity (no model load, fast)
pytest proofofconcept/tests/test_cache_integrity.py -v \
    --model ~/workspace/model/Qwen3-7B

# Phase 7: GPU vs CPU kernel agreement
pytest proofofconcept/tests/test_gpu_vs_cpu_kernel.py -v \
    --model ~/workspace/model/Qwen3-7B -s

# RoPE regression
pytest proofofconcept/tests/test_rope_initialization.py -v \
    --model ~/workspace/model/Qwen3-7B -s
```

---

## Troubleshooting

### `hipcc: command not found` during JIT compile
ROCm is installed but `hipcc` is not on PATH.  Add:
```bash
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
```

### `No kernel image available for device gfx906`
The PyTorch wheel was built for a different gfx target.  Use the correct
ROCm version wheel or build PyTorch from source with `PYTORCH_ROCM_ARCH=gfx906`.

### `--use_fast_math` warning/error
Edit `gpu_accelerated_functions.py` as described above — remove the flag.

### `HIP_AVAILABLE=True` but no GPU found
This is a known false positive on machines where PyTorch is installed with
ROCm but no GPU is present.  It's harmless — `CompressedModelLoader` falls
back to CPU if `to_empty('cuda')` fails.

### Slow first run
The JIT CUDA/HIP extension is compiled on first use and cached in
`~/.cache/torch_extensions/`.  Subsequent runs skip compilation.

---

## Performance expectations

On MI50 vs P2200 (estimated):

| Metric | P2200 (CUDA, 5GB) | MI50 (ROCm, 32GB) |
|--------|-------------------|-------------------|
| Memory bandwidth | ~140 GB/s | ~1 TB/s |
| Compute (fp16) | ~20 TFLOPS | ~53 TFLOPS |
| Expected tok/s (0.8B compressed) | ~5 tok/s | ~15–30 tok/s |
| Max model size (compressed) | ~3B params | ~15B params |

The bottleneck for our codebook lookup kernel is memory bandwidth (reading
packed indices and codebook entries), so the MI50's HBM2 gives a large speedup.
