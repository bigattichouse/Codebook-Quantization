# Adaptive Codebook Compression — Proof of Concept

Always-compressed inference for transformer models: weights stay as packed
bit-indexed codebook entries at all times.  The full weight matrix is never
materialised — only the small slice needed for each matrix multiply is
decompressed on the fly, inside a CUDA kernel or C/OpenMP kernel.

The primary goal is to run inference on models that do not fit in VRAM
uncompressed, without sacrificing output correctness.

---

## Write Up
**Medium Article** https://medium.com/@bigattichouse/codebook-lossless-llm-compression-10-25-ram-reduction-with-bitwise-generic-packing-of-indexed-c35ba49fc2b8
**Reddit post for Discussion:** https://www.reddit.com/r/LocalLLaMA/comments/1rtbbiw/codebook_lossless_llm_compression_1025_ram/

I demonstrate an LLM compression technique based on bit-packing by trading index lookups for size because model weight value uniqueness is surprisingly low across model components (embeddings, attention, etc.), and even lower for individual layers. Lossless this is around 10–30% smaller than original, quantization/loss at standard sizes with massive increases in mathematical accuracy over traditional methods is also possible. Essentially trading RAM usage for index lookups — so a bit slower.

---

## How It Works

See **[SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)** and
**[algorithm.md](./algorithm.md)** for full technical detail.

Short version:

1. **Compress once (offline)** — scan every weight tensor, build a shared
   codebook per layer category (attention / MLP / embedding), store each
   weight as a compact bitstream of codebook indices (~13 bits typical for
   lossless mode).  Written as `.npz` files alongside the model.  CPU-only,
   never causes VRAM OOM.

2. **Load compressed** — model is instantiated on PyTorch's `meta` device
   (zero RAM cost), then `nn.Linear` / `nn.Embedding` modules are swapped for
   `AdaptiveCodebookLinear` / `AdaptiveCodebookEmbedding`.  Only norms, biases,
   and SSM scalars are loaded as exact floats.

3. **Inference** — CUDA kernels compute `out = x @ W` directly from the packed
   index stream and codebook, without ever building the full `W` matrix.  A
   C/OpenMP CPU fallback is also available.

---

## Installation

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Requires: PyTorch ≥ 2.5 with CUDA or ROCm, transformers ≥ 5.0, gcc (for C kernel).

For AMD GPU (MI50 / ROCm) setup, see the MI50_ROCM_SETUP.md doc included with this package.

---

## Quick Start

### Step 1: Compress a model

```bash
./venv/bin/python compress.py ~/workspace/model/Qwen3-1.7B --mode lossless
```

Modes: `lossless` (best quality), `balanced`, `aggressive` (smallest).
Compression is CPU-only, one-time.  Interrupted runs can be safely resumed.

### Step 2: Benchmark

```bash
./venv/bin/python benchmark.py ~/workspace/model/Qwen3-1.7B \
    --prompt "Write a haiku about data compression" \
    --tokens 30
```

Runs uncompressed GPU, compressed GPU, and compressed CPU back-to-back and
prints a summary table.

### Step 3: Interactive chat

```bash
./venv/bin/python chat.py ~/workspace/model/Qwen3-1.7B --device cuda
```

---

## Measured Results

Tested on Quadro P2200 (5 GB VRAM, CUDA 12.2), Qwen3-1.7B, lossless mode:

| Mode             | VRAM load | VRAM peak | Speed       | Notes                        |
|------------------|-----------|-----------|-------------|------------------------------|
| Uncompressed GPU | 3875 MB   | 3969 MB   | 26.3 tok/s  | cuBLAS bf16 matmul           |
| Compressed GPU   | 3175 MB   | 3270 MB   | 11.3 tok/s  | CUDA codebook kernel         |
| Compressed CPU   | 0 MB VRAM | 0 MB VRAM |  0.5 tok/s  | C/OpenMP kernel, 4.5 GB RAM  |

- **VRAM saving**: ~700 MB (18%) for lossless — consistent with 13-bit vs 16-bit packing
- **Speed overhead**: ~2.3× slower on GPU; real-world benefit is running models that
  don't fit uncompressed at all (e.g. 3B+ on a 5 GB card)
- **Output quality**: cosine similarity > 0.999 at every layer vs uncompressed baseline;
  greedy token sequences match exactly (lossless mode)

### Layer-level correctness (Qwen3-1.7B, lossless)

All 28 transformer layers verified: cos > 0.999 vs uncompressed forward pass.

---

## Running Tests

```bash
# Fast unit tests (no model required)
./venv/bin/pytest tests/ -m "not integration and not slow" -v

# Integration tests (requires a compressed model cache)
./venv/bin/pytest tests/test_cache_integrity.py -v --model ~/workspace/model/Qwen3-1.7B
./venv/bin/pytest tests/test_rope_initialization.py -v --model ~/workspace/model/Qwen3-1.7B
./venv/bin/pytest tests/test_gpu_load_and_forward.py -v --model ~/workspace/model/Qwen3-1.7B

# All integration tests
./venv/bin/pytest tests/ -v --model ~/workspace/model/Qwen3-1.7B
```

Integration tests skip automatically if the model path or cache is missing.

---

## Project Structure

```
compress.py              — one-time offline compression
chat.py                  — compressed-model interactive chat / single-prompt
benchmark.py             — three-way benchmark: uncomp GPU / compr GPU / compr CPU
compare.py               — side-by-side output comparison
uncompressed_chat.py     — baseline uncompressed inference
analyze.py               — compression statistics and analysis

src/
  adaptive_compressor.py       — two-pass compression pipeline
  model_loader.py              — meta-device load, to_empty, module replacement
  name_resolver.py             — cache↔param name mapping (handles multimodal prefix)
  rope_utils.py                — RoPE inv_freq reinitialization after meta-device load
  memory_utils.py              — RAM / VRAM accounting helpers
  compressed_modules.py        — AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
  gpu_accelerated_functions.py — CUDA kernels (linear matmul, embedding lookup)
  fast_index_manager.py        — vectorized CPU bitstream index unpacker
  compressed_matmul.c          — C/OpenMP kernel (fallback and CPU mode)
  compressed_matmul_cpu.py     — Python wrapper + gcc JIT build for C kernel
  compressor.py                — base compressor, tensor classification
  bitpack.py                   — N-bit stream packing utilities
  analyze_tensor.py            — per-tensor statistics helpers
  q8_utils.py                  — Q8 quantization utilities

tests/
  conftest.py                  — shared fixtures, --model CLI option
  test_cache_integrity.py      — Phase 1: .npz cache completeness / validity
  test_name_mapping.py         — Phase 2: cache↔param name resolution
  test_memory_budget.py        — Phase 3: VRAM budget from cache (no model load)
  test_cpu_load_and_forward.py — Phase 4: CPU load + per-layer NaN detection
  test_gpu_load_and_forward.py — Phase 6: GPU load with OOM-aware skips
  test_gpu_vs_cpu_kernel.py    — Phase 7: GPU vs CPU kernel agreement
  test_rope_initialization.py  — RoPE inv_freq regression tests
  test_inference_quality.py    — end-to-end greedy token match
  test_layer_compare.py        — per-layer cosine similarity vs uncompressed
  test_c_kernel.py             — CPU C kernel correctness + bounds
  test_openmp_kernel.py        — OpenMP parallelism correctness + timing
  layer_compare.py             — interactive layer-by-layer comparison tool
  kernel_bench.py              — CUDA kernel microbenchmarks
  speed_compare.py             — compressed vs uncompressed tok/s comparison
  ...

SYSTEM_OVERVIEW.md       — detailed technical description
algorithm.md             — algorithm pseudocode and data-flow diagrams
INFERENCE_RECOVERY_PLAN.md — diagnostic phases, root-cause taxonomy, fix log
```

---

## Known Issues / Limitations

- **Speed**: on-the-fly codebook lookup adds ~2.3× overhead vs native bf16 matmul
  on GPU.  CPU mode is ~52× slower and intended for correctness testing only.
- **Compression time**: offline compression is slow (~60 min for 1.7B, CPU-only).
- **Thinking mode**: Qwen3 models generate a `<think>` block by default, adding
  many tokens before the visible response.  Pass `enable_thinking=False` to
  suppress it for cleaner benchmarks.

---

## Environment

```
GPU  : Quadro P2200, 5 GB VRAM, CUDA 12.2
CPU  : x86-64, gcc (for C kernel JIT compile)
PyTorch  : 2.5.1+cu121
transformers : 5.2.0
Python : 3.12
```

AMD Radeon Instinct MI50 (32 GB, ROCm) is also supported — see MI50_ROCM_SETUP.md.
