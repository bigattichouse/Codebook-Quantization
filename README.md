# Adaptive Codebook Compression — Proof of Concept

Always-compressed inference for transformer models: weights stay as packed
bit-indexed codebook entries at all times.  The full weight matrix is never
materialised — only the small slice needed for each matrix multiply is
decompressed on the fly, inside a CUDA kernel or C/OpenMP kernel.

The primary goal is to run inference on models that do not fit in VRAM
uncompressed, without sacrificing output correctness.

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

   Optional: add `--entropy-code` to apply Huffman entropy coding on top of
   the codebook indices.  Because the codebook is frequency-sorted (index 0 =
   most common weight value), the index distribution is highly non-uniform
   (~10.5 bits/symbol actual vs 12-13 bits fixed-width for lossless), so
   Huffman reduces on-disk and in-RAM size by roughly 1.5× with zero accuracy
   cost (bit-perfect lossless — pure entropy coding on integer indices).

2. **Load compressed** — model is instantiated on PyTorch's `meta` device
   (zero RAM cost), then `nn.Linear` / `nn.Embedding` modules are swapped for
   `AdaptiveCodebookLinear` / `AdaptiveCodebookEmbedding`.  Only norms, biases,
   and SSM scalars are loaded as exact floats.

   With `--entropy-code`, the Huffman bitstream **stays compressed in CPU RAM
   throughout inference**.  On each forward pass, each weight matrix is decoded
   from its bitstream to a transient float32 buffer, uploaded to GPU with
   `non_blocking=True`, used for one matmul, then immediately freed.  VRAM at
   rest holds only codebooks and SSM state — enabling models far larger than
   available VRAM.

3. **Inference** — CUDA kernels compute `out = x @ W` directly from the packed
   index stream and codebook, without ever building the full `W` matrix.  A
   C/OpenMP CPU fallback is also available.

---

## Installation

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Requires: PyTorch ≥ 2.4 with CUDA or ROCm, transformers ≥ 4.57, gcc (for C kernel).

For ROCm (AMD), install the ROCm wheel explicitly — the default PyPI torch is CPU-only:
```bash
./venv/bin/pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0
```

For AMD GPU (MI50 / ROCm) setup, see the MI50_ROCM_SETUP.md doc included with this package.

---

## Quick Start

### Step 1: Compress a model

```bash
./venv/bin/python compress.py ~/workspace/model/Qwen3-1.7B --mode lossless

# With Huffman entropy coding (~40% smaller on disk/RAM, same accuracy):
./venv/bin/python compress.py ~/workspace/model/Qwen3-1.7B --mode lossless --entropy-code
```

Modes: `lossless` (best quality), `balanced`, `aggressive` (smallest).
Compression is CPU-only, one-time.  Interrupted runs can be safely resumed.

`--entropy-code` applies canonical Huffman coding to the codebook index stream
(bit-perfect lossless).  The Huffman bitstream stays compressed in CPU RAM
throughout inference; each weight matrix is decoded on the fly per forward pass.
Pass `--huffman-max-params N` to cap which tensors are Huffman-encoded (default:
no limit — all tensors encoded).

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

# Enable Qwen3 chain-of-thought reasoning (off by default):
./venv/bin/python chat.py ~/workspace/model/Qwen3-1.7B --device cuda --thinking
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

### Compression ratio (Qwen3.5-9B, lossless)

| Stage                     | Disk      | RAM (inference) | vs BF16 | bits/weight |
|---------------------------|-----------|-----------------|---------|-------------|
| Original BF16             | 19.31 GB  | 19.31 GB        | —       | 16.0        |
| Codebook only             | 15.39 GB  | 15.39 GB        | 1.25×   | 12.8        |
| Codebook + Huffman        | 12.73 GB  | 12.73 GB        | 1.52×   | 10.5        |

With `--entropy-code` the Huffman bitstream is uploaded to VRAM (GPU Phase 2)
and decoded in-kernel per matmul, saving ~2.6 GB vs fixed-width indices while
keeping the stream compressed at all times — no full weight matrix is ever
materialised.

### Inference benchmark (Qwen3.5-9B, AMD MI50 32 GB, ROCm)

| Mode                             | tok/s |  VRAM peak | CPU RAM |
|----------------------------------|-------|------------|---------|
| Uncompressed GPU                 |  1.39 |  16.7 GB   |  ~1 GB  |
| Codebook GPU (lossless)          |  9.64 |  13.6 GB   |  ~3 GB  |
| Huffman GPU Phase 2 (in VRAM)    |  0.10 |  11.95 GB  | ~4.2 GB |

Notes:
- Codebook GPU is **faster** than uncompressed because it injects an optimised
  HIP kernel for the GatedDeltaNet (SSM/Mamba) layers present in Qwen3.5-9B.
- Huffman GPU Phase 2 loads in **~11 seconds** (vs minutes for pre-decoded paths)
  because the bitstream is read directly from disk with no up-front decode.
- Huffman GPU Phase 2 stores the Huffman stream in VRAM and decodes on the fly
  per matmul using a HIP kernel — saving ~2.6 GB vs fixed-width codebook indices
  at the cost of ~100× slower tok/s vs codebook-only GPU.

### Layer-level correctness (Qwen3-1.7B, lossless)

All 28 transformer layers verified: cos > 0.999 vs uncompressed forward pass.

---

## Running Tests

```bash
# Fast unit tests (no model required) — 100 tests
./venv/bin/pytest tests/test_compressed_roundtrip.py -v

# Integration tests (auto-discovers GPT-2 / Gemma / Qwen from HF cache)
./venv/bin/pytest tests/test_compressed_roundtrip.py -v --run-slow

# Legacy integration tests (require a compressed model cache)
./venv/bin/pytest tests/test_cache_integrity.py -v --model ~/workspace/model/Qwen3-1.7B
./venv/bin/pytest tests/test_rope_initialization.py -v --model ~/workspace/model/Qwen3-1.7B
./venv/bin/pytest tests/test_gpu_load_and_forward.py -v --model ~/workspace/model/Qwen3-1.7B
```

`tests/test_compressed_roundtrip.py` covers the full pipeline without requiring
a pre-compressed model: bit-pack round-trip, codebook assignment, linear/embedding
forward equality, Huffman encode/decode, Huffman layer forward, Huffman
CPU-decode-then-GPU-matmul path, stream-stays-in-CPU-RAM verification, tiny
LLaMA end-to-end compress→load→logits, RoPE buffer reinitialisation, embed-scale
detection, large-array Huffman round-trips (50M symbols, deep code distributions),
and real-model integration (GPT-2 auto-discovered from HF cache).

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
  huffman_codebook.py          — Huffman encode/decode for codebook index streams
  compressor.py                — base compressor, tensor classification
  bitpack.py                   — N-bit stream packing utilities
  rope_utils.py                — RoPE inv_freq reinitialisation (NaN/garbage detection)
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
- **Huffman RAM**: `--entropy-code` reduces both on-disk size and in-RAM size
  during inference (~40% smaller than fixed-width packed indices).  The Huffman
  bitstream stays compressed in RAM; each matmul row is decoded on-the-fly using
  a 12-bit LUT (one cache-friendly lookup per weight symbol).  Requires gcc
  (the C/OpenMP kernel); falls back to decode-at-load if gcc is unavailable.
- **Huffman cache regeneration required after 2026-03-23 fix**: a bug in
  `_build_code_lengths` (cap of 24 bits via simple truncation) produced invalid
  Huffman codes for layers whose natural tree depth exceeds 24 (e.g. large linear
  layers in Qwen3.5-9B with ~5500 unique values).  The cap is now 32 bits.  Any
  `--entropy-code` cache built before this fix must be regenerated with
  `compress.py ... --entropy-code --force`.
- **Thinking mode**: Qwen3 models generate a `<think>` block by default, adding
  many tokens before the visible response.  Thinking is disabled by default;
  pass `--thinking` to enable it.

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
