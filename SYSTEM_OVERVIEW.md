# Adaptive Codebook Compression — System Overview

This document describes the compression and inference system implemented in this
repository, including the two-pass compression pipeline, the bitstream packing
format, and the GPU/CPU inference kernels.

---

## 1. Problem Statement

A standard model loader (e.g., HuggingFace `from_pretrained`) materializes the
entire weight tensor in RAM before any quantization step.  For a 9 B parameter
model stored as BF16 this is ~18 GB; loading it on a card with 5 GB of VRAM
fails immediately.

**Goal:** run inference on a model whose full weight tensor does not fit in VRAM
by keeping weights compressed at all times and decompressing only the small slice
needed for each matrix multiplication.

---

## 2. Compression Pipeline

Compression is a one-time offline step.  Its output (a `codebook/tensors/`
directory of `.npz` files) is loaded on every subsequent run.

### Pass 1 — Global Histogram Analysis

The compressor streams every safetensors shard and accumulates a per-BF16-value
histogram for each layer category (embedding, attention, MLP/FFN, router, SSM
core, etc.).  **Every parameter is visited exactly once** — there is no sampling.

Because BF16 has only 65 536 representable values, the histogram for even a very
large model fits in a few hundred KB.  The histogram reveals the actual *unique*
count used by each category.  If a category uses fewer unique values than there
are BF16 values, its weights can be stored as indices into a codebook rather than
as raw BF16.

For Qwen3.5-0.8B (lossless mode) Pass 1 found:

| Category   | Samples | Unique values | Bits/index |
|------------|---------|---------------|------------|
| attention  | ~300 M  | 6 837–7 848   | 13 bits    |
| mlp_ffn    | ~200 M  | similar range | 13 bits    |
| embedding  | ~33 M   | varies        | up to 15   |

For Qwen3.5-9B (lossless mode) Pass 1 found:

| Category   | Entries | Bits/index |
|------------|---------|------------|
| mlp_ffn    | 9 686   | 14 bits    |
| attention  | 8 129   | 13 bits    |
| embedding  | 7 719   | 13 bits    |
| ssm_core   | 5 393   | 13 bits    |

### Pass 2 — Streaming Compression

Each tensor is compressed independently and written to disk as a `.npz` archive
containing:

- `indices` — packed bitstream of codebook indices (see §3)
- `codebook` — float32 array of unique values (or a reference to the global
  shared codebook)
- `metadata` — original shape, dtype, bits-per-index, compression mode

Tensors that are too small to benefit (norms, bias vectors, gates) are stored
exactly ("exact" mode) as BF16 bytes.

---

## 3. Bitstream Packing

Standard integer containers waste bits.  A 13-bit index stored in a `uint16`
uses 16 bits — 3 bits wasted per parameter.  At 300 M parameters that is 112 MB
of unnecessary storage.

Instead the compressor writes indices as a **contiguous bitstream**: each index
occupies exactly `bits` bits starting at bit offset `i × bits` from the beginning
of the byte array.

**Bit-unpacking (Python, vectorized):**

```python
logical_idx = arange(0, n_elements)
bit_pos  = logical_idx * bits          # bit offset of each index
byte_pos = bit_pos >> 3                # first byte containing that index
bit_shft = bit_pos & 7                 # how many bits into that byte

# Read 3 bytes (enough for up to 15-bit index crossing a byte boundary),
# combine, shift and mask:
b0, b1, b2 = padded[byte_pos], padded[byte_pos+1], padded[byte_pos+2]
combined = b0 | (b1 << 8) | (b2 << 16)
result   = (combined >> bit_shft) & ((1 << bits) - 1)
```

The 4-byte pad at the end of the stream ensures `byte_pos + 2` is always valid.
The numpy implementation is fully vectorized (no Python loops); the HIP/CUDA kernel
(§4) reproduces the same logic on-device.

**Storage savings example (13-bit vs uint16):**

```
300 M params × 13 bits = 487.5 MB
300 M params × 16 bits = 600.0 MB   (uint16)
Savings: ~19%
```

---

## 4. Inference Kernels

Three backends are available, selected automatically at runtime:

| Backend | When used | Entry point |
|---|---|---|
| **ROCm/HIP standalone** | AMD GPU (preferred) | `rocm/libcompressed_kernel.so` via ctypes |
| **CUDA/NVCC JIT** | NVIDIA GPU | PyTorch `load_inline`, cached after first compile |
| **CPU OpenMP** | No GPU / fallback | `compressed_matmul.c` compiled by gcc at import |

All backends expose the same Python interface through `GPUAcceleratedLinear` and
`GPUAcceleratedEmbedding`.  Both **compressed** and **uncompressed** (plain float)
weights are supported — use `from_weight()` for the uncompressed path.

### 4a. Compressed linear kernel

```
Grid(T × M),  Block(256 threads)
```

Each block handles one output element `out[tok, ofeat]`.  Threads reduce over K:

```
for j in range(tid, K, 256):
    idx  = unpack_idx(packed, ofeat * K + j, bits)
    acc += x[tok, j] * codebook[idx]
out[tok, ofeat] = reduce_sum(acc)
```

The codebook (~30 KB for 8192 entries) stays resident in L2/L3 cache.
No weight matrix is ever materialised.  A 1-D grid avoids the 65 535 limit
on `grid.y` for large `M` (e.g. `vocab_size = 248 320` on Qwen3.5-9B).

### 4b. Raw (uncompressed) linear kernel

Identical grid/block layout.  Inner loop reads directly from the weight matrix:

```
acc += x[tok, j] * weight[ofeat, j]
```

Use `GPUAcceleratedLinear.from_weight(name, weight_tensor, shape)` to create
one of these from a standard float weight tensor.

### 4c. Embedding kernels

```
Grid(T, ceil(H / 128)),  Block(128 threads)
```

Compressed: `out[tok, h] = codebook[ unpack_idx(packed, token_id * H + h, bits) ]`
Raw:        `out[tok, h] = weight[token_id, h]`

Both f32 and bf16 I/O are supported at the HIP level.  Accumulation is always
f32 internally.

### 4d. Standalone C library (`rocm/libcompressed_kernel.so`)

The HIP library has **no PyTorch or Python dependency** and can be embedded in
any C/C++ project:

```c
#include "compressed_kernel.h"

// Upload weights once
float* d_cb = ck_upload_codebook(codebook_f32, C);
uint8_t* d_pk = ck_upload_packed(packed_bytes, nbytes);
// Or for uncompressed:
float* d_w = ck_upload_weights_f32(weight_f32, M, K);

// Run forward pass (choice of compressed or raw, f32 or bf16)
ck_linear_f32(d_x, d_pk, d_cb, d_out, T, M, K, C, bits, stream);
ck_linear_raw_f32(d_x, d_w, d_out, T, M, K, stream);
```

Build: `make -C proofofconcept/rocm`
Test:  `make -C proofofconcept/rocm test`   (14 correctness tests, all PASS)

The Makefile auto-detects the GPU architecture via `rocminfo` and handles
ROCm/PyTorch version conflicts via `HIP_LIB_DIR` and `HIP_COV` overrides.

---

## 5. Memory-Safe Model Loading

Model instantiation follows this sequence to avoid large transient allocations:

1. **Meta device** — `AutoModelForCausalLM.from_config(..., device='meta')`.
   Creates the full model graph with zero bytes of real memory.
2. **`to_empty(cpu)`** — allocates uninitialized storage **on CPU** for every
   parameter and buffer.  Using CPU here avoids a transient VRAM spike equal to
   the full uncompressed model size (which can exceed VRAM capacity for 9B+ models
   even when the compressed footprint comfortably fits).
3. **Exact weight loading** — norm layers, small tensors, and "exact"-mode
   tensors are streamed in from disk.
4. **Compressed module replacement** — `nn.Linear` / `nn.Embedding` layers are
   replaced with `AdaptiveCodebookLinear` / `AdaptiveCodebookEmbedding` modules.
   Each replacement uploads its packed indices and codebook directly to GPU VRAM
   at creation time, independent of where the model shell lives.
5. **Shell → GPU** — the remaining small parameters (norms, biases, RoPE buffers)
   are moved to the inference device via `model.to(device)`.  The compressed
   modules are unaffected (their data is already on GPU).

At inference time the forward pass decompresses only the slice of weights needed
for the current matmul; the full weight matrix is never resident in memory.

### Parallel tensor preloading

The 775 compressed tensors for Qwen3.5-9B total ~15 GB on disk as
gzip-compressed `.npz` files.  Loading them sequentially on demand (once in
exact-weight loading, again in module replacement) took >10 minutes.

The loader now preloads **all tensors in parallel** via an 8-thread
`ThreadPoolExecutor` before the loading pipeline starts.  Subsequent accesses
are pure dict lookups.  The preloaded cache is freed after module replacement
to recover ~15 GB of RAM.

---

## 6. Measured Results

### 6a. Qwen3.5-0.8B (Quadro P2200, 5 GB VRAM, CUDA)

| Mode         | Load time | RAM    | VRAM   | Speed      | Correctness |
|--------------|-----------|--------|--------|------------|-------------|
| Uncompressed | —         | —      | 1.6 GB | 17.5 tok/s | baseline    |
| Compressed   | ~58 s*    | 4.2 GB | 1.6 GB | 5.2 tok/s  | cos > 0.999 |

\*First run compiles and caches the CUDA extension (~30–60 s).  Subsequent runs
skip compilation.

**Layer-by-layer comparison** (greedy decode, all 24 layers):
- Cosine similarity vs. uncompressed: > 0.999 for every layer
- Greedy 5-token prediction: exact match

### 6b. Qwen3.5-9B (AMD MI50, 32 GB VRAM, ROCm 6.0)

Prompt: *"Write a haiku about data compression"*

| Mode                       | Load time | RAM     | VRAM    | Speed          |
|----------------------------|-----------|---------|---------|----------------|
| Uncompressed (PyTorch)     | 7.2 s     | 1.5 GB  | 17.9 GB | 1.4 tok/s      |
| Lossless compressed (HIP)  | 18.2 s    | 16.7 GB | 14.6 GB | **8.6 tok/s**  |
| Uncompressed HIP (`--mode uncompressed`) | 7.6 s | 1.5 GB | 17.9 GB | **13.3 tok/s** |

**Key findings:**

- **Compressed HIP** is **6× faster** than stock PyTorch/ROCm uncompressed, because
  HuggingFace's PyTorch fallback for SSM layers (GatedDeltaNet) is very slow without
  the optional `flash-linear-attention` library, and our custom HIP kernel bypasses it.
- **Uncompressed HIP** (`--mode uncompressed`) is **54% faster** than compressed when
  the model fits in VRAM.  Sequential bf16 reads from a contiguous weight matrix have
  better cache utilisation than the compressed kernel's bitpack → scattered codebook
  lookup pattern.  Compression wins only when the model's bf16 weights would exceed
  available VRAM.
- The uncompressed mode uses `ck_linear_raw_bf16` / `ck_embedding_raw_bf16` — the
  same HIP library as compressed inference, but with full bf16 weight matrices instead
  of packed indices.

**When to use each mode:**

| Model fits in VRAM? | Recommended mode  | Why |
|---------------------|-------------------|-----|
| Yes (≤ VRAM)        | `--mode uncompressed` | Fastest: sequential bf16 reads |
| No (> VRAM)         | `--mode lossless`  | Only option; 6× faster than PyTorch |

**Lossless mode** uses ~13-bit codebooks (5 393–9 686 entries per category) which
achieve 0 MSE by construction — the codebook contains every unique BF16 value
that appears in the model.

---

## 7. Hardware and Software

```
# Original development (§6a results)
GPU  : Quadro P2200, 5 GB VRAM (Pascal, CUDA 12.2)
PyTorch : 2.5.1+cu121

# Current testing (§6b results)
GPU  : AMD Instinct MI50, 32 GB HBM2 (Vega20, gfx906)
PyTorch : 2.4.1+rocm6.0   (torch.version.hip = "6.0.32830")
hipcc   : ROCm 7.2.0  (system)  — version mismatch handled automatically
RAM  : 125 GB

# Common
transformers : 5.3.0   (requires >=4.57.0 for qwen3_5 model type)
Python  : 3.12
HIP ext cache : ~/.cache/torch_extensions/py312_cu121/compressed_matmul_v3/
```

---

## 8. Current Limitations and Known Issues

- **Speed overhead on small models**: On-the-fly index lookup adds ~3.4× latency
  per layer compared to uncompressed inference on Pascal CUDA (18–35 ms vs.
  3–11 ms per layer on 0.8B).  On ROCm with hybrid SSM models the compressed path
  is faster because HuggingFace lacks optimized SSM kernels without optional libs.
- **Compression time**: The two-pass pipeline is CPU-only and takes tens of
  minutes for large models.  Qwen3.5-0.8B took ~60 min; 9B scales ~linearly.
- **ROCm version mismatch**: System `hipcc` and PyTorch's bundled ROCm runtime
  can differ.  The build system handles this via `HIP_LIB_DIR` (link against
  PyTorch's HIP) and `HIP_COV` (code-object version, e.g. `5` for ROCm 6.x).
  Both are detected automatically; `make clean` + re-import forces a rebuild.
- **Kernel occupancy**: One block per output feature.  At T = 1 (decode) each
  block does little work.  A tiled GEMV or paged KV cache would help.
- **Load time**: Preloading 775 compressed tensors from gzip-npz takes ~18 s on
  the MI50 system.  An uncompressed `.npy`-format cache would be faster to read
  but would require ~15 GB more disk space.

---

## 9. Running the Chat / Comparison

```bash
# Step 1 — compress the model (one-time, ~hours for 9B)
python3 proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B

# Step 2a — run compressed inference (model does not need to fit in VRAM as bf16)
python3 proofofconcept/chat.py ~/workspace/model/Qwen3.5-9B \
    --mode lossless \
    --prompt "Write a haiku about data compression" \
    --max-tokens 100

# Step 2b — run uncompressed HIP inference (fastest when model fits in VRAM)
#   No compression cache needed; bypasses PyTorch/rocBLAS via ck_linear_raw_bf16
python3 proofofconcept/chat.py ~/workspace/model/Qwen3.5-9B \
    --mode uncompressed \
    --prompt "Write a haiku about data compression" \
    --max-tokens 100

# Interactive chat (uncompressed HIP — fastest for 9B on MI50 32GB)
python3 proofofconcept/chat.py ~/workspace/model/Qwen3.5-9B --mode uncompressed

# Compare uncompressed vs. compressed (0.8B, fits in 5 GB VRAM)
python3 proofofconcept/compare.py ~/workspace/model/Qwen3.5-0.8B \
    --prompt "Write a haiku about data compression" --tokens 80
```

`chat.py` supports three `--mode` values:

| Mode           | Description |
|----------------|-------------|
| `lossless`     | Compressed inference via bit-packed codebook (default) |
| `balanced`     | Lossy compressed inference (requires separate compression run) |
| `uncompressed` | Stock model weights via HIP raw bf16 kernel; no cache needed |

Output is automatically saved to a timestamped `.log` file in the current
directory.
