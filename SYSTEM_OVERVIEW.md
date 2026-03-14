# Adaptive Codebook Compression — System Overview

This document describes the compression and inference system implemented in this
repository, including the two-pass compression pipeline, the bitstream packing
format, and the CUDA inference kernels.

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
The numpy implementation is fully vectorized (no Python loops); the CUDA kernel
(§4) reproduces the same logic on-device.

**Storage savings example (13-bit vs uint16):**

```
300 M params × 13 bits = 487.5 MB
300 M params × 16 bits = 600.0 MB   (uint16)
Savings: ~19%
```

---

## 4. CUDA Inference Kernels

Kernels are JIT-compiled by `nvcc` via PyTorch's `load_inline` extension
mechanism and cached in `~/.cache/torch_extensions/py312_cu121/compressed_matmul_v3/`.
Compilation happens once on first import (~30–60 s) and is skipped on all
subsequent runs.

### 4a. Linear kernel (`compressed_linear_kernel`)

```
Grid(T × M),  Block(256 threads)
```

Each CUDA block handles one output element `out[tok, ofeat]`.  Threads
cooperatively reduce over the `K` (input-feature) dimension:

```
for j in range(tid, K, 256):
    idx  = unpack_idx(packed, ofeat * K + j, bits)
    acc += x[tok, j] * codebook[idx]
out[tok, ofeat] = reduce_sum(acc)
```

The codebook (~30 KB for 8192 entries) stays resident in L2 cache across all
warps.  No weight matrix is ever materialised.

A 1-D grid (`Grid(T × M)`) avoids the 65 535 limit on `grid.y` on Pascal GPUs,
which matters for the language-model head (`M = vocab_size = 151 552` on
Qwen3.5-0.8B).

### 4b. Embedding kernel (`compressed_embedding_kernel`)

```
Grid(T, ceil(H / 128)),  Block(128 threads)
```

Each thread looks up one output element:

```
out[tok, h] = codebook[ unpack_idx(packed, token_id * H + h, bits) ]
```

---

## 5. Memory-Safe Model Loading

Model instantiation follows this sequence to avoid any large transient allocation:

1. **Meta device** — `AutoModelForCausalLM.from_config(..., device='meta')`.
   Creates the full model graph with zero bytes of real memory.
2. **`to_empty(device)`** — allocates uninitialized storage on the target device
   (CUDA or CPU) for every parameter and buffer.
3. **Exact weight loading** — norm layers, small tensors, and "exact"-mode
   tensors are streamed in one at a time from disk.
4. **Compressed module replacement** — `nn.Linear` / `nn.Embedding` layers are
   replaced with `AdaptiveCodebookLinear` / `AdaptiveCodebookEmbedding` modules
   that hold compressed indices + a reference to the shared codebook.  This step
   must happen **after** `to_empty()` so the newly registered buffers are not
   wiped.

At inference time the forward pass decompresses only the slice of weights needed
for the current matmul; the full weight matrix is never resident in memory.

---

## 6. Measured Results — Qwen3.5-0.8B (Quadro P2200, 5 GB VRAM)

These numbers were measured on the hardware described in §7.

| Mode         | Load time | RAM    | VRAM   | Speed      | Correctness |
|--------------|-----------|--------|--------|------------|-------------|
| Uncompressed | —         | —      | 1.6 GB | 17.5 tok/s | baseline    |
| Compressed   | ~58 s*    | 4.2 GB | 1.6 GB | 5.2 tok/s  | cos > 0.999 |

*First run compiles and caches the CUDA extension (~30–60 s).  Subsequent runs
skip compilation.

**Layer-by-layer comparison** (greedy decode, all 24 layers):
- Cosine similarity vs. uncompressed: > 0.999 for every layer
- Greedy 5-token prediction: exact match

**Lossless mode** uses ~13-bit codebooks (6 837–7 848 entries per category) which
achieve 0 MSE by construction — the codebook contains every unique value that
appears in the model.

---

## 7. Hardware and Software

```
GPU  : Quadro P2200, 5 GB VRAM (Pascal, CUDA 12.2)
CPU  : (consumer desktop)
RAM  : system RAM used for CPU-offloaded layers
PyTorch : 2.5.1+cu121
transformers : 5.2.0
Python  : 3.12
CUDA ext cache : ~/.cache/torch_extensions/py312_cu121/compressed_matmul_v3/
```

---

## 8. Current Limitations

- **Speed overhead**: On-the-fly index lookup adds ~3.4× latency per layer
  compared to uncompressed inference (18–35 ms vs. 3–11 ms per layer on 0.8B).
  Total throughput is 5.2 tok/s vs. 17.5 tok/s.
- **Compression time**: The two-pass pipeline is CPU-only and takes tens of
  minutes for large models.  Qwen3.5-0.8B took ~60 min; 9B is expected to scale
  roughly linearly (~8–10 h).
- **9 B model**: Compression in progress.  See `COMPRESSING_9B.md` for
  status, resume instructions, and what to do if the process is interrupted.
- **Kernel occupancy**: The un-tiled kernel uses one block per output feature.
  At T = 1 (decode) each block does little work.  A fused attention kernel or
  paged KV cache could improve end-to-end throughput.

---

## 9. Running the Comparison

```bash
# Step 1 — compress the model (one-time, ~minutes)
./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B

# Step 2 — compare uncompressed vs. compressed
./venv/bin/python proofofconcept/compare.py ~/workspace/model/Qwen3.5-9B \
    --prompt "Write a haiku about data compression" --tokens 80

# Skip uncompressed run if VRAM is tight (model won't fit without CPU offload):
./venv/bin/python proofofconcept/compare.py ~/workspace/model/Qwen3.5-9B \
    --skip-uncompressed
```

Output is automatically saved to a timestamped `.log` file in the current
directory.
