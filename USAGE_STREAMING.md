# Compressed Loading — Usage Guide

## Quick Start

Compress a model once, then run chat or compare:

```bash
# Compress (one-time, CPU-only — safe to run, no GPU involved)
./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --mode lossless

# Interactive chat
./venv/bin/python proofofconcept/chat.py ~/workspace/model/Qwen3.5-9B

# Benchmark comparison
./venv/bin/python proofofconcept/compare.py ~/workspace/model/Qwen3.5-9B \
    --prompt "Write a haiku about data compression"
```

## Compression Modes

Pass `--mode` to `compress.py` or `chat.py`:

| Mode         | Description |
|--------------|-------------|
| `lossless`   | Codebook contains every unique value in the model — zero MSE. Typical bit-width: 13–15 bits. |
| `balanced`   | Target 99.5% accuracy threshold via frequency-weighted k-means centroids. |
| `aggressive` | Lower MSE threshold; smaller indices; faster load; some accuracy loss. |

Default is `balanced`.

## How It Works

### Pass 1: Global Histogram Analysis

The compressor reads every safetensors shard byte-by-byte, accumulating a
histogram of all BF16 values per layer category (embedding, attention, MLP/FFN,
SSM core, etc.).  No sampling — 100% parameter coverage.

For lossless mode the codebook *is* the histogram's support set: all unique
values that actually appear.  If a category uses ≤ 32 768 unique values, the
indices fit in ≤ 15 bits.

### Pass 2: Streaming Compression

Each tensor is compressed and written to `<model>/codebook/tensors/<name>.npz`.
Tensors that don't benefit from compression (norms, small layers) are stored as
exact BF16.

### Bit-Packing

Indices are packed as a contiguous bitstream: index `i` occupies bits
`[i*b, (i+1)*b)` in the byte array.  At 13 bits/index this saves ~19% storage
vs. uint16.  See `src/bitpack.py` and `src/fast_index_manager.py` for the
implementation.

### CUDA Kernels

On CUDA hardware, compressed matmul is computed directly from packed indices
without materialising the weight matrix.  See `src/gpu_accelerated_functions.py`
and `SYSTEM_OVERVIEW.md §4` for kernel details.

## GPU Safety

`compress.py` is **CPU-only**.  It reads safetensors files, builds numpy
histograms, and writes `.npz` files.  The GPU is never touched during
compression — there is no VRAM OOM risk.

## If the Compression Run Is Interrupted

The compressor writes each `.npz` file to disk immediately after that tensor is
done.  If the process is killed mid-run:

1. Check how many tensors were saved:
   ```bash
   ls ~/workspace/model/Qwen3.5-9B/codebook/tensors/*.npz | wc -l
   ```
2. A partial cache is treated as "complete" (any `.npz` present = skip).
   To rebuild from scratch use `--force`:
   ```bash
   ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B \
       --mode lossless --force
   ```
3. Note: the global codebook `.npy` files are written at the very end of Pass 2.
   If the run was interrupted before that point, `--force` is required.

## Troubleshooting

**"No compression cache found"** — Run `compress.py` first.

**Compression is slow** — Expected.  The two-pass pipeline is CPU-bound.
Qwen3.5-0.8B (1.75 GB) took ~60 min; Qwen3.5-9B (~19 GB) is expected to take
several hours.

**Slow first inference run** — The CUDA extension compiles on first use (~30–60 s)
and is then cached in `~/.cache/torch_extensions/`.

**CUDA extension cache stale** — Delete
`~/.cache/torch_extensions/py312_cu121/compressed_matmul_v3/` to force
recompilation.
