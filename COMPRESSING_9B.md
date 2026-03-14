# Compressing Qwen3.5-9B — Status and Recovery Guide

This document describes the compression run for Qwen3.5-9B and what to do if
the process is interrupted or the session is lost.

---

## Model Details

```
Model  : Qwen3.5-9B
Path   : ~/workspace/model/Qwen3.5-9B
Shards : 4 safetensors files (~19 GB total)
Mode   : lossless
Cache  : ~/workspace/model/Qwen3.5-9B/codebook/tensors/
```

## Why This Is Safe

`compress.py` is **CPU-only**.  It never calls any CUDA API and never allocates
VRAM.  The GPU cannot OOM during compression.  System RAM usage is bounded
(~2–4 GB) because only one tensor is processed at a time.

## Checking Compression Status

```bash
# How many tensors are done?
ls ~/workspace/model/Qwen3.5-9B/codebook/tensors/*.npz | wc -l

# What is the cache size so far?
du -sh ~/workspace/model/Qwen3.5-9B/codebook/

# Are the global codebook .npy files present? (written at the very end)
ls ~/workspace/model/Qwen3.5-9B/codebook/codebooks/
```

A complete run will produce:
- One `.npz` file per tensor (expected: ~2000 tensors for 9B)
- `codebook/codebooks/` directory with 4–6 `.npy` files
- `codebook/metadata.json`

## Starting the Compression

Run this from the repo root.  It will take several hours.

```bash
./venv/bin/python proofofconcept/compress.py \
    ~/workspace/model/Qwen3.5-9B \
    --mode lossless
```

To run in the background and log output:

```bash
nohup ./venv/bin/python proofofconcept/compress.py \
    ~/workspace/model/Qwen3.5-9B \
    --mode lossless \
    > /tmp/compress_9b.log 2>&1 &

echo "PID: $!"
tail -f /tmp/compress_9b.log
```

## If the Run Was Interrupted

The compressor writes `.npz` files one by one.  A partial cache is safe — no
files are corrupted.  However:

- If the run was interrupted **before** the end of Pass 2 (before the codebook
  `.npy` files were saved), inference will not work because the global codebooks
  are missing.
- If the run was interrupted **after** codebooks were saved but some tensors are
  missing, inference will fail on the missing layers.

**The safest recovery is a full rebuild:**

```bash
./venv/bin/python proofofconcept/compress.py \
    ~/workspace/model/Qwen3.5-9B \
    --mode lossless \
    --force
```

`--force` deletes the existing cache and starts over.

## After Compression Completes

### Verify the cache

```bash
ls ~/workspace/model/Qwen3.5-9B/codebook/tensors/*.npz | wc -l
ls ~/workspace/model/Qwen3.5-9B/codebook/codebooks/
cat ~/workspace/model/Qwen3.5-9B/codebook/metadata.json | python3 -m json.tool | head -20
```

### Run inference

```bash
# Compressed-only comparison (skip uncompressed — 9B won't fit in 5 GB VRAM):
./venv/bin/python proofofconcept/compare.py \
    ~/workspace/model/Qwen3.5-9B \
    --mode lossless \
    --skip-uncompressed \
    --prompt "Write a haiku about data compression"

# Interactive chat:
./venv/bin/python proofofconcept/chat.py ~/workspace/model/Qwen3.5-9B
```

### Expected results

- Load time: longer than 0.8B (more tensors to index)
- VRAM: should stay under 5 GB (the full 9B BF16 would be ~18 GB)
- Speed: likely similar or slower tok/s vs. 0.8B due to larger layers
- Correctness: layer cosine similarity > 0.999 (same as 0.8B in lossless mode)

## Expected Timing (Estimate)

| Step                 | Estimate |
|----------------------|----------|
| Pass 1 (histogram)   | 20–40 min |
| Pass 2 (compression) | 4–8 h    |
| Total                | 5–9 h    |

These are rough estimates based on 0.8B timing scaled by parameter count.
Actual time depends on CPU speed and available RAM for the sliding window.
