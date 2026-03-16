# Inference Benchmark Results

All benchmarks use greedy decoding (`do_sample=False`) with a 200-token output
budget unless noted.  "Load time" is wall-clock from process start to first token.

---

## Machine A — Quadro P2200 (CUDA, 5 GB VRAM)

```
GPU  : Quadro P2200, 5 GB VRAM (Pascal gp107, CUDA 12.2)
CPU  : (6 cores)
RAM  : (host)
PyTorch : 2.5.1+cu121
Model   : Qwen3.5-0.8B (lossless compressed)
Prompt  : "Write a haiku about data compression"
```

| Mode              | Load time | CPU RAM Δ | VRAM peak | Speed        |
|-------------------|-----------|-----------|-----------|--------------|
| Uncompressed GPU  | 2.8 s     | +517 MB   | 1 471 MB  | 15.5 tok/s   |
| Compressed GPU    | 17.8 s    | +727 MB   | 1 599 MB  | 12.1 tok/s   |
| Compressed CPU    | 16.8 s    | +1 611 MB | —         | 0.52 tok/s   |

**Notes:**
- Compressed GPU uses `compressed_matmul_v3` CUDA JIT kernel (cached after first compile).
- Compressed CPU uses `compressed_matmul.c` + OpenMP.
- 0.8B fits in VRAM uncompressed; compression adds ~9% overhead for this card.

---

## Machine B — AMD Instinct MI50 (ROCm, 32 GB HBM2)

```
GPU  : AMD Instinct MI50, 32 GB HBM2 (Vega20, gfx906)
CPU  : (host)
RAM  : 125 GB
ROCm : 7.2.0 (system hipcc) / 6.0.32830 (PyTorch bundled)
PyTorch : 2.4.1+rocm6.0
transformers : 5.3.0
Model   : Qwen3.5-9B (hybrid full-attention + GatedDeltaNet SSM)
Prompt  : "Write a haiku about data compression"
Tokens  : 100
```

### Speed summary

All decode speeds are **warmed-up** (5-token warmup, then measure 30 tokens).
Cold-start first tokens are slower due to HIP kernel cache initialisation.

| Mode                          | Load   | VRAM    | Decode (warm)  | 200-tok prefill |
|-------------------------------|--------|---------|----------------|-----------------|
| Stock PyTorch uncompressed    | 7.2 s  | 17.9 GB | 1.4 tok/s      | ~60 s           |
| Compressed lossless (HIP)     | 18.2 s | 14.6 GB | **10.53 tok/s**| (not measured)  |
| Compressed lossless + GDR     | 18.2 s | 14.6 GB | **10.71 tok/s**| (not measured)  |
| Uncompressed HIP v1 + GDR     | 7.6 s  | 17.9 GB | 15.51 tok/s    | ~7 s            |
| Uncompressed HIP v2 + GDR     | 7.6 s  | 17.9 GB | **16.16 tok/s**| **~1 s**        |

v2 changes vs v1: `RawKernelLinear` uses `F.linear` (rocBLAS) for T > 1 (prefill),
raw HIP kernel for T = 1 (decode).  GDR decode kernel (`ck_gdr_decode_step_f32`)
injected for both compressed and uncompressed modes.

**Note:** the previously reported 8.6 tok/s for compressed was measured cold (no
warmup).  The true steady-state speed with warmup is 10.53 tok/s.

**rocBLAS vs raw HIP kernel at T=1 (key finding):**

Our raw kernel is **15–21× faster** than rocBLAS for single-token GEMV:

| Shape [M×K]  | rocBLAS  | Raw HIP | Speedup |
|--------------|----------|---------|---------|
| 4096 × 4096  | 1.629 ms | 0.097 ms | 16.8×  |
| 4096 × 14336 | 5.676 ms | 0.266 ms | 21.4×  |
| 14336 × 4096 | 3.925 ms | 0.247 ms | 15.9×  |

rocBLAS dispatch overhead dominates for GEMV (T=1).  Our kernel skips it.
For T > 1 (prefill), rocBLAS tiled GEMM is used instead (see prefill section).

### Explanation of speed differences

**Stock PyTorch → Compressed HIP (+6×)**

Two compounding bottlenecks removed:
1. `Qwen3_5GatedDeltaNet` SSM layers fall back to pure Python loops without
   `flash-linear-attention` library. Compressed inference routes all matmuls
   through our HIP kernel, which happens to bypass the slow Python SSM path.
2. PyTorch → rocBLAS dispatch overhead per layer call is eliminated.

**Compressed HIP → Uncompressed HIP (+54%)**

When the model fits in VRAM, sequential reads from a contiguous bf16 weight
matrix outperform the compressed kernel's pattern of:
```
  for each output row:
      for each input col:
          idx = unpack_bits(packed, row*K+col, bits)   ← scatter, cache-unfriendly
          acc += x[col] * codebook[idx]               ← codebook lookup
```
The raw kernel reads `weight[row, col]` linearly — better L2/HBM2 utilisation.

**Compressed HIP wins when:** model bf16 size > available VRAM.

### When to use each mode

| Scenario                          | Recommended mode      |
|-----------------------------------|-----------------------|
| 9B model, MI50 32 GB VRAM         | `--mode uncompressed` |
| 9B model, card < 18 GB VRAM       | `--mode lossless`     |
| No compression cache available    | `--mode uncompressed` |
| Maximum quality, any VRAM         | `--mode lossless`     |

---

## SSM kernel correctness (gfx906)

GDR decode kernel validated against PyTorch reference on MI50:

```
Input dims : B=1, H=64, KD=64, VD=128
cos similarity : 0.999988
MSE            : 1.01e-08
```

GDR decode kernel is injected in **both** `--mode lossless` and
`--mode uncompressed` via `inject_ssm_kernels()` in `model_loader.py` and
`uncompressed_loader.py`.  Warmed-up impact:

| Mode              | Without GDR | With GDR    | Δ       |
|-------------------|-------------|-------------|---------|
| Compressed HIP    | 10.53 tok/s | 10.71 tok/s | +1.7%   |
| Uncompressed HIP  | 15.51 tok/s | 16.16 tok/s | +4.2%   |

The previously reported regression (8.6 → 6.3 tok/s) was measured cold
(no warmup).  With proper warmup the kernel is a net gain in both modes.

Expected larger win zone for the GDR kernel: prefill paths with seq_len >> 1
(i.e. `chunk_gated_delta_rule`, Phase 3b — sequential HIP kernel implemented
but not injected because Python chunk path uses rocBLAS GEMMs and is faster).

---

## Prefill latency (Qwen3.5-9B, MI50, --mode uncompressed)

Measured as wall-clock time for `generate(max_new_tokens=5)`:

| Input length | Before fix | After fix | Speedup |
|-------------|-----------|-----------|---------|
| 1 token     | 1.43 s    | 1.43 s    | 1×      |
| ~60 tokens  | 2.20 s    | 0.92 s    | **2.4×** |
| ~250 tokens | 7.03 s    | 1.38 s    | **5.1×** |

Steady-state decode (tokens 2+): **14.75 tok/s** (up from 13.3 tok/s).

**Root cause of original slowness:** `RawKernelLinear` launched one block per
output row (`Grid(T×M)`).  For prefill with T=251, this launches ~1M blocks
with only 16 iterations each — poor GPU occupancy vs rocBLAS tiled GEMM.

**Fix:** `RawKernelLinear.forward()` now branches on T:
- T == 1 (decode): our raw kernel — avoids rocBLAS dispatch overhead
- T  > 1 (prefill): `F.linear()` (rocBLAS) — optimized tiled GEMM

The GDR prefill path (`chunk_gated_delta_rule`) was NOT the bottleneck:
Python `torch_chunk_gated_delta_rule` takes only 6.7 ms/layer × 24 = 160 ms
total for 251 tokens, vs 5.4 s in other ops.  The sequential HIP kernel
(`ck_gdr_prefill_sequential_f32`) is slower than the Python fallback for
large T because the Python path uses rocBLAS for intra-chunk GEMMs.
The sequential kernel is kept for correctness testing.

**First-token latency for a ~200-word prompt:** ~1 s (was ~7 s).
