# Mode Comparison Chart — Qwen3.5-9B on AMD Instinct MI50 (gfx906)

_Last updated: 2026-03-14_

All numbers use warmed-up decode (5-token warmup, then 30-token timed run),
greedy decoding, single prompt.  Prefill measured as wall-clock to first
token for `generate(max_new_tokens=5)`.

---

## Decode Speed (tok/s) — higher is better

```
Stock PyTorch          █ 1.4
Compressed HIP         ███████ 10.53
Compressed + GDR       ███████ 10.71
Uncompressed HIP v1    ██████████ 15.51
Uncompressed HIP v2    ██████████ 16.16
                       0         5        10        15        20
                       |         |         |         |         |
```

| Mode                        | Decode (warm) | vs Stock |
|-----------------------------|---------------|----------|
| Stock PyTorch uncompressed  |  1.4 tok/s    |   1×     |
| Compressed lossless (HIP)   | 10.53 tok/s   |   7.5×   |
| Compressed + GDR decode     | 10.71 tok/s   |   7.6×   |
| Uncompressed HIP v1 + GDR   | 15.51 tok/s   |  11.1×   |
| Uncompressed HIP v2 + GDR   | **16.16 tok/s**| **11.5×**|

v2 = `RawKernelLinear` branches on T: rocBLAS for T>1 (prefill), raw HIP
kernel for T=1 (decode).

---

## VRAM Usage — lower is better

```
Stock PyTorch          █████████████████ 17.9 GB
Compressed lossless    ██████████████ 14.6 GB
Uncompressed HIP       █████████████████ 17.9 GB
                       0     5    10    15    20 GB
                       |     |     |     |     |
```

| Mode                        | VRAM    | vs Stock  |
|-----------------------------|---------|-----------|
| Stock PyTorch uncompressed  | 17.9 GB |   —       |
| Compressed lossless (HIP)   | 14.6 GB | **−18%**  |
| Uncompressed HIP v2         | 17.9 GB |   same    |

**Compressed wins when model bf16 size exceeds available VRAM.**

---

## Prefill Latency (~250-token prompt) — lower is better

```
Stock PyTorch          ████████████████████████████████████████ ~60 s
Uncompressed HIP v1    ████████████████████████████ ~7 s
Uncompressed HIP v2    ██ ~1 s
                       0s        15s       30s       45s       60s
                       |         |         |         |         |
```

| Mode                       | ~250-tok prefill | Speedup vs stock |
|----------------------------|------------------|------------------|
| Stock PyTorch uncompressed | ~60 s            |  1×              |
| Uncompressed HIP v1        | ~7 s             |  8.6×            |
| Uncompressed HIP v2        | **~1 s**         | **~60×**         |

Root cause of v1 slowness: `RawKernelLinear` launched `Grid(T×M)` blocks —
poor occupancy for large T.  Fix: `F.linear()` (rocBLAS) for T>1.

---

## Load Time — lower is better

```
Stock PyTorch          ███ 7.2 s
Uncompressed HIP       ███ 7.6 s
Compressed (HIP)       █████████ 18.2 s
                       0s   5s   10s  15s  20s
                       |    |    |    |    |
```

| Mode                        | Load time | Notes                            |
|-----------------------------|-----------|----------------------------------|
| Stock PyTorch uncompressed  | 7.2 s     | safetensors → GPU direct         |
| Uncompressed HIP v2         | 7.6 s     | same + layer replacement pass    |
| Compressed lossless (HIP)   | 18.2 s    | gzip-decompress 775 .npz files   |

Compressed load time dominated by CPU-bound gzip decompression.  Expected
improvement: re-export as uncompressed `.npy` or `zarr+lz4` → ~3–4 s.

---

## rocBLAS vs Raw HIP Kernel at T=1 (decode GEMV)

Our raw kernel avoids rocBLAS dispatch overhead, which dominates at T=1:

```
Shape 4096×4096
  rocBLAS  ████████████████ 1.629 ms
  Raw HIP  █ 0.097 ms   (16.8× faster)

Shape 4096×14336
  rocBLAS  ██████████████████████████████████████████████████████ 5.676 ms
  Raw HIP  ███ 0.266 ms   (21.4× faster)

Shape 14336×4096
  rocBLAS  ████████████████████████████████████ 3.925 ms
  Raw HIP  ███ 0.247 ms   (15.9× faster)
```

| Shape [M×K]  | rocBLAS   | Raw HIP  | Speedup |
|--------------|-----------|----------|---------|
| 4096 × 4096  | 1.629 ms  | 0.097 ms | 16.8×   |
| 4096 × 14336 | 5.676 ms  | 0.266 ms | 21.4×   |
| 14336 × 4096 | 3.925 ms  | 0.247 ms | 15.9×   |

For T>1 (prefill), rocBLAS tiled GEMM wins — no custom kernel needed.

---

## Machine A — Quadro P2200 (CUDA 12.2, 5 GB VRAM) — Qwen3.5-0.8B

```
Decode speed
  Uncompressed GPU  ███████████████ 15.5 tok/s
  Compressed GPU    ████████████ 12.1 tok/s
  Compressed CPU    █ 0.52 tok/s

VRAM
  Uncompressed GPU  ███████████████████████████ 1,471 MB
  Compressed GPU    ████████████████████████████ 1,599 MB
  Compressed CPU    — (CPU only)
```

| Mode              | Load  | CPU RAM Δ | VRAM     | Decode      |
|-------------------|-------|-----------|----------|-------------|
| Uncompressed GPU  | 2.8 s | +517 MB   | 1,471 MB | 15.5 tok/s  |
| Compressed GPU    | 17.8 s| +727 MB   | 1,599 MB | 12.1 tok/s  |
| Compressed CPU    | 16.8 s| +1,611 MB | —        | 0.52 tok/s  |

Note: 0.8B fits in VRAM uncompressed; compression adds ~9% decode overhead
for this card.  Compression saves VRAM only when model > card capacity.

---

## Summary: When to Use Each Mode

| Scenario                           | Recommended mode       |
|------------------------------------|------------------------|
| 9B model, MI50 32 GB VRAM          | `--mode uncompressed`  |
| 9B model, GPU with < 18 GB VRAM    | `--mode lossless`      |
| No compression cache available     | `--mode uncompressed`  |
| Maximum throughput, VRAM not tight  | `--mode uncompressed`  |
| Minimize VRAM footprint             | `--mode lossless`      |
