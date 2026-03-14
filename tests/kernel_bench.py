#!/usr/bin/env python3
"""
Microbenchmark and correctness tests for compressed_linear kernels.

Tests:
  1. Correctness: tiled vs un-tiled output must match
  2. Boundary conditions: M/K not divisible by tile sizes, T>1
  3. lm_head scale: M=248320 correctness + timing
  4. Speed comparison: tiled vs un-tiled vs cuBLAS across model shapes
  5. (Run separately) layer_compare regression — see tests/layer_compare.py

Usage:
    python proofofconcept/tests/kernel_bench.py
    python proofofconcept/tests/kernel_bench.py --test 1   # run single test
    python proofofconcept/tests/kernel_bench.py --bench    # run speed table only
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'proofofconcept' / 'src'))

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_packed(M, K, bits, seed=42):
    """Generate random bit-packed indices simulating a compressed weight matrix."""
    rng = np.random.default_rng(seed)
    n_elements = M * K
    max_idx = (1 << bits) - 1
    indices = rng.integers(0, max_idx, size=n_elements, dtype=np.int64)

    # Vectorised bit-pack using numpy (matches AdaptiveCompressor packing).
    # Equivalent to the Python loop but ~1000x faster for large arrays.
    total_bits = n_elements * bits
    n_bytes = (total_bits + 7) // 8 + 4   # +4 pad bytes
    packed = np.zeros(n_bytes, dtype=np.uint8)

    i = np.arange(n_elements, dtype=np.int64)
    bit_pos  = i * bits
    byte_pos = (bit_pos >> 3).astype(np.int64)
    bit_shft = (bit_pos & 7).astype(np.int32)

    idx = indices.astype(np.int64)
    np.add.at(packed, byte_pos,     ((idx << bit_shft) & 0xFF).astype(np.uint8))
    np.add.at(packed, byte_pos + 1, ((idx >> (8  - bit_shft)) & 0xFF).astype(np.uint8))
    if bits > 8:
        np.add.at(packed, byte_pos + 2, ((idx >> (16 - bit_shft)) & 0xFF).astype(np.uint8))

    return packed, indices


def make_codebook(n_entries, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n_entries).astype(np.float32)


def reference_matmul(x_np, indices, codebook_np, M, K):
    """CPU reference: materialise weight then F.linear."""
    W = codebook_np[indices].reshape(M, K).astype(np.float32)
    x = torch.from_numpy(x_np)
    w = torch.from_numpy(W)
    return F.linear(x, w).numpy()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────────────────────────────────────
# Import kernel wrappers (loaded after sys.path is set)
# ──────────────────────────────────────────────────────────────────────────────

def load_ext():
    from gpu_accelerated_functions import _load_extension, _pad_packed
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA extension failed to load")
    return ext, _pad_packed


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Correctness: tiled vs un-tiled
# ──────────────────────────────────────────────────────────────────────────────

def test_correctness(ext, pad_packed, shapes=None, bits=13, tol=1e-3):
    """
    For each (M, K) shape: tiled and un-tiled must produce output within tol of
    the CPU reference (materialised weights + F.linear).
    """
    if shapes is None:
        shapes = [
            (896,  256,  "Qwen0.8B attn small"),
            (3584, 896,  "Qwen0.8B MLP"),
            (4096, 4096, "7B-sized square"),
        ]

    print("\n=== Test 1: Correctness (tiled vs un-tiled vs reference) ===")
    all_pass = True

    for M, K, label in shapes:
        C = (1 << bits) - 1
        packed_np, indices = make_packed(M, K, bits)
        codebook_np = make_codebook(C)

        T = 1
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((T, K)).astype(np.float32)

        ref = reference_matmul(x_np, indices, codebook_np, M, K)

        packed_t   = pad_packed(packed_np).to(DEVICE)
        codebook_t = torch.from_numpy(codebook_np).to(DEVICE)
        x_t        = torch.from_numpy(x_np).to(DEVICE)

        # Un-tiled
        out_untiled = ext.fused_compressed_linear(
            x_t, packed_t, codebook_t, M, K, bits
        ).cpu().numpy()

        # Tiled
        out_tiled = ext.fused_compressed_linear_tiled(
            x_t, packed_t, codebook_t, M, K, bits
        ).cpu().numpy()

        err_untiled = np.abs(out_untiled - ref).max()
        err_tiled   = np.abs(out_tiled   - ref).max()
        match       = np.abs(out_tiled - out_untiled).max()

        ok = err_untiled < tol and err_tiled < tol
        all_pass = all_pass and ok
        status = "✅" if ok else "❌"
        print(f"  {status} {label:30s} M={M:6d} K={K:4d}  "
              f"err_untiled={err_untiled:.2e}  err_tiled={err_tiled:.2e}  "
              f"tiled_vs_untiled={match:.2e}")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Boundary conditions
# ──────────────────────────────────────────────────────────────────────────────

def test_boundaries(ext, pad_packed, bits=13, tol=1e-3):
    """
    M or K not divisible by tile sizes; T > 1.
    """
    print("\n=== Test 2: Boundary conditions ===")
    all_pass = True

    cases = [
        (3585, 896,  1, "M not div by TILE_M"),
        (3584, 897,  1, "K not div by K_TILE"),
        (3585, 897,  1, "both non-divisible"),
        (3584, 896,  4, "T=4 batch"),
        (3584, 896,  8, "T=8 batch"),
    ]

    C = (1 << bits) - 1
    for M, K, T, label in cases:
        packed_np, indices = make_packed(M, K, bits)
        codebook_np = make_codebook(C)
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((T, K)).astype(np.float32)

        ref = reference_matmul(x_np, indices, codebook_np, M, K)

        packed_t   = pad_packed(packed_np).to(DEVICE)
        codebook_t = torch.from_numpy(codebook_np).to(DEVICE)
        x_t        = torch.from_numpy(x_np).to(DEVICE)

        out_tiled = ext.fused_compressed_linear_tiled(
            x_t, packed_t, codebook_t, M, K, bits
        ).cpu().numpy()

        err = np.abs(out_tiled - ref).max()
        ok = err < tol
        all_pass = all_pass and ok
        status = "✅" if ok else "❌"
        print(f"  {status} {label:30s} M={M:6d} K={K:4d} T={T}  err={err:.2e}")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — lm_head scale
# ──────────────────────────────────────────────────────────────────────────────

def test_lm_head(ext, pad_packed, bits=13, tol=1e-2):
    """
    M=248320 (Qwen vocab size). Correctness + confirm no CUDA config error.
    Tolerance is relaxed (1e-2) because accumulated float32 error is larger at scale.
    """
    print("\n=== Test 3: lm_head scale (M=248320) ===")
    M, K, T = 248320, 2048, 1
    C = (1 << bits) - 1

    packed_np, indices = make_packed(M, K, bits, seed=5)
    codebook_np = make_codebook(C, seed=5)
    x_np = np.random.default_rng(5).standard_normal((T, K)).astype(np.float32)

    ref = reference_matmul(x_np, indices, codebook_np, M, K)

    packed_t   = pad_packed(packed_np).to(DEVICE)
    codebook_t = torch.from_numpy(codebook_np).to(DEVICE)
    x_t        = torch.from_numpy(x_np).to(DEVICE)

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_untiled = ext.fused_compressed_linear(
            x_t, packed_t, codebook_t, M, K, bits
        )
        torch.cuda.synchronize()
        t_untiled = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_tiled = ext.fused_compressed_linear_tiled(
            x_t, packed_t, codebook_t, M, K, bits
        )
        torch.cuda.synchronize()
        t_tiled = (time.perf_counter() - t0) * 1000

        err_u = np.abs(out_untiled.cpu().numpy() - ref).max()
        err_t = np.abs(out_tiled.cpu().numpy()   - ref).max()
        ok = err_u < tol and err_t < tol

        status = "✅" if ok else "❌"
        print(f"  {status} M=248320 K=2048  "
              f"err_untiled={err_u:.2e}  err_tiled={err_t:.2e}  "
              f"untiled={t_untiled:.1f}ms  tiled={t_tiled:.1f}ms")
        return ok

    except RuntimeError as e:
        print(f"  ❌ CUDA error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Speed comparison
# ──────────────────────────────────────────────────────────────────────────────

def bench_shape(ext, pad_packed, M, K, T=1, bits=13, n=50, warmup=5):
    """Returns (ms_untiled, ms_tiled, ms_cublas)."""
    C = (1 << bits) - 1
    packed_np, indices = make_packed(M, K, bits)
    codebook_np = make_codebook(C)
    x_np = np.random.default_rng(1).standard_normal((T, K)).astype(np.float32)

    packed_t   = pad_packed(packed_np).to(DEVICE)
    codebook_t = torch.from_numpy(codebook_np).to(DEVICE)
    x_t        = torch.from_numpy(x_np).to(DEVICE)

    # Materialised weight for cuBLAS reference
    W_np = codebook_np[indices].reshape(M, K).astype(np.float32)
    W_t  = torch.from_numpy(W_np).to(DEVICE)

    def time_fn(fn, n=n, warmup=warmup):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    ms_untiled = time_fn(lambda: ext.fused_compressed_linear(
        x_t, packed_t, codebook_t, M, K, bits))
    ms_tiled = time_fn(lambda: ext.fused_compressed_linear_tiled(
        x_t, packed_t, codebook_t, M, K, bits))
    ms_cublas = time_fn(lambda: F.linear(x_t, W_t))

    return ms_untiled, ms_tiled, ms_cublas


def test_speed(ext, pad_packed):
    """
    Compare un-tiled vs tiled vs cuBLAS across shapes and batch sizes.
    Also shows 'auto' kernel choice for each shape.
    Test passes if un-tiled is used for small T (auto mode selects correctly).
    """
    from gpu_accelerated_functions import _select_kernel
    print("\n=== Test 4: Speed comparison ===")
    print(f"  auto threshold: T*M > 50000 → tiled (covers lm_head, not MLP)")
    print(f"  {'Shape':30s}  {'un-tiled':>10}  {'tiled':>10}  {'cuBLAS':>10}  {'auto':>8}  {'winner':>8}")
    print("  " + "-" * 90)

    shapes = [
        (896,    256,   1, "Qwen0.8B attn small"),
        (3584,   896,   1, "Qwen0.8B MLP T=1"),
        (3584,   896,   8, "Qwen0.8B MLP T=8"),
        (4096,  4096,   1, "7B-sized T=1"),
        (11008, 4096,   1, "7B MLP T=1"),
        (248320, 2048,  1, "lm_head T=1"),       # T*M=248320 > 50000 → auto=tiled
    ]

    all_pass = True
    for M, K, T, label in shapes:
        try:
            u, t, c = bench_shape(ext, pad_packed, M, K, T)
            auto_fn = _select_kernel(ext, T, M, K)
            auto_is_tiled = (auto_fn == ext.fused_compressed_linear_tiled)
            auto_ms = t if auto_is_tiled else u
            auto_label = "tiled" if auto_is_tiled else "untiled"
            winner = "tiled" if t < u else "untiled"
            # Test passes if auto mode picks the faster kernel
            ok = (auto_is_tiled == (t < u))
            all_pass = all_pass and ok
            status = "✅" if ok else "⚠️ "
            print(f"  {status} {label:30s}  {u:9.2f}ms  {t:9.2f}ms  {c:9.2f}ms  "
                  f"{auto_ms:>7.2f}ms  {winner:>8s}")
        except Exception as e:
            print(f"  ❌ {label}: {e}")
            all_pass = False

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=0, help='Run only test N (1-4)')
    parser.add_argument('--bench', action='store_true', help='Speed table only')
    args = parser.parse_args()

    if DEVICE == 'cpu':
        print("ERROR: CUDA not available. These tests require a GPU.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    ext, pad_packed = load_ext()

    results = {}

    run_all = args.test == 0 and not args.bench

    if run_all or args.test == 1:
        results[1] = test_correctness(ext, pad_packed)
    if run_all or args.test == 2:
        results[2] = test_boundaries(ext, pad_packed)
    if run_all or args.test == 3:
        results[3] = test_lm_head(ext, pad_packed)
    if run_all or args.test == 4 or args.bench:
        results[4] = test_speed(ext, pad_packed)

    if results:
        print("\n=== Summary ===")
        all_ok = True
        for k, v in sorted(results.items()):
            status = "✅ PASS" if v else "❌ FAIL"
            print(f"  Test {k}: {status}")
            all_ok = all_ok and v
        print()
        sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
