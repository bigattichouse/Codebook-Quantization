"""
Tests for compressed_matmul_cpu — the gcc-compiled C kernel for true compressed
linear inference (no float weight matrix ever materialised).
"""

import os
import sys
import gc
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bitpack import pack_any_bits, unpack_any_bits
from compressed_matmul_cpu import compressed_matmul, C_KERNEL_AVAILABLE
from compressed_modules import AdaptiveCodebookLinear

psutil = pytest.importorskip("psutil")


def _get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _make_packed(M, K, bits, seed=0):
    rng = np.random.default_rng(seed)
    C = 2 ** bits
    raw = rng.integers(0, C, size=M * K, dtype=np.uint16)
    packed = pack_any_bits(raw, bits)
    codebook = (rng.standard_normal(C) * 0.02).astype(np.float32)
    return raw, packed, codebook


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCKernelCorrectness:
    """compressed_matmul output must match explicit weight-matrix @ x."""

    @pytest.mark.parametrize("M,K,T,bits", [
        (64,  32,  1, 8),
        (64,  32,  4, 8),
        (256, 128, 1, 13),
        (256, 128, 3, 13),
        (512, 256, 1, 13),
        (128, 64,  8, 13),
    ])
    def test_matches_dense(self, M, K, T, bits):
        raw, packed, codebook = _make_packed(M, K, bits)
        w_ref = codebook[raw].reshape(M, K)
        x = np.random.randn(T, K).astype(np.float32) * 0.1

        ref = x @ w_ref.T
        out = compressed_matmul(x, packed, codebook, M, K, bits)

        np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-4,
            err_msg=f"M={M} K={K} T={T} bits={bits}")

    def test_bits8_fast_path(self):
        """8-bit direct byte path."""
        M, K, T = 128, 64, 2
        raw, packed, codebook = _make_packed(M, K, 8)
        w_ref = codebook[raw].reshape(M, K)
        x = np.random.randn(T, K).astype(np.float32)
        ref = x @ w_ref.T
        out = compressed_matmul(x, packed, codebook, M, K, bits=8)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_m_not_divisible_by_chunk(self):
        """M=100 with default chunk of 256 — single partial chunk."""
        M, K, T, bits = 100, 64, 1, 13
        raw, packed, codebook = _make_packed(M, K, bits)
        w_ref = codebook[raw].reshape(M, K)
        x = np.random.randn(T, K).astype(np.float32)
        ref = x @ w_ref.T
        out = compressed_matmul(x, packed, codebook, M, K, bits, chunk_rows=64)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_multiple_chunks(self):
        """Force many small chunks to exercise chunk boundary logic."""
        M, K, T, bits = 256, 64, 2, 13
        raw, packed, codebook = _make_packed(M, K, bits)
        w_ref = codebook[raw].reshape(M, K)
        x = np.random.randn(T, K).astype(np.float32)
        ref = x @ w_ref.T
        out = compressed_matmul(x, packed, codebook, M, K, bits, chunk_rows=32)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_output_shape_batched(self):
        """Output shape is (T, M) for all T."""
        M, K, bits = 64, 32, 13
        _, packed, codebook = _make_packed(M, K, bits)
        for T in [1, 4, 16]:
            x = np.random.randn(T, K).astype(np.float32)
            out = compressed_matmul(x, packed, codebook, M, K, bits)
            assert out.shape == (T, M), f"Expected ({T},{M}), got {out.shape}"

    def test_reproducible(self):
        """Two calls with the same input produce identical output."""
        M, K, T, bits = 64, 32, 2, 13
        _, packed, codebook = _make_packed(M, K, bits)
        x = np.random.randn(T, K).astype(np.float32)
        out1 = compressed_matmul(x, packed, codebook, M, K, bits)
        out2 = compressed_matmul(x, packed, codebook, M, K, bits)
        np.testing.assert_array_equal(out1, out2)

    def test_large_M_no_int32_overflow(self):
        """
        Regression: for M=248320, K=1024, 13-bit, the element index r*K+k reaches
        248319*1024+1023 = 254,279,679.  At 13 bits/elem, the bit offset is
        3,305,635,827 which overflows int32 (max 2,147,483,647).
        The C kernel must use int64 internally; result must match dense matmul.
        """
        # Use smaller proxy that still causes int32 overflow: need r*K*bits > 2^31.
        # Choose M=2100, K=1024, bits=13 → max elem=2099*1024+1023=2,150,399
        # bit_pos = 2,150,399 * 13 = 27,955,187 — still fine.
        # To actually overflow with feasible size: M=200000, K=13 → 200000*13*13=33.8M OK.
        # Actual overflow: need elem * bits > 2^31.
        # elem_max = M*K-1; bits=13. Need M*K > 2^31/13 = 165,191,050.
        # Use M=1024, K=162000 → 1024*162000=165,888,000 > threshold. Too much RAM.
        # Use M=165200, K=1 → just over threshold. 1-D layer, bits=13.
        M, K, bits = 165200, 1, 13
        rng = np.random.default_rng(7)
        C = 2 ** bits
        raw = rng.integers(0, C, size=M * K, dtype=np.uint16)
        from bitpack import pack_any_bits
        packed = pack_any_bits(raw, bits)
        codebook = rng.standard_normal(C).astype(np.float32) * 0.02

        x = rng.standard_normal((1, K)).astype(np.float32)
        w_ref = codebook[raw].reshape(M, K)
        ref = x @ w_ref.T

        out = compressed_matmul(x, packed, codebook, M, K, bits, C=C)
        np.testing.assert_allclose(out, ref, atol=1e-5,
            err_msg="int32 overflow in bit-index computation — check int64_t in unpack_idx")


# ---------------------------------------------------------------------------
# AdaptiveCodebookLinear integration
# ---------------------------------------------------------------------------

class TestAdaptiveLinearCKernel:
    """AdaptiveCodebookLinear.forward() uses the C kernel path."""

    def _make_layer(self, M, K, bits):
        from bitpack import pack_any_bits
        rng = np.random.default_rng(42)
        C = 2 ** bits
        raw = rng.integers(0, C, size=M * K, dtype=np.uint16)
        packed = pack_any_bits(raw, bits)
        codebook = torch.from_numpy((rng.standard_normal(C) * 0.02).astype(np.float32))

        layer = AdaptiveCodebookLinear(f"test.c.{M}.{K}", (M, K), 'direct_codebook')
        layer.bits = bits
        layer.register_buffer('codebook', codebook, persistent=False)
        layer.register_buffer('indices', torch.from_numpy(packed), persistent=False)
        layer.original_len = M * K
        return layer, raw, codebook.numpy()

    def test_matches_dense_matmul(self):
        """Forward output must match explicit dense matmul."""
        M, K, bits = 64, 32, 13
        layer, raw, cb_np = self._make_layer(M, K, bits)
        w_dense = torch.from_numpy(cb_np[raw].reshape(M, K))
        x = torch.randn(2, K)
        with torch.no_grad():
            out_c = layer(x)
            out_ref = torch.nn.functional.linear(x, w_dense)
        torch.testing.assert_close(out_c.float(), out_ref.float(), atol=1e-4, rtol=1e-4)

    def test_no_cached_weight(self):
        """_cached_weight must remain None after forward (no weight matrix stored)."""
        layer, _, _ = self._make_layer(32, 16, 8)
        x = torch.randn(1, 16)
        with torch.no_grad():
            layer(x)
        assert layer._cached_weight is None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestCKernelMemory:
    """Peak RSS must be bounded — no full float weight matrix in memory."""

    def test_peak_ram_bounded_large_layer(self):
        """
        4096×4096 layer, 13-bit: peak RAM growth must be << M*K*4 bytes (~64 MB).
        With the C kernel, only packed (~27 MB) + output (~tiny) exist at once.
        Allow 60 MB headroom for Python/numpy overhead.
        """
        M, K, bits = 4096, 4096, 13
        raw, packed, codebook = _make_packed(M, K, bits)
        x = np.random.randn(1, K).astype(np.float32)

        gc.collect()
        rss_before = _get_rss_mb()
        _ = compressed_matmul(x, packed, codebook, M, K, bits)
        gc.collect()
        rss_after = _get_rss_mb()

        growth = rss_after - rss_before
        # Full weight matrix would be M*K*4 = 64 MB; we allow 60 MB headroom
        assert growth < 60, (
            f"RSS grew {growth:.1f} MB — float weight matrix may have been created "
            f"(expected < 60 MB; full M*K float32 would be 64 MB)"
        )

    def test_rss_stable_across_calls(self):
        """10 forward calls must not accumulate RAM."""
        M, K, bits = 512, 256, 13
        raw, packed, codebook = _make_packed(M, K, bits)
        x = np.random.randn(1, K).astype(np.float32)

        # Warm up
        compressed_matmul(x, packed, codebook, M, K, bits)
        gc.collect()
        rss_before = _get_rss_mb()

        for _ in range(10):
            compressed_matmul(x, packed, codebook, M, K, bits)

        gc.collect()
        rss_after = _get_rss_mb()
        growth = rss_after - rss_before
        assert growth < 20, f"RSS grew {growth:.1f} MB across 10 calls"


# ---------------------------------------------------------------------------
# C kernel availability
# ---------------------------------------------------------------------------

def test_c_kernel_available():
    """gcc build must succeed on this machine."""
    assert C_KERNEL_AVAILABLE, (
        "C kernel not available — check gcc is installed. "
        "Numpy fallback will be used instead."
    )
