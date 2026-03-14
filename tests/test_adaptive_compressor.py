"""
Tests for AdaptiveCompressor: histogram building, codebook construction,
MSE threshold checking, and safe parallel workers.
"""

import pytest
import numpy as np
import sys
import tempfile
import os
from pathlib import Path

from adaptive_compressor import _compress_adaptive_worker, AdaptiveCompressor
from compressor import float32_to_bfloat16, kmeans_1d


def _write_bf16_tmp(arr: np.ndarray):
    """Write float32 array as BF16 bytes to a temp file.

    Returns (file_path, offset, size, shape, unique_count) so callers can
    pass straight into _compress_adaptive_worker's new file-based signature.
    """
    flat = arr.flatten()
    u16 = (flat.view(np.uint32) >> 16).astype(np.uint16)
    raw = u16.tobytes()
    hist = np.bincount(u16, minlength=65536)
    unique_count = int(np.count_nonzero(hist))

    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    tf.write(raw)
    tf.close()
    return tf.name, 0, len(raw), arr.shape, unique_count


class TestCompressAdaptiveWorker:
    """Test the standalone worker function with synthetic data."""

    def _make_data(self, n=10000, unique=100):
        """Create synthetic weight data with a known number of unique BF16 values."""
        pool = np.linspace(-0.05, 0.05, unique, dtype=np.float32)
        return np.random.choice(pool, size=n)

    def test_exact_for_tiny(self):
        """Tensors with < 1000 elements should get 'exact' mode."""
        arr = np.random.randn(500).astype(np.float32)
        fpath, off, sz, shape, uc = _write_bf16_tmp(arr)
        try:
            name, result, label, uniq, mse, snr = _compress_adaptive_worker(
                "tiny", fpath, off, sz, shape, 'BF16',
                "tiny.weight", "balanced", {}, 0.001, unique_count=uc
            )
            assert result["mode"] == "exact"
        finally:
            os.unlink(fpath)

    def test_lossless_finds_codebook(self):
        """Lossless mode with few unique values should find a codebook."""
        arr = self._make_data(n=50000, unique=64)
        fpath, off, sz, shape, uc = _write_bf16_tmp(arr)
        try:
            name, result, label, uniq, mse, snr = _compress_adaptive_worker(
                "test", fpath, off, sz, shape, 'BF16',
                "model.layers.0.self_attn.q_proj.weight",
                "lossless", {}, 0.001, unique_count=uc
            )
            assert result["mode"] in ("direct_codebook", "exact")
            if result["mode"] == "direct_codebook":
                assert mse == 0.0
        finally:
            os.unlink(fpath)

    def test_balanced_mode(self):
        arr = np.random.randn(50000).astype(np.float32) * 0.02
        fpath, off, sz, shape, uc = _write_bf16_tmp(arr)
        try:
            name, result, label, uniq, mse, snr = _compress_adaptive_worker(
                "test", fpath, off, sz, shape, 'BF16',
                "model.layers.0.mlp.gate_proj.weight",
                "balanced", {}, 0.001, unique_count=uc
            )
            assert result["mode"] in ("direct_codebook", "linear_quant", "exact")
            assert result["shape"] == (50000,)
        finally:
            os.unlink(fpath)

    def test_global_codebook_used(self):
        """Worker should use a global codebook when it gives acceptable quality."""
        arr = np.random.randn(50000).astype(np.float32) * 0.02
        global_cb = kmeans_1d(arr, k=256, seed=42)
        codebooks = {"attention": global_cb}
        fpath, off, sz, shape, uc = _write_bf16_tmp(arr)
        try:
            name, result, label, uniq, mse, snr = _compress_adaptive_worker(
                "test", fpath, off, sz, shape, 'BF16',
                "model.layers.0.self_attn.q_proj.weight",
                "balanced", codebooks, 0.01, unique_count=uc
            )
            assert result["mode"] in ("direct_codebook", "linear_quant", "exact")
        finally:
            os.unlink(fpath)


class TestSafeParallelWorkers:
    def test_returns_at_least_one(self):
        workers = AdaptiveCompressor._safe_parallel_workers(headroom_gb=1000.0)
        assert workers >= 1

    def test_returns_positive_int(self):
        workers = AdaptiveCompressor._safe_parallel_workers()
        assert isinstance(workers, int)
        assert workers >= 1


class TestMSEThreshold:
    def test_critical_layer_strict_threshold(self):
        """Norm layers should get near-zero threshold regardless of mode."""
        arr = np.random.randn(50000).astype(np.float32) * 0.02
        fpath, off, sz, shape, uc = _write_bf16_tmp(arr)
        try:
            name, result, label, uniq, mse, snr = _compress_adaptive_worker(
                "norm", fpath, off, sz, shape, 'BF16',
                "model.layers.0.input_layernorm.weight",
                "lossy", {}, 0.01, unique_count=uc
            )
            if result["mode"] != "exact":
                assert mse < 1e-6
        finally:
            os.unlink(fpath)
