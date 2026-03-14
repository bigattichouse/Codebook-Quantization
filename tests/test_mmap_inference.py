"""
Tests for memory-mapped packed-weight inference.

Goal: run models whose packed indices don't fit in RAM by streaming them
from disk via mmap.  The OS pages data in on demand; RSS stays bounded.

These tests define the contract before implementation:
  1. MmappedPackedBuffer reads the same bytes as in-RAM loading.
  2. AdaptiveCodebookLinear.forward() with mmap gives identical output to RAM.
  3. Loading a model with use_mmap=True does NOT increase RSS by the index size.
  4. A proxy "large model" (fake oversized indices) runs forward without OOM.
"""

import os
import sys
import gc
import tempfile
import mmap as mmap_mod
import numpy as np
import pytest
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bitpack import pack_any_bits, unpack_any_bits

psutil = pytest.importorskip("psutil")


def _get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _write_idx_file(path: Path, packed: np.ndarray):
    """Write a flat .idx file (raw uint8 bytes)."""
    packed.tofile(str(path))


# ---------------------------------------------------------------------------
# MmappedPackedBuffer contract
# ---------------------------------------------------------------------------

class TestMmappedPackedBuffer:
    """MmappedPackedBuffer must behave like a uint8 numpy array."""

    def test_reads_same_bytes_as_numpy(self, tmp_path):
        """Bytes read via mmap must equal bytes written."""
        from compressed_matmul_cpu import MmappedPackedBuffer
        rng = np.random.default_rng(0)
        data = rng.integers(0, 256, size=4096, dtype=np.uint8)
        idx_file = tmp_path / "test.idx"
        _write_idx_file(idx_file, data)

        buf = MmappedPackedBuffer(idx_file)
        np.testing.assert_array_equal(buf.as_numpy(), data,
            err_msg="mmap bytes differ from original")

    def test_slice_correctness(self, tmp_path):
        """Slicing the mmap buffer must return the correct bytes."""
        from compressed_matmul_cpu import MmappedPackedBuffer
        data = np.arange(1024, dtype=np.uint8)
        idx_file = tmp_path / "slice.idx"
        _write_idx_file(idx_file, data)

        buf = MmappedPackedBuffer(idx_file)
        arr = buf.as_numpy()
        np.testing.assert_array_equal(arr[100:200], data[100:200])
        np.testing.assert_array_equal(arr[-10:], data[-10:])

    def test_zero_copy_view(self, tmp_path):
        """as_numpy() should return a view of the mmap (no copy)."""
        from compressed_matmul_cpu import MmappedPackedBuffer
        data = np.ones(512, dtype=np.uint8)
        idx_file = tmp_path / "view.idx"
        _write_idx_file(idx_file, data)

        buf = MmappedPackedBuffer(idx_file)
        arr = buf.as_numpy()
        # A view shares memory — base is the mmap object (not None)
        assert arr.base is not None, "as_numpy() should be a view, not a copy"

    def test_large_file_no_ram_spike(self, tmp_path):
        """Opening a large .idx file via mmap must not load it into RSS."""
        from compressed_matmul_cpu import MmappedPackedBuffer

        # 200 MB of packed data
        size_mb = 200
        data = np.zeros(size_mb * 1024 * 1024, dtype=np.uint8)
        idx_file = tmp_path / "large.idx"
        _write_idx_file(idx_file, data)
        del data
        gc.collect()

        rss_before = _get_rss_mb()
        buf = MmappedPackedBuffer(idx_file)
        _ = buf.as_numpy()  # obtain view — should NOT load all 200 MB
        rss_after = _get_rss_mb()

        growth = rss_after - rss_before
        assert growth < 50, (
            f"Opening mmap caused {growth:.0f} MB RSS growth — "
            f"mmap should map address space, not read pages eagerly"
        )
        buf.close()


# ---------------------------------------------------------------------------
# AdaptiveCodebookLinear mmap path
# ---------------------------------------------------------------------------

class TestAdaptiveLinearMmap:
    """AdaptiveCodebookLinear with mmap indices must match RAM-loaded output."""

    def _make_layer_ram(self, M, K, bits, seed=0):
        """Create a RAM-backed AdaptiveCodebookLinear."""
        from compressed_modules import AdaptiveCodebookLinear
        rng = np.random.default_rng(seed)
        C = min(2 ** bits, 8192)
        raw = rng.integers(0, C, size=M * K, dtype=np.uint16)
        packed = pack_any_bits(raw, bits)
        codebook = torch.from_numpy(
            rng.standard_normal(C).astype(np.float32) * 0.02)

        layer = AdaptiveCodebookLinear(f"test.mmap.{M}.{K}", (M, K), 'direct_codebook')
        layer.bits = bits
        layer.register_buffer('codebook', codebook, persistent=False)
        layer.register_buffer('indices', torch.from_numpy(packed), persistent=False)
        return layer, packed, codebook

    def _make_layer_mmap(self, M, K, bits, packed, codebook, idx_file):
        """Create a mmap-backed AdaptiveCodebookLinear using a pre-written .idx file."""
        from compressed_modules import AdaptiveCodebookLinear
        from compressed_matmul_cpu import MmappedPackedBuffer

        layer = AdaptiveCodebookLinear(f"test.mmap.{M}.{K}", (M, K), 'direct_codebook')
        layer.bits = bits
        layer.register_buffer('codebook', codebook, persistent=False)
        layer.register_buffer('indices', None)        # no RAM copy
        layer._mmap_buf = MmappedPackedBuffer(idx_file)
        return layer

    @pytest.mark.parametrize("M,K,bits", [
        (64, 32, 8),
        (256, 128, 13),
        (512, 256, 13),
    ])
    def test_mmap_matches_ram(self, tmp_path, M, K, bits):
        """mmap forward must be numerically identical to RAM forward."""
        layer_ram, packed, codebook = self._make_layer_ram(M, K, bits)
        idx_file = tmp_path / "test.idx"
        _write_idx_file(idx_file, packed)

        layer_mmap = self._make_layer_mmap(M, K, bits, codebook, codebook, idx_file)

        x = torch.randn(2, K)
        with torch.no_grad():
            out_ram  = layer_ram(x)
            out_mmap = layer_mmap(x)

        torch.testing.assert_close(out_ram.float(), out_mmap.float(), atol=1e-5, rtol=0,
            msg=f"M={M} K={K} bits={bits}: mmap output differs from RAM output")

    def test_mmap_lm_head_shape(self, tmp_path):
        """Proxy for lm_head: M=4096, K=1024 — the int32-overflow shape."""
        M, K, bits = 4096, 1024, 13
        layer_ram, packed, codebook = self._make_layer_ram(M, K, bits)
        idx_file = tmp_path / "lm_head_proxy.idx"
        _write_idx_file(idx_file, packed)
        layer_mmap = self._make_layer_mmap(M, K, bits, codebook, codebook, idx_file)

        x = torch.randn(1, K)
        with torch.no_grad():
            out_ram  = layer_ram(x)
            out_mmap = layer_mmap(x)

        torch.testing.assert_close(out_ram.float(), out_mmap.float(), atol=1e-5, rtol=0)

    def test_mmap_rss_does_not_grow_with_index_size(self, tmp_path):
        """
        Creating a layer with a 100 MB .idx file via mmap must NOT add
        100 MB to RSS.  The test fails if we copy bytes into RAM.
        """
        from compressed_modules import AdaptiveCodebookLinear
        from compressed_matmul_cpu import MmappedPackedBuffer

        # 100 MB of fake packed data
        size = 100 * 1024 * 1024
        fake_packed = np.zeros(size, dtype=np.uint8)
        idx_file = tmp_path / "big.idx"
        _write_idx_file(idx_file, fake_packed)
        del fake_packed
        gc.collect()

        rss_before = _get_rss_mb()
        # Just open the mmap — don't run forward (shape would be wrong)
        buf = MmappedPackedBuffer(idx_file)
        gc.collect()
        rss_after = _get_rss_mb()

        growth = rss_after - rss_before
        assert growth < 30, (
            f"mmap open grew RSS by {growth:.0f} MB for a 100 MB file — "
            f"should be near zero (address space reservation only)"
        )
        buf.close()


# ---------------------------------------------------------------------------
# .idx file exporter
# ---------------------------------------------------------------------------

class TestIdxExporter:
    """export_flat_idx() must produce files that round-trip correctly."""

    def test_export_and_reload_matches_original(self, tmp_path):
        """export_flat_idx then MmappedPackedBuffer must give original bytes."""
        from adaptive_compressor import export_flat_idx
        from compressed_matmul_cpu import MmappedPackedBuffer

        # Write a fake cache structure
        rng = np.random.default_rng(7)
        bits = 13
        M, K = 64, 32
        raw = rng.integers(0, 2 ** bits, size=M * K, dtype=np.uint16)
        packed = pack_any_bits(raw, bits)

        tensor_name = "model_layers_0_mlp_weight"
        cache_dir = tmp_path / "codebook" / "tensors"
        cache_dir.mkdir(parents=True)
        np.savez(cache_dir / f"{tensor_name}.npz",
                 indices=packed, shape=np.array([M, K]), bits=np.array([bits]),
                 mode=np.array(["direct_codebook"]))

        # Export
        export_flat_idx(tmp_path)

        idx_file = cache_dir / f"{tensor_name}.idx"
        assert idx_file.exists(), ".idx file not created"

        buf = MmappedPackedBuffer(idx_file)
        np.testing.assert_array_equal(buf.as_numpy()[:len(packed)], packed,
            err_msg="exported .idx bytes differ from original packed array")
        buf.close()
