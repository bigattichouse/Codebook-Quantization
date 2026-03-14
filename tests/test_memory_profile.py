"""
Memory profile tests: verify no leaks during compress/decompress cycles.

Also covers the specific bugs that caused 100GB RAM explosion during inference:

1. _cached_weight accumulation: AdaptiveCodebookLinear must NOT permanently
   cache decompressed weights. 250 layers × ~144MB = 36GB was the primary cause.

2. FastIndexManager byte_offsets/bit_shifts waste: prepare_lookup_table was
   allocating O(N) arrays that _fast_packed_lookup never used.
   For lm_head (1B params, 13-bit): 4GB + 1GB = 5GB wasted per LRU entry.

3. _load_exact_weights loading codebook tensors: decompressed ALL tensors
   including direct_codebook ones immediately overwritten by module replacement.
"""

import pytest
import numpy as np
import torch
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compressor import kmeans_1d, assign_to_codebook, float32_to_bfloat16
from bitpack import pack_any_bits, unpack_any_bits
from fast_index_manager import FastIndexManager
from compressed_modules import AdaptiveCodebookLinear

psutil = pytest.importorskip("psutil")


def _get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


_layer_counter = 0

def _make_codebook_layer(n_rows=1024, n_cols=512, bits=13):
    """Create an AdaptiveCodebookLinear with synthetic compressed data.

    Uses a unique name per layer so the global FastIndexManager singleton
    doesn't confuse entries across tests.
    """
    global _layer_counter
    _layer_counter += 1
    n_elements = n_rows * n_cols
    codebook_size = 2 ** bits
    codebook = torch.randn(codebook_size, dtype=torch.float32) * 0.02
    raw_indices = np.random.randint(0, codebook_size, size=n_elements, dtype=np.uint16)
    packed = pack_any_bits(raw_indices, bits)

    layer = AdaptiveCodebookLinear(f"test.weight.{_layer_counter}", (n_rows, n_cols), 'direct_codebook')
    layer.bits = bits
    layer.register_buffer('codebook', codebook, persistent=False)
    layer.register_buffer('indices', torch.from_numpy(packed), persistent=False)
    layer.original_len = n_elements
    return layer


# ── Original leak tests ───────────────────────────────────────────────────────

class TestNoMemoryLeak:
    """Compress and decompress a synthetic tensor repeatedly; RSS should not grow."""

    def test_compress_decompress_no_leak(self):
        """10 rounds of compress+decompress should not grow RSS by more than 50 MB."""
        gc.collect()
        rss_before = _get_rss_mb()

        for _ in range(10):
            data = np.random.randn(100_000).astype(np.float32) * 0.02
            cb = kmeans_1d(data, k=256, seed=42)
            idx = assign_to_codebook(data, cb)
            packed = pack_any_bits(idx, 8)
            unpacked = unpack_any_bits(packed, 8, len(data))
            _ = cb[unpacked]
            del data, cb, idx, packed, unpacked
            gc.collect()

        rss_after = _get_rss_mb()
        growth = rss_after - rss_before
        assert growth < 50, f"RSS grew by {growth:.1f} MB over 10 iterations"

    def test_bf16_roundtrip_no_leak(self):
        """BF16 encode/decode should not leak."""
        gc.collect()
        rss_before = _get_rss_mb()

        for _ in range(20):
            data = np.random.randn(500_000).astype(np.float32)
            bf16 = float32_to_bfloat16(data)
            _ = (bf16.astype(np.uint32) << 16).view(np.float32)
            del data, bf16
            gc.collect()

        rss_after = _get_rss_mb()
        growth = rss_after - rss_before
        assert growth < 50, f"RSS grew by {growth:.1f} MB over 20 iterations"


# ── Bug 1: _cached_weight must not accumulate ────────────────────────────────

class TestNoCachedWeightLeak:
    """AdaptiveCodebookLinear must not permanently cache decompressed weights."""

    def test_no_cached_weight_after_forward(self):
        """_cached_weight must be None after a forward call."""
        layer = _make_codebook_layer(64, 32, bits=8)
        x = torch.randn(2, 32)
        with torch.no_grad():
            _ = layer(x)
        assert layer._cached_weight is None, (
            "_cached_weight must be None after forward — persistent caching causes "
            "RAM explosion: 250 layers × decompressed weight size for a 9B model."
        )

    def test_two_layers_no_double_accumulation(self):
        """Two separate layers must not each hold a cached weight after forward."""
        # (32, 32): takes (*, 32) → outputs (*, 32) so they can chain
        layer_a = _make_codebook_layer(32, 32, bits=8)
        layer_b = _make_codebook_layer(32, 32, bits=8)
        x = torch.randn(2, 32)
        with torch.no_grad():
            _ = layer_b(layer_a(x))
        assert layer_a._cached_weight is None
        assert layer_b._cached_weight is None

    def test_rss_stable_across_many_forward_calls(self):
        """RSS growth from 20 forward calls must be small (< 20 MB)."""
        layer = _make_codebook_layer(256, 128, bits=13)
        x = torch.randn(4, 128)
        gc.collect()
        rss_before = _get_rss_mb()
        with torch.no_grad():
            for _ in range(20):
                _ = layer(x)
        gc.collect()
        rss_after = _get_rss_mb()
        growth = rss_after - rss_before
        assert growth < 20, (
            f"RSS grew {growth:.1f} MB across 20 forward calls — "
            f"decompressed weight is likely accumulating"
        )

    def test_many_layers_rss_stable(self):
        """Simulating 28 layers: RSS should not grow by (28 × weight_size)."""
        layers = [_make_codebook_layer(64, 64, bits=13) for _ in range(28)]
        x = torch.randn(1, 64)
        gc.collect()
        rss_before = _get_rss_mb()
        with torch.no_grad():
            out = x
            for layer in layers:
                out = layer(out)
        gc.collect()
        rss_after = _get_rss_mb()
        growth = rss_after - rss_before
        # 28 × 64×64×4 bytes = 28 × 16KB = ~450KB, allow 50× headroom
        assert growth < 50, (
            f"RSS grew {growth:.1f} MB running 28 layers — "
            f"likely caching all decompressed weights simultaneously"
        )


# ── Bug 2: FastIndexManager no unused O(N) arrays ────────────────────────────

class TestFastIndexManagerMemory:
    """prepare_lookup_table must not store unused byte_offsets/bit_shifts."""

    def test_no_byte_offsets_stored(self):
        """byte_offsets must not be in the lookup table entry."""
        mgr = FastIndexManager()
        packed = torch.from_numpy(pack_any_bits(
            np.random.randint(0, 2**13, size=1000, dtype=np.uint16), 13
        ))
        mgr.prepare_lookup_table("t", packed, 13)
        assert 'byte_offsets' not in mgr.lookup_tables["t"], (
            "byte_offsets wastes 4×N bytes and is never used by _fast_packed_lookup. "
            "For 1B-param tensors this is 4 GB per LRU slot."
        )

    def test_no_bit_shifts_stored(self):
        """bit_shifts must not be in the lookup table entry."""
        mgr = FastIndexManager()
        packed = torch.from_numpy(pack_any_bits(
            np.random.randint(0, 2**13, size=1000, dtype=np.uint16), 13
        ))
        mgr.prepare_lookup_table("t", packed, 13)
        assert 'bit_shifts' not in mgr.lookup_tables["t"], (
            "bit_shifts wastes N bytes and is never used by _fast_packed_lookup."
        )

    def test_only_expected_keys_in_entry(self):
        """Lookup table entry should have only the fields actually used."""
        mgr = FastIndexManager()
        packed = torch.from_numpy(pack_any_bits(
            np.random.randint(0, 2**13, size=1000, dtype=np.uint16), 13
        ))
        mgr.prepare_lookup_table("t", packed, 13)
        allowed = {'type', 'indices', 'bits', 'total_elements'}
        unexpected = set(mgr.lookup_tables["t"].keys()) - allowed
        assert not unexpected, f"Unexpected keys waste RAM: {unexpected}"

    def test_8bit_lookup_correct(self):
        """8-bit path must still return correct results."""
        mgr = FastIndexManager()
        indices = np.array([0, 5, 255, 128, 1], dtype=np.uint8)
        mgr.prepare_lookup_table("t8", torch.from_numpy(indices), 8)
        result = mgr.fast_index_lookup("t8", 5)
        np.testing.assert_array_equal(result.numpy(), indices.astype(np.int64))

    def test_13bit_lookup_correct(self):
        """13-bit lookup must return correct indices after cleanup."""
        mgr = FastIndexManager()
        n, bits = 200, 13
        original = np.random.randint(0, 2**bits, size=n, dtype=np.uint16)
        packed = torch.from_numpy(pack_any_bits(original, bits))
        mgr.prepare_lookup_table("t13", packed, bits)
        result = mgr.fast_index_lookup("t13", n)
        np.testing.assert_array_equal(result.numpy(), original.astype(np.int64))

    def test_lru_eviction_still_works(self):
        """LRU eviction must still trigger when max_lookup_tables is reached."""
        mgr = FastIndexManager(max_lookup_tables=3)
        for i in range(4):
            packed = torch.from_numpy(np.random.randint(0, 256, size=100, dtype=np.uint8))
            mgr.prepare_lookup_table(f"tensor_{i}", packed, 8)
        assert len(mgr.lookup_tables) <= 3


# ── Forward correctness: no regression from removing cache ───────────────────

class TestForwardCorrectness:
    """Forward output must remain correct even without persistent _cached_weight."""

    def test_direct_codebook_matches_dense(self):
        """Decompressed-on-the-fly output must match explicit dense matmul."""
        torch.manual_seed(0)
        n_rows, n_cols, codebook_size = 32, 16, 256
        codebook = torch.randn(codebook_size) * 0.02
        raw_indices = np.random.randint(0, codebook_size, size=n_rows * n_cols, dtype=np.uint16)
        weight_dense = codebook[torch.from_numpy(raw_indices.astype(np.int64))].reshape(n_rows, n_cols)

        layer = AdaptiveCodebookLinear("test.w", (n_rows, n_cols), 'direct_codebook')
        layer.bits = 8
        layer.register_buffer('codebook', codebook, persistent=False)
        layer.register_buffer('indices', torch.from_numpy(pack_any_bits(raw_indices, 8)), persistent=False)

        x = torch.randn(4, n_cols)
        with torch.no_grad():
            out_compressed = layer(x)
            out_dense = torch.nn.functional.linear(x, weight_dense)
        torch.testing.assert_close(out_compressed, out_dense, atol=1e-5, rtol=1e-4)

    def test_forward_reproducible_across_calls(self):
        """Two forward calls must produce identical output (no stochastic decompression)."""
        layer = _make_codebook_layer(32, 16, bits=13)
        x = torch.randn(2, 16)
        with torch.no_grad():
            out1 = layer(x).clone()
            out2 = layer(x).clone()
        torch.testing.assert_close(out1, out2)
