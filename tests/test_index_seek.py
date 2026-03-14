"""
Tests for FastIndexManager direct seek optimization.

Reviewer noted: "FastIndexManager currently unpacks from element 0 up to end
for every call. It should calculate the exact byte-offset and bit-shift for
the requested start_offset and unpack only the required bytes."

These tests define the expected behaviour and serve as the spec for the fix.
Current implementation may be slow (but still correct) on large offsets.
"""

import os
import sys
import time
import gc
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bitpack import pack_any_bits, unpack_any_bits, _group_params
from fast_index_manager import FastIndexManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack(n_values: int, bits: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_indices uint16, packed uint8) for n_values random indices."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2 ** bits, size=n_values, dtype=np.uint16)
    packed = pack_any_bits(raw, bits)
    return raw, packed


def _make_mgr() -> FastIndexManager:
    return FastIndexManager(device='cpu', max_lookup_tables=512)


# ---------------------------------------------------------------------------
# Correctness: seek must return exactly the same values as a full unpack+slice
# ---------------------------------------------------------------------------

class TestSeekCorrectness:
    """fast_index_lookup(start_offset=X) must equal unpack_all()[X:X+N]."""

    @pytest.mark.parametrize("bits", [8, 13])
    @pytest.mark.parametrize("offset_frac", [0.0, 0.25, 0.5, 0.99])
    def test_seek_matches_full_unpack(self, bits, offset_frac):
        """Seeking to any offset must produce the same values as slicing a full unpack."""
        total = 8192  # power-of-group-size multiple
        hidden = 64
        raw, packed = _pack(total, bits)

        mgr = _make_mgr()
        mgr.prepare_lookup_table("t", packed, bits)

        start = int(total * offset_frac) // hidden * hidden  # align to hidden
        result = mgr.fast_index_lookup("t", hidden, start_offset=start)

        expected = torch.from_numpy(raw[start: start + hidden].astype(np.int64))
        torch.testing.assert_close(result.cpu(), expected,
            msg=f"bits={bits} offset_frac={offset_frac} start={start}")

    @pytest.mark.parametrize("bits", [8, 13])
    def test_seek_first_element(self, bits):
        """start_offset=0 must return the first `hidden` elements."""
        total, hidden = 4096, 32
        raw, packed = _pack(total, bits)
        mgr = _make_mgr()
        mgr.prepare_lookup_table("t0", packed, bits)
        result = mgr.fast_index_lookup("t0", hidden, start_offset=0)
        expected = torch.from_numpy(raw[:hidden].astype(np.int64))
        torch.testing.assert_close(result.cpu(), expected)

    @pytest.mark.parametrize("bits", [8, 13])
    def test_seek_last_row(self, bits):
        """Seeking to the last row must work without out-of-bounds."""
        vocab, hidden = 1024, 32
        total = vocab * hidden
        raw, packed = _pack(total, bits)
        mgr = _make_mgr()
        mgr.prepare_lookup_table("tlast", packed, bits)
        start = (vocab - 1) * hidden
        result = mgr.fast_index_lookup("tlast", hidden, start_offset=start)
        expected = torch.from_numpy(raw[start: start + hidden].astype(np.int64))
        torch.testing.assert_close(result.cpu(), expected)

    def test_seek_13bit_group_boundary(self):
        """
        For 13-bit packing, group_values=8. Test offsets that fall at group
        boundaries (multiples of 8) and within groups (e.g. offset=3).
        """
        group_values, _ = _group_params(13)
        assert group_values == 8, "sanity: 13-bit group should be 8 values"

        total = 128
        raw, packed = _pack(total, bits=13)
        mgr = _make_mgr()
        mgr.prepare_lookup_table("tgrp", packed, 13)

        for start in [0, 8, 16, 3, 5, 11, 24]:
            n = min(8, total - start)
            if n <= 0:
                continue
            result = mgr.fast_index_lookup("tgrp", n, start_offset=start)
            expected = torch.from_numpy(raw[start: start + n].astype(np.int64))
            torch.testing.assert_close(result.cpu(), expected,
                msg=f"start={start} n={n}")

    def test_large_vocab_high_token_id(self):
        """
        Simulate embedding lookup for token near end of a large vocab.
        vocab=32768, hidden=128, bits=13 — token_id=32700 (>99th percentile).
        Result must match a direct unpack+slice.
        """
        vocab, hidden, bits = 32768, 128, 13
        total = vocab * hidden
        raw, packed = _pack(total, bits, seed=99)

        mgr = _make_mgr()
        mgr.prepare_lookup_table("embed", packed, bits)

        tok = 32700
        start = tok * hidden
        result = mgr.fast_index_lookup("embed", hidden, start_offset=start)
        expected = torch.from_numpy(raw[start: start + hidden].astype(np.int64))
        torch.testing.assert_close(result.cpu(), expected,
            msg=f"Token {tok} lookup mismatch")


# ---------------------------------------------------------------------------
# Performance: seeking must NOT scale with start_offset size
# ---------------------------------------------------------------------------

class TestSeekPerformance:
    """
    The seek operation should be O(target_elements), not O(start_offset).
    We test this by checking that looking up a late token is not much slower
    than looking up an early token for a large vocab embedding tensor.
    """

    def _time_lookup(self, mgr, name, hidden, start, n_reps=5):
        # warm up
        mgr.fast_index_lookup(name, hidden, start_offset=start)
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            mgr.fast_index_lookup(name, hidden, start_offset=start)
        return (time.perf_counter() - t0) / n_reps

    def test_late_token_not_slower_than_early_token(self):
        """
        Looking up the last token should not be significantly slower
        than looking up the first token. Acceptable ratio: <= 5×.
        (Current unfixed implementation has ratio >> 100×.)

        NOTE: If this test fails, implement direct group-seek in
        FastIndexManager._fast_packed_lookup.
        """
        vocab, hidden, bits = 16384, 128, 13
        total = vocab * hidden
        raw, packed = _pack(total, bits, seed=0)

        mgr = _make_mgr()
        mgr.prepare_lookup_table("perf_embed", packed, bits)

        early_ms = self._time_lookup(mgr, "perf_embed", hidden, start=0) * 1000
        late_ms  = self._time_lookup(mgr, "perf_embed", hidden,
                                      start=(vocab - 1) * hidden) * 1000
        ratio = late_ms / early_ms if early_ms > 0 else float('inf')
        print(f"\n  early={early_ms:.2f}ms  late={late_ms:.2f}ms  ratio={ratio:.1f}×")
        assert ratio <= 5.0, (
            f"Late token lookup is {ratio:.1f}× slower than early token "
            f"(early={early_ms:.2f}ms, late={late_ms:.2f}ms). "
            f"FastIndexManager._fast_packed_lookup must use direct byte-seek, "
            f"not unpack-from-0."
        )

    def test_seek_time_constant_across_offsets(self):
        """
        Lookup time must not grow linearly with start_offset.
        Run 5 lookups at linearly spaced offsets; max/min ratio must be <= 5.
        """
        vocab, hidden, bits = 8192, 64, 13
        total = vocab * hidden
        raw, packed = _pack(total, bits, seed=1)

        mgr = _make_mgr()
        mgr.prepare_lookup_table("perf2", packed, bits)

        offsets = [i * (vocab // 5) * hidden for i in range(5)]
        times = []
        for start in offsets:
            t = self._time_lookup(mgr, "perf2", hidden, start, n_reps=10)
            times.append(t)

        ratio = max(times) / min(times) if min(times) > 0 else float('inf')
        print(f"\n  times(ms): {[f'{t*1000:.2f}' for t in times]}  ratio={ratio:.1f}×")
        assert ratio <= 5.0, (
            f"Lookup time varies {ratio:.1f}× across offsets — should be O(1) in offset. "
            f"Times: {[f'{t*1000:.2f}ms' for t in times]}"
        )


# ---------------------------------------------------------------------------
# Correctness: embedding decode via AdaptiveCodebookEmbedding
# ---------------------------------------------------------------------------

class TestEmbeddingDecodeCorrectness:
    """
    End-to-end: AdaptiveCodebookEmbedding.forward() for high token IDs must
    produce the same values as explicit decompression.
    """

    def _make_embedding_layer(self, vocab, hidden, bits, seed=42):
        from compressed_modules import AdaptiveCodebookEmbedding
        rng = np.random.default_rng(seed)
        C = min(2 ** bits, 8192)
        raw = rng.integers(0, C, size=vocab * hidden, dtype=np.uint16)
        packed = pack_any_bits(raw, bits)
        codebook = rng.standard_normal(C).astype(np.float32) * 0.02

        layer = AdaptiveCodebookEmbedding(f"embed.{vocab}.{hidden}", (vocab, hidden), 'direct_codebook')
        layer.bits = bits
        layer.register_buffer('codebook', torch.from_numpy(codebook), persistent=False)
        layer.register_buffer('indices', torch.from_numpy(packed), persistent=False)
        return layer, raw, codebook

    @pytest.mark.parametrize("tok_id", [0, 1, 100, 999, 4095])
    def test_token_decode_matches_explicit(self, tok_id):
        """Embedding output for token tok_id must match explicit codebook[raw[tok_id*hidden:]]."""
        vocab, hidden, bits = 4096, 32, 13
        layer, raw, codebook = self._make_embedding_layer(vocab, hidden, bits)

        with torch.no_grad():
            out = layer(torch.tensor([[tok_id]]))  # (1, 1, hidden)

        expected_raw = raw[tok_id * hidden: tok_id * hidden + hidden]
        expected = codebook[expected_raw]
        np.testing.assert_allclose(
            out.squeeze().float().numpy(), expected, atol=1e-5,
            err_msg=f"Embedding mismatch for token {tok_id}"
        )

    def test_batch_of_tokens_matches_explicit(self):
        """A batch of token IDs must all decode correctly."""
        vocab, hidden, bits = 4096, 32, 13
        layer, raw, codebook = self._make_embedding_layer(vocab, hidden, bits)

        tok_ids = [0, 10, 100, 500, 4095]
        with torch.no_grad():
            out = layer(torch.tensor([tok_ids]))  # (1, 5, hidden)

        for i, tok in enumerate(tok_ids):
            expected_raw = raw[tok * hidden: tok * hidden + hidden]
            expected = codebook[expected_raw]
            np.testing.assert_allclose(
                out[0, i].float().numpy(), expected, atol=1e-5,
                err_msg=f"Mismatch for token {tok}"
            )
