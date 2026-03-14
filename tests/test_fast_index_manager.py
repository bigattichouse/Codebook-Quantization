"""
Tests for FastIndexManager: pack -> prepare -> lookup -> verify exact match.
"""

import pytest
import torch
import numpy as np

from fast_index_manager import FastIndexManager, get_index_manager
from bitpack import pack_any_bits


class TestFastIndexManager:

    @pytest.mark.parametrize("bits", [4, 7, 8, 13])
    def test_roundtrip(self, bits):
        """Pack indices, prepare lookup table, retrieve — must match originals."""
        n = 5000
        max_val = (1 << bits) - 1
        original = np.random.randint(0, max_val + 1, size=n, dtype=np.uint16)

        if bits == 8:
            packed = original.astype(np.uint8)
        else:
            packed = pack_any_bits(original, bits)

        mgr = FastIndexManager(device="cpu")
        mgr.prepare_lookup_table("test", torch.from_numpy(packed), bits)
        result = mgr.fast_index_lookup("test", n)

        np.testing.assert_array_equal(result.numpy(), original.astype(np.int64))

    def test_start_offset(self):
        """Lookup with a non-zero start_offset should return correct slice."""
        bits = 7
        n = 1000
        original = np.random.randint(0, 128, size=n, dtype=np.uint16)
        packed = pack_any_bits(original, bits)

        mgr = FastIndexManager(device="cpu")
        mgr.prepare_lookup_table("test", torch.from_numpy(packed), bits)

        offset = 100
        count = 200
        result = mgr.fast_index_lookup("test", count, start_offset=offset)
        expected = original[offset : offset + count].astype(np.int64)
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_8bit_direct(self):
        """8-bit mode uses direct lookup (no bit unpacking)."""
        n = 500
        original = np.random.randint(0, 256, size=n, dtype=np.uint8)
        mgr = FastIndexManager(device="cpu")
        mgr.prepare_lookup_table("direct8", torch.from_numpy(original), 8)
        info = mgr.lookup_tables["direct8"]
        assert info["type"] == "direct"
        result = mgr.fast_index_lookup("direct8", n)
        np.testing.assert_array_equal(result.numpy(), original.astype(np.int64))

    def test_eviction(self):
        """LRU eviction should keep table count within limits."""
        mgr = FastIndexManager(device="cpu", max_lookup_tables=3)
        for i in range(5):
            data = torch.from_numpy(np.random.randint(0, 256, size=100, dtype=np.uint8))
            mgr.prepare_lookup_table(f"t{i}", data, 8)
        assert len(mgr.lookup_tables) <= 3

    def test_missing_table_raises(self):
        mgr = FastIndexManager(device="cpu")
        with pytest.raises(KeyError):
            mgr.fast_index_lookup("nonexistent", 10)


class TestGlobalSingleton:
    def test_returns_same_instance(self):
        a = get_index_manager("cpu")
        b = get_index_manager("cpu")
        assert a is b
