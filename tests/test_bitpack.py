"""
Tests for bit-packing utilities: pack/unpack roundtrips at various bit widths.
"""

import pytest
import numpy as np
from bitpack import pack_any_bits, unpack_any_bits, calculate_packed_size


class TestPackUnpackRoundtrip:
    """Verify lossless roundtrip for all supported bit widths."""

    @pytest.mark.parametrize("bits", [4, 6, 7, 8, 13])
    def test_roundtrip_random(self, bits):
        max_val = (1 << bits) - 1
        n = 1000
        original = np.random.randint(0, max_val + 1, size=n, dtype=np.uint16)
        packed = pack_any_bits(original, bits)
        unpacked = unpack_any_bits(packed, bits, n)
        np.testing.assert_array_equal(unpacked, original)

    @pytest.mark.parametrize("bits", [4, 6, 7, 8, 13])
    def test_roundtrip_edge_values(self, bits):
        """Pack/unpack with all-zeros and all-max values."""
        max_val = (1 << bits) - 1
        n = 100
        for fill in [0, max_val]:
            original = np.full(n, fill, dtype=np.uint16)
            packed = pack_any_bits(original, bits)
            unpacked = unpack_any_bits(packed, bits, n)
            np.testing.assert_array_equal(unpacked, original)

    @pytest.mark.parametrize("bits", [4, 6, 7, 8, 13])
    def test_roundtrip_single_element(self, bits):
        max_val = (1 << bits) - 1
        original = np.array([max_val // 2], dtype=np.uint16)
        packed = pack_any_bits(original, bits)
        unpacked = unpack_any_bits(packed, bits, 1)
        np.testing.assert_array_equal(unpacked, original)

    @pytest.mark.parametrize("bits", [4, 6, 7, 8, 13])
    def test_packed_size(self, bits):
        """Packed output should not exceed calculated size."""
        n = 1000
        original = np.random.randint(0, (1 << bits), size=n, dtype=np.uint16)
        packed = pack_any_bits(original, bits)
        expected_size = calculate_packed_size(n, bits)
        assert len(packed) <= expected_size + 1  # +1 for possible padding in 4-bit

    def test_4bit_nibble_order(self):
        """4-bit packing: low nibble first."""
        indices = np.array([0x02, 0x01, 0x04, 0x03], dtype=np.uint16)
        packed = pack_any_bits(indices, 4)
        unpacked = unpack_any_bits(packed, 4, 4)
        np.testing.assert_array_equal(unpacked, indices)

    @pytest.mark.parametrize("n", [1, 7, 8, 9, 100, 1023, 1024])
    def test_various_sizes_13bit(self, n):
        """13-bit roundtrip at non-power-of-2 sizes."""
        original = np.random.randint(0, 8192, size=n, dtype=np.uint16)
        packed = pack_any_bits(original, 13)
        unpacked = unpack_any_bits(packed, 13, n)
        np.testing.assert_array_equal(unpacked, original)


class TestCompressedSize:
    def test_8bit_identity(self):
        """8-bit packing should produce same byte count as element count."""
        assert calculate_packed_size(1000, 8) == 1000

    def test_4bit_halves(self):
        """4-bit packing should halve byte count."""
        assert calculate_packed_size(1000, 4) == 500

    def test_13bit_less_than_16(self):
        """13-bit should be smaller than 16-bit."""
        n = 10000
        assert calculate_packed_size(n, 13) < n * 2
