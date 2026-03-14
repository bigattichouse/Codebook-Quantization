"""
Bit-Packing Utilities

Pack indices into minimal bit representation for efficient storage.
Supports generic N-bit packing for flexible compression ratios.

Generic path uses vectorized group packing:
  - Compute g = lcm(bits, 8): the smallest number of bits that aligns to a byte boundary.
  - Group size = g // bits values, g // 8 output bytes.
  - Pack each group into two uint64s (lo covers bits 0-63, hi covers 64-127).
  - No Python loops over individual values — loop is over at most lcm/bits iterations
    (max 8 for bits=9/11/13/15), each body is a vectorized numpy op over all groups.

Example for bits=13: lcm(13,8)=104 → 8 values → 13 bytes per group.
  1B values / 8 per group = 125M vectorized groups — ~1000x faster than element loop.
"""

import numpy as np
from math import gcd as _gcd
from typing import Tuple


def _group_params(bits: int) -> Tuple[int, int]:
    """Return (group_values, group_bytes) for a given bit width.

    group_values values of `bits` bits each pack exactly into group_bytes bytes.
    """
    g = (bits * 8) // _gcd(bits, 8)   # lcm(bits, 8)
    return g // bits, g // 8


def pack_any_bits(indices: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack indices into a bitstream using a generic bit-width.
    Optimized fast paths for 4, 8, and 16 bits.
    Generic path uses vectorized uint64 group packing (no Python loop over values).
    """
    indices = np.asarray(indices).flatten()

    # --- FAST PATH: 8-bit ---
    if bits == 8:
        return indices.astype(np.uint8)

    # --- FAST PATH: 4-bit (vectorized nibble packing) ---
    if bits == 4:
        n = len(indices)
        if n % 2:
            indices = np.append(indices, 0)
        return ((indices[1::2].astype(np.uint8) & 0x0F) << 4) | (indices[0::2].astype(np.uint8) & 0x0F)

    if bits == 16:
        return indices.astype(np.uint16).view(np.uint8).copy()

    # --- Vectorized group packing ---
    # For bits=13: group_values=8, group_bytes=13.
    # Loop over at most 8 values/group; each body is a numpy op on all M groups.
    n = len(indices)
    group_values, group_bytes = _group_params(bits)

    # Pad to multiple of group_values
    pad = (-n) % group_values
    if pad:
        indices = np.concatenate([indices, np.zeros(pad, dtype=np.uint64)])
    M = len(indices) // group_values
    vals = indices.astype(np.uint64).reshape(M, group_values)

    lo = np.zeros(M, dtype=np.uint64)
    hi = np.zeros(M, dtype=np.uint64)

    for i in range(group_values):
        bit_pos = i * bits
        v = vals[:, i]
        if bit_pos + bits <= 64:
            lo |= v << np.uint64(bit_pos)
        elif bit_pos >= 64:
            hi |= v << np.uint64(bit_pos - 64)
        else:
            # Split: lower (64-bit_pos) bits into lo, remainder into hi
            lo_bits = np.uint64(64 - bit_pos)
            lo |= v << np.uint64(bit_pos)   # overflow beyond bit 63 is silently discarded ✓
            hi |= v >> lo_bits

    # Lay out bytes: lo (8 bytes) then hi (up to 8 bytes) per group
    lo_u8 = lo.view(np.uint8).reshape(M, 8)
    result = np.empty((M, group_bytes), dtype=np.uint8)
    if group_bytes <= 8:
        result[:] = lo_u8[:, :group_bytes]
    else:
        hi_u8 = hi.view(np.uint8).reshape(M, 8)
        result[:, :8] = lo_u8
        result[:, 8:group_bytes] = hi_u8[:, :group_bytes - 8]

    total_bytes = (n * bits + 7) // 8
    return result.reshape(-1)[:total_bytes]


def unpack_any_bits(packed: np.ndarray, bits: int, original_len: int) -> np.ndarray:
    """
    Unpack indices from a generic bitstream.
    Optimized fast paths for 4, 8, and 16 bits.
    Generic path uses vectorized uint64 group unpacking.
    """
    # --- FAST PATH: 8-bit ---
    if bits == 8:
        return packed[:original_len].astype(np.uint16)

    # --- FAST PATH: 4-bit ---
    if bits == 4:
        unpacked = np.zeros(len(packed) * 2, dtype=np.uint16)
        unpacked[0::2] = packed & 0x0F
        unpacked[1::2] = (packed >> 4) & 0x0F
        return unpacked[:original_len]

    if bits == 16:
        return packed.view(np.uint16)[:original_len]

    # --- Vectorized group unpacking ---
    group_values, group_bytes = _group_params(bits)
    M = (original_len + group_values - 1) // group_values

    # Ensure we have enough bytes
    total_bytes = M * group_bytes
    if len(packed) < total_bytes:
        packed = np.concatenate([packed, np.zeros(total_bytes - len(packed), dtype=np.uint8)])

    groups_u8 = packed[:total_bytes].reshape(M, group_bytes)

    # Load bytes into lo uint64
    lo_buf = np.zeros((M, 8), dtype=np.uint8)
    lo_buf[:, :min(group_bytes, 8)] = groups_u8[:, :min(group_bytes, 8)]
    lo = lo_buf.view(np.uint64).reshape(M)

    # Load remaining bytes into hi uint64 (only needed when group_bytes > 8)
    hi = np.zeros(M, dtype=np.uint64)
    if group_bytes > 8:
        hi_buf = np.zeros((M, 8), dtype=np.uint8)
        hi_buf[:, :group_bytes - 8] = groups_u8[:, 8:group_bytes]
        hi = hi_buf.view(np.uint64).reshape(M)

    mask = np.uint64((1 << bits) - 1)
    result = np.zeros((M, group_values), dtype=np.uint16)

    for i in range(group_values):
        bit_pos = i * bits
        if bit_pos + bits <= 64:
            result[:, i] = ((lo >> np.uint64(bit_pos)) & mask).astype(np.uint16)
        elif bit_pos >= 64:
            result[:, i] = ((hi >> np.uint64(bit_pos - 64)) & mask).astype(np.uint16)
        else:
            # Split: gather bits from both lo and hi
            lo_bits = 64 - bit_pos
            lo_part = lo >> np.uint64(bit_pos)
            hi_part = hi & np.uint64((1 << (bits - lo_bits)) - 1)
            result[:, i] = (lo_part | (hi_part << np.uint64(lo_bits))).astype(np.uint16)

    return result.reshape(-1)[:original_len]


# Keep legacy functions for compatibility but redirect to generic ones
def pack_7bit_indices(indices: np.ndarray) -> np.ndarray:
    return pack_any_bits(indices, 7)

def unpack_7bit_indices(packed: np.ndarray, original_len: int) -> np.ndarray:
    return unpack_any_bits(packed, 7, original_len)

def pack_6bit_indices(indices: np.ndarray) -> np.ndarray:
    return pack_any_bits(indices, 6)

def unpack_6bit_indices(packed: np.ndarray, original_len: int) -> np.ndarray:
    return unpack_any_bits(packed, 6, original_len)

def pack_4bit_indices(indices: np.ndarray) -> np.ndarray:
    return pack_any_bits(indices, 4)

def unpack_4bit_indices(packed: np.ndarray, original_len: int) -> np.ndarray:
    return unpack_any_bits(packed, 4, original_len)

def pack_indices_minimal(indices: np.ndarray, bits: int) -> np.ndarray:
    return pack_any_bits(indices, bits)

def unpack_indices_minimal(packed: np.ndarray, bits: int, original_len: int) -> np.ndarray:
    return unpack_any_bits(packed, bits, original_len)

def calculate_packed_size(num_indices: int, bits: int) -> int:
    return (num_indices * bits + 7) // 8
