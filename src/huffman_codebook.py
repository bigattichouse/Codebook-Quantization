"""
Huffman entropy coding for codebook index streams.

Why this helps:
  After codebook assignment, indices are frequency-sorted (index 0 = most common
  weight value).  The distribution is non-uniform — a fixed-width 13-bit code
  wastes (13 - H) bits per weight where H is the Shannon entropy (~10 bits for
  typical LLM weights).  Replacing LCM bit-packing with canonical Huffman codes
  reclaims that gap, giving ~18% additional compression on the index stream.

  This is the same insight exploited by DFloat11 (Zhao et al. 2025,
  https://arxiv.org/abs/2412.19437), which Huffman-codes BFloat16 exponent bytes
  because their distribution is also highly non-uniform.  We apply the same
  technique to the full codebook index distribution, which is even more
  concentrated when indices are frequency-sorted.

Storage format (NPZ fields):
  huff_lengths : uint8[C]     canonical code length per symbol (0 = unused)
  huff_stream  : uint8[N_B]   packed Huffman bitstream (MSB-first within each byte)
  huff_n       : int64[1]     number of encoded symbols

Load-time decode (CPU):
  indices = huffman_decode_indices(huff_stream, huff_lengths, huff_n)
  packed  = pack_any_bits(indices, bits)   # then → existing GPU kernel unchanged

References:
  DFloat11: "No Free Lunch in Neural Network Compression" (Zhao et al., 2025)
    https://arxiv.org/abs/2412.19437
  Canonical Huffman codes: RFC 1951 (DEFLATE), Section 3.2.2
"""

from __future__ import annotations
import heapq
import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def _build_code_lengths(freq: np.ndarray) -> np.ndarray:
    """
    Build optimal Huffman code lengths from a frequency array.
    Returns uint8 array of code lengths (0 = unused symbol).
    Length is capped at 32 bits.  Simple truncation at 24 causes overcomplete trees
    (Kraft sum > 1) for skewed distributions with depths 25-26, which makes
    canonical code assignment overflow 24 bits and corrupts the bitstream.
    32-bit cap is always sufficient for realistic LLM weight distributions
    (natural depths ≤ 28 for up to 65536 unique values) without truncation.
    """
    n = len(freq)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    active = [(int(freq[i]), i) for i in range(n) if freq[i] > 0]
    if not active:
        return np.zeros(n, dtype=np.uint8)
    if len(active) == 1:
        lengths = np.zeros(n, dtype=np.uint8)
        lengths[active[0][1]] = 1
        return lengths

    heapq.heapify(active)
    parent: Dict[int, int] = {}
    nid = n  # internal node IDs start above leaf range
    while len(active) > 1:
        f1, a = heapq.heappop(active)
        f2, b = heapq.heappop(active)
        parent[a] = nid
        parent[b] = nid
        heapq.heappush(active, (f1 + f2, nid))
        nid += 1

    root = active[0][1]
    lengths = np.zeros(n, dtype=np.uint8)
    for sym in range(n):
        if freq[sym] == 0:
            continue
        depth, node = 0, sym
        while node != root:
            node = parent[node]
            depth += 1
        lengths[sym] = min(depth, 32)  # cap at 32; 24 caused overcomplete trees for depths 25-26

    return lengths


def _canonical_codes(lengths: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """
    Assign canonical Huffman codes from a code-length array.
    Returns {symbol: (code_int, code_len)}.
    Canonical codes are deterministic given lengths → only lengths need storing.
    """
    max_len = int(lengths.max()) if len(lengths) and lengths.max() > 0 else 0
    if max_len == 0:
        return {}

    # Count symbols per length
    bl_count = np.zeros(max_len + 1, dtype=np.int64)
    for sym, L in enumerate(lengths):
        if L > 0:
            bl_count[L] += 1

    # First code for each length (RFC 1951 §3.2.2)
    next_code = np.zeros(max_len + 2, dtype=np.int64)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    # Assign canonical codes (symbols sorted by length, then by symbol value)
    codes: Dict[int, Tuple[int, int]] = {}
    for sym in range(len(lengths)):
        L = int(lengths[sym])
        if L > 0:
            codes[sym] = (int(next_code[L]), L)
            next_code[L] += 1
    return codes


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def huffman_encode_indices(indices: np.ndarray, shape: tuple = None) -> dict:
    """
    Huffman-encode a flat uint16 index array.

    Args:
        indices: 1-D or N-D array of codebook indices (uint16).
        shape:   Optional (M, K) weight matrix shape.  When provided, the
                 returned dict also includes Phase 2 GPU decode tables:
                   huff_row_bit_starts : int64[M]    bit offset per weight row
                   huff_lut_sym        : uint16[4096] 12-bit GPU LUT symbols
                   huff_lut_len        : uint8[4096]  12-bit GPU LUT lengths
                   huff_sl_first_code  : int64[max+2] slow-path first codes
                   huff_sl_base_offset : int32[max+2] slow-path offsets
                   huff_sl_sym         : uint16[N]    slow-path symbols

    Returns dict with:
        huff_lengths : uint8[C]    code lengths (store this; rebuild codes from it)
        huff_stream  : uint8[N_B]  compressed bitstream
        huff_n       : int64[1]    number of encoded symbols
    """
    idx = np.asarray(indices, dtype=np.uint16).ravel()
    n = len(idx)
    C = int(idx.max()) + 1 if n > 0 else 1

    freq = np.bincount(idx, minlength=C).astype(np.int64)
    lengths = _build_code_lengths(freq)
    codes = _canonical_codes(lengths)

    # ── Fast vectorized encoder ──────────────────────────────────────────────
    # Outer loop is over max_len (~20 iterations), NOT over N symbols.
    # Each iteration does numpy scatter on arrays of size N → O(max_len × N)
    # total numpy ops, no Python loop over N.
    #
    # Build per-symbol lookup arrays (C-indexed, gathered via numpy).
    C_used = len(lengths)
    sym_code = np.zeros(C_used, dtype=np.uint32)
    sym_len  = np.zeros(C_used, dtype=np.uint8)
    for sym, (code, L) in codes.items():
        sym_code[sym] = code
        sym_len[sym]  = L

    codes_arr = sym_code[idx]           # uint32[N]
    lens_arr  = sym_len[idx].astype(np.int32)  # int32[N]

    # Cumulative starting bit position for each symbol.
    bit_start = np.zeros(n, dtype=np.int64)
    bit_start[1:] = np.cumsum(lens_arr[:-1])
    total_bits = int(bit_start[-1]) + int(lens_arr[-1]) if n > 0 else 0
    n_bytes = (total_bits + 7) // 8

    # Output bit array: bit_arr[i] = the i-th bit in the final stream (MSB-first).
    bit_arr = np.zeros(total_bits, dtype=np.uint8)

    # For each bit position b within a code (b=0 is the MSB of the code):
    # Find all symbols where b < len, compute their output bit position,
    # extract that bit from their code, and scatter into bit_arr.
    max_len = int(lens_arr.max()) if n > 0 else 0
    for b in range(max_len):
        active = (b < lens_arr)          # bool mask: symbols that have a bit at position b
        if not active.any():
            continue
        out_pos  = bit_start[active] + b                           # int64 output positions
        # Bit b of code (MSB first): bit_val = (code >> (L-1-b)) & 1
        shifts   = (lens_arr[active] - 1 - b).astype(np.int32)
        bit_vals = ((codes_arr[active] >> shifts) & 1).astype(np.uint8)
        # Positions are unique (symbols are non-overlapping) → safe direct assignment.
        bit_arr[out_pos] = bit_vals

    # Pack bit array MSB-first into bytes, then trim to exact byte count.
    # Append 4 zero-pad bytes required by the GPU kernel's 4-byte read window
    # (huff_read_bits reads stream[byte_pos..byte_pos+3] for any bit position).
    pad = (8 - total_bits % 8) % 8
    if pad:
        bit_arr = np.concatenate([bit_arr, np.zeros(pad, dtype=np.uint8)])
    packed_bytes = np.concatenate([np.packbits(bit_arr)[:n_bytes],
                                   np.zeros(4, dtype=np.uint8)])

    result = {
        'huff_lengths': lengths,
        'huff_stream': packed_bytes,
        'huff_n': np.array([n], dtype=np.int64),
    }

    # ── Phase 2 GPU tables (only when caller provides weight matrix shape) ──
    if shape is not None and len(shape) == 2:
        M_s, K_s = int(shape[0]), int(shape[1])
        if n > 0 and M_s * K_s == n:
            result['huff_row_bit_starts'] = bit_start[::K_s].astype(np.int64)
        else:
            result['huff_row_bit_starts'] = np.zeros(max(M_s, 1), dtype=np.int64)

        lut_sym, lut_len = build_gpu_lut(lengths)
        result['huff_lut_sym'] = lut_sym
        result['huff_lut_len'] = lut_len

        sl_first_code, sl_base_offset, sl_sym = _build_slow_path_tables(lengths)
        result['huff_sl_first_code']  = sl_first_code
        result['huff_sl_base_offset'] = sl_base_offset
        result['huff_sl_sym']         = sl_sym

    return result


# ---------------------------------------------------------------------------
# Decode (load-time, CPU)
# ---------------------------------------------------------------------------

def _build_decode_lut(lengths: np.ndarray, lut_bits: int = 16):
    """
    Build a lookup table for fast Huffman decode.

    lut_sym[key] → symbol (0xFFFF = no match at this length)
    lut_len[key] → code length consumed

    For codes ≤ lut_bits the table gives O(1) decode.
    Codes longer than lut_bits fall back to the bit-by-bit path.
    """
    lut_size = 1 << lut_bits
    lut_sym = np.full(lut_size, 0xFFFF, dtype=np.uint32)
    lut_len = np.zeros(lut_size, dtype=np.uint8)

    codes = _canonical_codes(lengths)
    for sym, (code, L) in codes.items():
        if L <= lut_bits:
            n_ext = lut_bits - L
            base = code << n_ext
            count = 1 << n_ext
            lut_sym[base:base + count] = sym
            lut_len[base:base + count] = L

    return lut_sym, lut_len, codes


def huffman_decode_indices(
    huff_stream: np.ndarray,
    huff_lengths: np.ndarray,
    n: int,
    lut_bits: int = 16,
) -> np.ndarray:
    """
    Decode Huffman bitstream to a uint16 index array.

    Uses a 16-bit LUT for O(1) decode of short codes (covers >99% of symbols
    for typical LLM weight distributions).  Long codes (> lut_bits) fall back
    to a bit-by-bit loop — these are rare in practice.

    Args:
        huff_stream  : uint8 bitstream from huffman_encode_indices()
        huff_lengths : uint8[C] code lengths
        n            : number of symbols to decode
        lut_bits     : LUT width (16 = 64 KB table, good balance of speed/memory)

    Returns:
        uint16 array of decoded indices, length n.
    """
    if n == 0:
        return np.empty(0, dtype=np.uint16)

    lengths = np.asarray(huff_lengths, dtype=np.uint8)
    max_len = int(lengths.max()) if len(lengths) and lengths.max() > 0 else 1

    lut_sym, lut_len, codes_full = _build_decode_lut(lengths, lut_bits)

    # Inverse table for the rare long-code fallback path
    codes_inv: Dict[Tuple[int, int], int] = {(c, L): sym for sym, (c, L) in codes_full.items()}

    stream = np.asarray(huff_stream, dtype=np.uint8)
    out = np.empty(n, dtype=np.uint16)

    buf = 0         # bit accumulator (Python int, arbitrary precision)
    buf_bits = 0    # valid bits in buf (MSB side)
    byte_pos = 0
    lut_mask = (1 << lut_bits) - 1

    for i in range(n):
        # --- Refill buffer to at least lut_bits ---
        while buf_bits < lut_bits and byte_pos < len(stream):
            buf = (buf << 8) | int(stream[byte_pos])
            buf_bits += 8
            byte_pos += 1

        # --- Fast path: LUT lookup ---
        if buf_bits >= lut_bits:
            key = (buf >> (buf_bits - lut_bits)) & lut_mask
            sym = int(lut_sym[key])
            consumed = int(lut_len[key])
            if sym != 0xFFFF:
                out[i] = sym
                buf_bits -= consumed
                buf &= (1 << buf_bits) - 1
                continue

        # --- Slow path: bit-by-bit (codes longer than lut_bits) ---
        code = 0
        found = False
        for bit_i in range(max_len):
            if buf_bits == 0:
                if byte_pos < len(stream):
                    buf = int(stream[byte_pos])
                    buf_bits = 8
                    byte_pos += 1
                else:
                    break
            buf_bits -= 1
            code = (code << 1) | ((buf >> buf_bits) & 1)
            buf &= (1 << buf_bits) - 1
            key = (code, bit_i + 1)
            if key in codes_inv:
                out[i] = codes_inv[key]
                found = True
                break
        if not found:
            out[i] = 0  # corrupt stream guard

    return out


# ---------------------------------------------------------------------------
# GPU LUT builder
# ---------------------------------------------------------------------------

# GPU kernel uses a 12-bit LUT (4096 entries) — small enough to fit in L1/shared
# memory on any modern GPU, covers all codes ≤ 12 bits (>99% in practice).
GPU_LUT_BITS = 12
GPU_LUT_SIZE = 1 << GPU_LUT_BITS


def _build_slow_path_tables(lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build canonical decode tables for codes longer than GPU_LUT_BITS.

    Returns:
        sl_first_code  : int64[max_len+2]  first canonical code per length (-1 = no symbols)
        sl_base_offset : int32[max_len+2]  offset into sl_sym[] for each length; [max_len+1] = total
        sl_sym         : uint16[N]         symbols sorted by (length, canonical code order)
    """
    codes = _canonical_codes(lengths)
    max_len = int(lengths.max()) if len(lengths) and lengths.max() > 0 else GPU_LUT_BITS

    arr_size = max_len + 2  # indices 0..max_len+1
    sl_first_code  = np.full(arr_size, -1, dtype=np.int64)
    sl_base_offset = np.zeros(arr_size, dtype=np.int32)

    # Group long symbols (code length > GPU_LUT_BITS) by length
    by_length: Dict[int, list] = {}
    for sym, (code, L) in codes.items():
        if L > GPU_LUT_BITS:
            by_length.setdefault(L, []).append((code, sym))

    sl_sym_list: list = []
    offset = 0
    for L in range(GPU_LUT_BITS + 1, max_len + 1):
        sl_base_offset[L] = offset
        if L in by_length:
            # sort by code value (= canonical order within this length)
            syms_at_L = sorted(by_length[L])
            sl_first_code[L] = syms_at_L[0][0]
            sl_sym_list.extend(sym for _code, sym in syms_at_L)
            offset += len(syms_at_L)
    sl_base_offset[max_len + 1] = offset  # sentinel: total long-symbol count

    # Ensure at least one element so PyTorch doesn't baulk at empty tensors
    if not sl_sym_list:
        sl_sym_list = [0]
    # int32 to match the GPU LUT (avoids 16-bit shared-memory access issues)
    return sl_first_code, sl_base_offset, np.array(sl_sym_list, dtype=np.int32)


def build_gpu_lut(lengths: np.ndarray) -> tuple:
    """
    Build a fixed-width LUT suitable for uploading to GPU memory.

    Returns:
        lut_sym : int32[GPU_LUT_SIZE]  — symbol for each 12-bit prefix
                  -1 = no valid code ends at exactly this prefix length
        lut_len : uint8[GPU_LUT_SIZE]  — bits consumed (0 if no match)

    int32 is used (instead of uint16) to avoid 16-bit shared-memory access
    issues on some GPU/driver combinations (notably ROCm/HIP).

    Codes longer than GPU_LUT_BITS (rare, <1% of symbols) are indicated by
    lut_sym == -1; the GPU kernel falls back to a slow bit-by-bit path.
    """
    lut_sym = np.full(GPU_LUT_SIZE, -1, dtype=np.int32)
    # int32 for lut_len (not uint8): byte-wide global reads return 0 on some
    # ROCm/HIP JIT setups (gfx906/MI50 with ROCm 6.x), so we use 32-bit.
    lut_len = np.zeros(GPU_LUT_SIZE, dtype=np.int32)

    codes = _canonical_codes(lengths)
    for sym, (code, L) in codes.items():
        if L <= GPU_LUT_BITS:
            n_ext = GPU_LUT_BITS - L
            base  = code << n_ext
            count = 1 << n_ext
            lut_sym[base:base + count] = sym
            lut_len[base:base + count] = L

    return lut_sym, lut_len


# ---------------------------------------------------------------------------
# Convenience: round-trip helpers
# ---------------------------------------------------------------------------

def compression_ratio(indices: np.ndarray, bits: int) -> float:
    """
    Estimate compression ratio of Huffman vs fixed-width packing.
    Returns bytes_huffman / bytes_fixed.
    """
    d = huffman_encode_indices(indices)
    n = len(indices.ravel())
    fixed_bytes = (n * bits + 7) // 8
    huffman_bytes = len(d['huff_stream'])
    return huffman_bytes / fixed_bytes if fixed_bytes else 1.0
