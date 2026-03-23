"""
compressed_matmul_cpu.py

Compiles compressed_matmul.c with gcc at import time and exposes a Python
wrapper.  Falls back to the numpy chunked path if gcc is unavailable.

Usage:
    from compressed_matmul_cpu import compressed_matmul, C_KERNEL_AVAILABLE
"""

import ctypes
import hashlib
import mmap
import os
import subprocess
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Memory-mapped packed index buffer
# ---------------------------------------------------------------------------

class MmappedPackedBuffer:
    """
    Read-only mmap of a flat .idx file (raw uint8 bytes).

    Behaves like a uint8 numpy array but the OS pages data from disk on demand.
    Opening the file does NOT load it into RSS — only the pages actually read
    by the C kernel are faulted in.  This lets packed indices live on disk,
    enabling models larger than available RAM.

    Usage:
        buf = MmappedPackedBuffer(path)
        arr = buf.as_numpy()   # zero-copy view of the mmap
        buf.close()            # release mmap + file handle
    """

    def __init__(self, path):
        path = Path(path)
        self._f = open(path, 'rb')
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._arr = np.frombuffer(self._mm, dtype=np.uint8)

    def as_numpy(self):
        """Return a zero-copy uint8 view of the mmap."""
        return self._arr

    def close(self):
        self._arr = None
        try:
            self._mm.close()
        except BufferError:
            pass  # outstanding numpy views still reference the mmap; OS will clean up when they're GC'd
        self._f.close()

    def __len__(self):
        return len(self._arr)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

_SRC = os.path.join(os.path.dirname(__file__), 'compressed_matmul.c')
_lib = None
C_KERNEL_AVAILABLE = False


def _so_path():
    """Return a stable .so path derived from the source hash so the library
    is automatically recompiled when the C source changes."""
    with open(_SRC, 'rb') as f:
        src_hash = hashlib.md5(f.read()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(__file__), '__pycache__')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'compressed_matmul_{src_hash}.so')


def _build_lib():
    so = _so_path()
    if not os.path.exists(so):
        subprocess.check_call([
            'gcc', '-O3', '-march=native', '-funroll-loops', '-fopenmp',
            '-shared', '-fPIC', '-o', so, _SRC, '-lgomp'
        ], stderr=subprocess.PIPE)
    lib = ctypes.CDLL(so)

    # compressed_matmul_f32
    lib.compressed_matmul_f32.restype = None
    lib.compressed_matmul_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # x
        ctypes.POINTER(ctypes.c_uint8),   # packed
        ctypes.POINTER(ctypes.c_float),   # codebook
        ctypes.POINTER(ctypes.c_float),   # out
        ctypes.c_int,                     # T
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # C (codebook size, for bounds check)
        ctypes.c_int,                     # bits
    ]

    # compressed_matmul_f32_chunk
    lib.compressed_matmul_f32_chunk.restype = None
    lib.compressed_matmul_f32_chunk.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # x
        ctypes.POINTER(ctypes.c_uint8),   # packed
        ctypes.POINTER(ctypes.c_float),   # codebook
        ctypes.POINTER(ctypes.c_float),   # out
        ctypes.c_int,                     # T
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # C (codebook size, for bounds check)
        ctypes.c_int,                     # bits
        ctypes.c_int,                     # r_start
        ctypes.c_int,                     # r_end
    ]

    # raw_matmul_f32
    lib.raw_matmul_f32.restype = None
    lib.raw_matmul_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # x      [T, K]
        ctypes.POINTER(ctypes.c_float),   # weight [M, K]
        ctypes.POINTER(ctypes.c_float),   # out    [T, M]
        ctypes.c_int,                     # T
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
    ]

    # raw_matmul_f32_chunk
    lib.raw_matmul_f32_chunk.restype = None
    lib.raw_matmul_f32_chunk.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # x
        ctypes.POINTER(ctypes.c_float),   # weight
        ctypes.POINTER(ctypes.c_float),   # out
        ctypes.c_int,                     # T
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # r_start
        ctypes.c_int,                     # r_end
    ]

    # raw_embedding_f32
    lib.raw_embedding_f32.restype = None
    lib.raw_embedding_f32.argtypes = [
        ctypes.POINTER(ctypes.c_int),     # token_ids [T]
        ctypes.POINTER(ctypes.c_float),   # weight    [vocab, H]
        ctypes.POINTER(ctypes.c_float),   # out       [T, H]
        ctypes.c_int,                     # T
        ctypes.c_int,                     # H
    ]

    # huffman_matmul_f32_chunk
    lib.huffman_matmul_f32_chunk.restype = None
    lib.huffman_matmul_f32_chunk.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # x              [T, K]
        ctypes.POINTER(ctypes.c_uint8),   # huff_stream    uint8[]
        ctypes.POINTER(ctypes.c_uint16),  # lut_sym        uint16[4096]
        ctypes.POINTER(ctypes.c_uint8),   # lut_len        uint8[4096]
        ctypes.POINTER(ctypes.c_int64),   # sl_first_code  int64[max+2]
        ctypes.POINTER(ctypes.c_int32),   # sl_base_offset int32[max+2]
        ctypes.POINTER(ctypes.c_uint16),  # sl_sym         uint16[N]
        ctypes.POINTER(ctypes.c_int64),   # row_bit_starts int64[M]
        ctypes.POINTER(ctypes.c_float),   # codebook       [C]
        ctypes.POINTER(ctypes.c_float),   # out            [T, M]
        ctypes.c_int,                     # T
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # C
        ctypes.c_int,                     # r_start
        ctypes.c_int,                     # r_end
        ctypes.c_int,                     # sl_max_len
    ]

    # huffman_embedding_f32_rows
    lib.huffman_embedding_f32_rows.restype = None
    lib.huffman_embedding_f32_rows.argtypes = [
        ctypes.POINTER(ctypes.c_int),     # token_ids      int32[T]
        ctypes.POINTER(ctypes.c_uint8),   # huff_stream    uint8[]
        ctypes.POINTER(ctypes.c_uint16),  # lut_sym        uint16[4096]
        ctypes.POINTER(ctypes.c_uint8),   # lut_len        uint8[4096]
        ctypes.POINTER(ctypes.c_int64),   # sl_first_code  int64[max+2]
        ctypes.POINTER(ctypes.c_int32),   # sl_base_offset int32[max+2]
        ctypes.POINTER(ctypes.c_uint16),  # sl_sym         uint16[N]
        ctypes.POINTER(ctypes.c_int64),   # row_bit_starts int64[vocab]
        ctypes.POINTER(ctypes.c_float),   # codebook       [C]
        ctypes.POINTER(ctypes.c_float),   # out            [T, H]
        ctypes.c_int,                     # T
        ctypes.c_int,                     # H
        ctypes.c_int,                     # C
        ctypes.c_int,                     # sl_max_len
    ]

    # huffman_decode_to_weights_f32
    lib.huffman_decode_to_weights_f32.restype = None
    lib.huffman_decode_to_weights_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),   # huff_stream    uint8[]
        ctypes.POINTER(ctypes.c_uint16),  # lut_sym        uint16[4096]
        ctypes.POINTER(ctypes.c_uint8),   # lut_len        uint8[4096]
        ctypes.POINTER(ctypes.c_int64),   # sl_first_code  int64[max+2]
        ctypes.POINTER(ctypes.c_int32),   # sl_base_offset int32[max+2]
        ctypes.POINTER(ctypes.c_uint16),  # sl_sym         uint16[N]
        ctypes.POINTER(ctypes.c_int64),   # row_bit_starts int64[M]
        ctypes.POINTER(ctypes.c_float),   # codebook       [C]
        ctypes.POINTER(ctypes.c_float),   # out_weights    [M, K]
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # C
        ctypes.c_int,                     # sl_max_len
    ]

    return lib


def _get_lib():
    global _lib, C_KERNEL_AVAILABLE
    if _lib is not None:
        return _lib
    try:
        _lib = _build_lib()
        C_KERNEL_AVAILABLE = True
        return _lib
    except Exception as e:
        print(f"[compressed_matmul_cpu] gcc build failed, using numpy fallback: {e}")
        _lib = False
        return None


# Attempt build at import time (silent on failure).
_get_lib()


# ---------------------------------------------------------------------------
# Numpy helpers (contiguous float32 arrays)
# ---------------------------------------------------------------------------

def _as_f32(arr):
    """Return a contiguous float32 numpy array (zero-copy if already correct)."""
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.float32 and arr.flags['C_CONTIGUOUS']:
            return arr
        return np.ascontiguousarray(arr, dtype=np.float32)
    import torch
    return np.ascontiguousarray(arr.cpu().numpy().astype(np.float32))


def _as_u8(arr):
    """Return a contiguous uint8 numpy array."""
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.uint8 and arr.flags['C_CONTIGUOUS']:
            return arr
        return np.ascontiguousarray(arr, dtype=np.uint8)
    import torch
    return np.ascontiguousarray(arr.cpu().numpy().view(np.uint8))


def _ptr_f32(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _ptr_u8(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Number of rows processed per C call when chunking large layers.
# Tunable via env var COMPRESSED_CHUNK_ROWS (default 256).
_CHUNK_ROWS = int(os.environ.get('COMPRESSED_CHUNK_ROWS', '256'))


def compressed_matmul(x_np, packed_np, codebook_np, M, K, bits,
                      chunk_rows=None, C=None):
    """
    Compute y = x @ W^T where W is implicitly defined by packed_np + codebook_np.

    No float weight matrix is ever created.  Peak RAM is O(T*M + T*K + codebook)
    regardless of M*K.

    Args:
        x_np       : (T, K) float32 numpy array or torch tensor
        packed_np  : uint8 numpy array, bit-packed indices (with 2 pad bytes at end)
        codebook_np: (C,)  float32 numpy array or torch tensor
        M          : output features
        K          : input features
        bits       : index bit-width
        chunk_rows : rows per C call (default _CHUNK_ROWS); None = all at once

    Returns:
        (T, M) float32 numpy array
    """
    x_f32  = _as_f32(x_np)
    cb_f32 = _as_f32(codebook_np)
    pk_u8  = _as_u8(packed_np)

    # Ensure 2-byte pad at end (safe over-read in the C kernel).
    if len(pk_u8) < (M * K * bits + 7) // 8 + 2:
        pk_u8 = np.concatenate([pk_u8, np.zeros(2, dtype=np.uint8)])

    # Flatten x to (T, K)
    orig_shape = x_f32.shape
    x_f32 = x_f32.reshape(-1, K)
    T = x_f32.shape[0]

    out_np = np.zeros((T, M), dtype=np.float32)

    C_size = C if C is not None else len(cb_f32)

    lib = _get_lib()

    if lib:
        cr = chunk_rows or _CHUNK_ROWS
        r = 0
        while r < M:
            r_end = min(r + cr, M)
            lib.compressed_matmul_f32_chunk(
                _ptr_f32(x_f32), _ptr_u8(pk_u8), _ptr_f32(cb_f32),
                _ptr_f32(out_np),
                T, M, K, C_size, bits, r, r_end
            )
            r = r_end
    else:
        # Numpy fallback: chunked row decompress (avoids full weight matrix)
        from bitpack import unpack_any_bits
        cr = chunk_rows or _CHUNK_ROWS
        for r_start in range(0, M, cr):
            r_end = min(r_start + cr, M)
            n = (r_end - r_start) * K
            start = r_start * K
            # unpack_any_bits returns from element 0..end; slice the window
            idx = unpack_any_bits(pk_u8, bits, start + n)[start:start + n]
            chunk_w = cb_f32[idx].reshape(r_end - r_start, K)
            out_np[:, r_start:r_end] = x_f32 @ chunk_w.T
            del chunk_w, idx

    return out_np.reshape(*orig_shape[:-1], M)


def _ptr_i32(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _ptr_u16(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _ptr_i64(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def raw_matmul(x_np, weight_np, M, K, chunk_rows=None):
    """
    Compute y = x @ weight.T where weight is a plain float32 matrix.

    No codebook, no bit-packing — standard dense linear layer.

    Args:
        x_np      : (T, K) float32 numpy array or torch tensor
        weight_np : (M, K) float32 numpy array or torch tensor (row-major)
        M         : output features
        K         : input features
        chunk_rows: rows per C call; None = all at once

    Returns:
        (T, M) float32 numpy array
    """
    x_f32 = _as_f32(x_np)
    w_f32 = _as_f32(weight_np)

    orig_shape = x_f32.shape
    x_f32 = x_f32.reshape(-1, K)
    T = x_f32.shape[0]

    out_np = np.zeros((T, M), dtype=np.float32)

    lib = _get_lib()
    if lib:
        cr = chunk_rows or _CHUNK_ROWS
        r = 0
        while r < M:
            r_end = min(r + cr, M)
            lib.raw_matmul_f32_chunk(
                _ptr_f32(x_f32), _ptr_f32(w_f32), _ptr_f32(out_np),
                T, M, K, r, r_end
            )
            r = r_end
    else:
        # Numpy fallback
        out_np = (x_f32 @ w_f32.T)

    return out_np.reshape(*orig_shape[:-1], M)


def raw_embedding(token_ids_np, weight_np, H):
    """
    Standard embedding lookup: out[t] = weight[token_ids[t]].

    Args:
        token_ids_np : (T,) int32/int64 numpy array or torch tensor
        weight_np    : (vocab, H) float32 numpy array or torch tensor
        H            : hidden size

    Returns:
        (T, H) float32 numpy array
    """
    if hasattr(token_ids_np, 'cpu'):
        ids = np.ascontiguousarray(token_ids_np.cpu().numpy().astype(np.int32))
    else:
        ids = np.ascontiguousarray(token_ids_np.astype(np.int32))

    w_f32 = _as_f32(weight_np)
    T = len(ids)
    out_np = np.zeros((T, H), dtype=np.float32)

    lib = _get_lib()
    if lib:
        lib.raw_embedding_f32(_ptr_i32(ids), _ptr_f32(w_f32), _ptr_f32(out_np), T, H)
    else:
        out_np = w_f32[ids]

    return out_np


def _coerce_huff_arrays(huff_data):
    """Coerce a huff_data dict to the exact dtypes/contiguity required by the C kernel."""
    stream  = np.ascontiguousarray(huff_data['huff_stream'],    dtype=np.uint8)
    lut_sym = np.ascontiguousarray(huff_data['lut_sym'],        dtype=np.uint16)
    lut_len = np.ascontiguousarray(huff_data['lut_len'],        dtype=np.uint8)
    sl_fc   = np.ascontiguousarray(huff_data['sl_first_code'],  dtype=np.int64)
    sl_bo   = np.ascontiguousarray(huff_data['sl_base_offset'], dtype=np.int32)
    sl_sym  = np.ascontiguousarray(huff_data['sl_sym'],         dtype=np.uint16)
    rbs     = np.ascontiguousarray(huff_data['row_bit_starts'], dtype=np.int64)
    sl_max  = int(huff_data['sl_max_len'])
    # Guarantee 4-byte padding at stream end for safe 4-byte window reads
    stream  = np.concatenate([stream, np.zeros(4, dtype=np.uint8)])
    return stream, lut_sym, lut_len, sl_fc, sl_bo, sl_sym, rbs, sl_max


def huffman_matmul(x_np, huff_data, codebook_np, M, K, chunk_rows=None, C=None):
    """
    Compute y = x @ W^T where W is stored as a Huffman bitstream in RAM.

    Decodes K symbols per output row on-the-fly — no packed index buffer is
    ever created.  In-RAM footprint is the Huffman stream (~60% of packed bits)
    rather than the full fixed-width index array.

    Args:
        x_np       : (T, K) float32 numpy array or torch tensor
        huff_data  : dict with keys huff_stream, lut_sym, lut_len,
                     sl_first_code, sl_base_offset, sl_sym,
                     row_bit_starts, sl_max_len
        codebook_np: (C,) float32
        M, K       : weight matrix shape
        chunk_rows : rows per C call (default _CHUNK_ROWS)
        C          : codebook size (default len(codebook_np))

    Returns:
        (T, M) float32 numpy array
    """
    x_f32  = _as_f32(x_np)
    cb_f32 = _as_f32(codebook_np)

    orig_shape = x_f32.shape
    x_f32 = x_f32.reshape(-1, K)
    T = x_f32.shape[0]

    out_np = np.zeros((T, M), dtype=np.float32)
    C_size = C if C is not None else len(cb_f32)

    stream, lut_sym, lut_len, sl_fc, sl_bo, sl_sym, rbs, sl_max = \
        _coerce_huff_arrays(huff_data)

    lib = _get_lib()
    if not lib:
        raise RuntimeError(
            "huffman_matmul requires the C kernel (gcc). "
            "Huffman inference-time decode is unavailable without gcc."
        )

    cr = chunk_rows or _CHUNK_ROWS
    r = 0
    while r < M:
        r_end = min(r + cr, M)
        lib.huffman_matmul_f32_chunk(
            _ptr_f32(x_f32),
            _ptr_u8(stream),
            _ptr_u16(lut_sym),
            _ptr_u8(lut_len),
            _ptr_i64(sl_fc),
            _ptr_i32(sl_bo),
            _ptr_u16(sl_sym),
            _ptr_i64(rbs),
            _ptr_f32(cb_f32),
            _ptr_f32(out_np),
            T, M, K, C_size, r, r_end, sl_max
        )
        r = r_end

    return out_np.reshape(*orig_shape[:-1], M)


def huffman_embedding(token_ids_np, huff_data, codebook_np, H, C=None):
    """
    Embedding lookup for a Huffman-compressed weight table.

    Decodes only the rows for the requested token IDs — never decodes the full
    vocabulary table.

    Args:
        token_ids_np : (T,) int32/int64 numpy array or torch tensor
        huff_data    : same dict as huffman_matmul; row_bit_starts is indexed
                       by token ID and must cover the full vocabulary
        codebook_np  : (C,) float32
        H            : embedding / hidden dimension
        C            : codebook size (default len(codebook_np))

    Returns:
        (T, H) float32 numpy array
    """
    if hasattr(token_ids_np, 'cpu'):
        ids = np.ascontiguousarray(token_ids_np.cpu().numpy().astype(np.int32))
    else:
        ids = np.ascontiguousarray(np.asarray(token_ids_np, dtype=np.int32))

    cb_f32 = _as_f32(codebook_np)
    T = len(ids)
    out_np = np.zeros((T, H), dtype=np.float32)
    C_size = C if C is not None else len(cb_f32)

    stream, lut_sym, lut_len, sl_fc, sl_bo, sl_sym, rbs, sl_max = \
        _coerce_huff_arrays(huff_data)

    lib = _get_lib()
    if not lib:
        raise RuntimeError(
            "huffman_embedding requires the C kernel (gcc)."
        )

    lib.huffman_embedding_f32_rows(
        _ptr_i32(ids),
        _ptr_u8(stream),
        _ptr_u16(lut_sym),
        _ptr_u8(lut_len),
        _ptr_i64(sl_fc),
        _ptr_i32(sl_bo),
        _ptr_u16(sl_sym),
        _ptr_i64(rbs),
        _ptr_f32(cb_f32),
        _ptr_f32(out_np),
        T, H, C_size, sl_max
    )

    return out_np


def huffman_decode_weights(huff_data, codebook_np, M, K, C=None):
    """
    Decode a Huffman-compressed weight matrix to float32 [M, K].

    Unlike huffman_matmul this does NOT perform the matrix multiply — it only
    decodes the Huffman bitstream and does the codebook lookup, producing the
    full float32 weight matrix.  The caller is responsible for the matmul
    (typically by uploading the result to GPU and calling torch.matmul there).

    Args:
        huff_data   : dict from AdaptiveCodebookLinear._huff_data
        codebook_np : (C,) float32
        M, K        : weight matrix dimensions
        C           : codebook size (default len(codebook_np))

    Returns:
        (M, K) float32 numpy array — freshly allocated, safe to pass to
        torch.from_numpy() and .to(device, non_blocking=True).
    """
    cb_f32 = _as_f32(codebook_np)
    C_size = C if C is not None else len(cb_f32)

    stream, lut_sym, lut_len, sl_fc, sl_bo, sl_sym, rbs, sl_max = \
        _coerce_huff_arrays(huff_data)

    lib = _get_lib()
    if not lib:
        raise RuntimeError(
            "huffman_decode_weights requires the C kernel (gcc)."
        )

    out_weights = np.empty((M, K), dtype=np.float32)

    lib.huffman_decode_to_weights_f32(
        _ptr_u8(stream),
        _ptr_u16(lut_sym),
        _ptr_u8(lut_len),
        _ptr_i64(sl_fc),
        _ptr_i32(sl_bo),
        _ptr_u16(sl_sym),
        _ptr_i64(rbs),
        _ptr_f32(cb_f32),
        _ptr_f32(out_weights),
        M, K, C_size, sl_max
    )

    return out_weights
