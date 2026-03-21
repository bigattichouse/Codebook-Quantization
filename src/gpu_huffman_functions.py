"""
GPU-Accelerated Huffman + Codebook Kernel
==========================================

Separate from gpu_accelerated_functions.py so the existing fixed-width
bit-pack kernel is never touched.

Phase 1 (always available):
  CPU-side Huffman decode at load time → LCM-packed indices → existing GPU kernel.
  This delivers the disk/RAM savings without changing the inference hot path.

Phase 2 (GPU kernel):
  The Huffman-compressed stream lives in VRAM (~18% smaller than LCM-packed).
  Each forward pass:
    1. huffman_decode_to_i32_kernel  — decode stream → int32 index buffer
       (one GPU thread per weight-matrix row, 12-bit LUT in shared memory)
    2. huffman_i32_linear_kernel     — codebook matmul reading int32 directly
       (same grid structure as the existing compressed_linear_kernel)

  Note: int32 is used (not uint16) to avoid 16-bit shared-memory issues on
  ROCm/HIP (AMD GPUs).  The buffer is transient so VRAM impact is minimal.

  VRAM at rest: Huffman stream + codebook (no full index buffer).
  Peak VRAM during forward: adds one int32[M,K] temp buffer per layer
  (freed automatically after the forward call returns).

Key difference from DFloat11's fused two-pass kernel:
  We use the simpler decode-then-matmul approach (two separate kernel launches).
  This is easier to verify correct and avoids the thread-block prefix-sum
  synchronisation that DFloat11 requires.  A single-pass fused version
  (eliminating the temp buffer entirely) can be added as Phase 3 if VRAM
  headroom is critical.

References:
  DFloat11 decode.cu — thread-block two-pass Huffman kernel design
    https://github.com/LeanModels/DFloat11
  DFloat11 paper: https://arxiv.org/abs/2412.19437
  Canonical Huffman codes: RFC 1951, §3.2.2
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

from huffman_codebook import huffman_decode_indices
from bitpack import pack_any_bits

# ---------------------------------------------------------------------------
# Phase 2 CUDA/HIP kernel source
# ---------------------------------------------------------------------------

_HUFFMAN_CUDA_SRC = r"""
// ============================================================================
// GPU Huffman + Codebook Kernel  (Phase 2)
// Compatible with NVCC (CUDA) and hipcc (ROCm/HIP via JIT hipification).
// ============================================================================
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <torch/extension.h>
    #include <stdint.h>
#else
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <stdint.h>
#endif

// ── Constants ────────────────────────────────────────────────────────────────
#define HUFF_LUT_BITS  12
#define HUFF_LUT_SIZE  (1 << HUFF_LUT_BITS)   // 4096
#define HUFF_NO_SYM    (-1)   // sentinel in int32 LUT: no valid code at this prefix
#define DECODE_BLK     64     // threads per block for decode kernel

// NOTE: We use int32_t throughout to avoid sub-32-bit memory access issues
// on some GPU/driver combinations (known to affect ROCm/HIP JIT compilation).
// Specifically: uint8_t and uint16_t global-memory reads have been observed to
// return 0 on gfx906 (MI50/MI60) with ROCm 6.x JIT hipification.
// All reads therefore use 32-bit or 64-bit access with manual byte extraction.

// ── Helper: extract one stream byte from an int32 word array ────────────────
// The stream is packed as little-endian int32 words: stream[0] holds stream
// bytes 0,1,2,3 with byte 0 in the lowest-order bits.
//
// Uses ONLY 32-bit reads AND 32-bit array indices (sub-32-bit reads AND
// 64-bit pointer arithmetic have both been observed to return 0 on gfx906
// with ROCm 6.x JIT hipification).  Max stream size supported by int indices:
// 4 GB, which is far larger than any single weight tensor.
__device__ __forceinline__ uint32_t stream_byte(
    const int32_t* __restrict__ stream_i32,
    int byte_pos)
{
    int word_pos = byte_pos >> 2;             // 32-bit divide — avoids int64 ptr arith
    int byte_off = byte_pos & 3;
    return ((uint32_t)stream_i32[word_pos] >> (byte_off * 8)) & 0xFFu;
}

// ── Helper: read n bits at bit-position bit_pos in an MSB-first bitstream ───
// The Huffman stream is MSB-first: bit 0 of stream = MSB of byte 0.
// stream_i32 holds the raw bytes packed as little-endian int32 words.
//
// Implementation: read 3 consecutive bytes from the stream (covering up to
// 24 bits, which is more than the 12-bit LUT key + 7-bit intra-byte offset =
// 19 bits worst case).  Combine them into a 24-bit big-endian window, then
// shift-and-mask to extract the n bits starting at bit_off from the MSB.
//
// All indices are int (32-bit) to avoid the 64-bit pointer-arithmetic bug
// seen on ROCm 6.x JIT.  bit_pos is passed as int64 by the caller for
// accumulation accuracy but is safely narrowed inside this function
// (max stream size ≪ 2^31 bits = 2 GB, well within signed int32 range).
__device__ __forceinline__ uint32_t huff_read_bits(
    const int32_t* __restrict__ stream_i32,
    int64_t bit_pos_i64,
    int n)
{
    // Narrow to int32 for all arithmetic (safe: streams < 2 GB bits).
    int byte_pos = (int)(bit_pos_i64 >> 3);
    int bit_off  = (int)(bit_pos_i64 & 7);   // bit offset within byte (0 = MSB)

    // Read 3 consecutive stream bytes.
    // For n <= 12 and bit_off <= 7 we need at most ceil((7+12)/8) = 3 bytes.
    uint32_t b0 = stream_byte(stream_i32, byte_pos);
    uint32_t b1 = stream_byte(stream_i32, byte_pos + 1);
    uint32_t b2 = stream_byte(stream_i32, byte_pos + 2);

    // Pack into a 24-bit MSB-first window: b0 is the most-significant byte.
    uint32_t w24 = (b0 << 16) | (b1 << 8) | b2;

    // The n desired bits start at bit_off below the MSB of w24 (bit 23).
    // (24 - bit_off - n) is in [5..24] for n<=12, bit_off<=7 — no UB.
    return (w24 >> (24 - bit_off - n)) & ((1u << n) - 1u);
}

// ── Huffman stream → int32 index buffer ──────────────────────────────────────
// Grid : ceil(M / DECODE_BLK)
// Block: DECODE_BLK threads
// Each thread decodes one row (K symbols) of the [M, K] weight matrix.
// LUT reads go through global memory (L2 cached) — the 16 KB LUT fits in L2
// after the first block's accesses.  Shared memory is intentionally avoided:
// cooperative load of 4096 int32 entries showed unreliable results on
// ROCm/HIP (the root cause is likely a JIT hipification issue with the
// cooperative load pattern; bypassing shared memory gives correct results).
// All parameters use 32-bit or 64-bit types (uint8/uint16 reads return 0 on
// some ROCm/JIT setups; stream bytes are extracted via 32-bit shift+mask).
// NOTE: row_bit_start is int32_t* (not int64_t*) to avoid 64-bit pointer
// arithmetic bugs on ROCm/HIP JIT.  Max stream: 2^31 bits = 256 MB/tensor,
// which covers all practical LLM weight tensors.  The value is widened to
// int64 inside the kernel for accumulation arithmetic.
//
// sl_first_code is also int32_t*: canonical codes at any bit-length fit in
// int32 (max code = 2^24 = 16M < 2^31).
__global__ void huffman_decode_to_i32_kernel(
    const int32_t*  __restrict__ huff_stream,    // [ceil(N_B/4)+1] stream as int32 words
    const int32_t*  __restrict__ row_bit_start,  // [M] bit offset (int32; < 2 GB bits)
    const int32_t*  __restrict__ lut_sym_g,      // [HUFF_LUT_SIZE] int32 LUT (L2 cached)
    const int32_t*  __restrict__ lut_len_g,      // [HUFF_LUT_SIZE] code lengths (int32)
    const int32_t*  __restrict__ sl_first_code,  // [max_len+2] slow-path first codes (int32)
    const int32_t*  __restrict__ sl_base_offset, // [max_len+2] slow-path offsets
    const int32_t*  __restrict__ sl_sym,         // [num_long_syms] slow-path symbols (int32)
    int32_t*        __restrict__ out_indices,    // [M * K] output (int32)
    int M, int K, int max_code_len)
{
    int row = (int)blockIdx.x * DECODE_BLK + (int)threadIdx.x;
    if (row >= M) return;

    // Read int32 bit offset, widen to int64 for accumulation safety.
    int64_t  bit_pos  = (int64_t)row_bit_start[row];
    // Use 32-bit index arithmetic for the output row (avoids int64 ptr arith).
    int32_t* out_row  = out_indices + row * K;

    for (int k = 0; k < K; k++) {

        // ── Fast path: 12-bit LUT (read from global/L2-cached memory) ────
        uint32_t key = huff_read_bits(huff_stream, bit_pos, HUFF_LUT_BITS);
        int32_t  sym = lut_sym_g[key];
        int32_t  len = lut_len_g[key];   // int32 (was uint8, broken on some ROCm/JIT)

        if (sym != HUFF_NO_SYM) {
            out_row[k] = sym;
            bit_pos   += (int64_t)len;
            continue;
        }

        // ── Slow path: extend code bit-by-bit for codes > 12 bits ────────
        // (rare in practice — <1% of symbols for typical LLM distributions)
        uint32_t code    = key;   // already have HUFF_LUT_BITS bits
        int32_t  out_sym = 0;     // default on corrupt stream
        int      out_len = 1;     // default advance (avoid infinite loop)

        for (int L = HUFF_LUT_BITS + 1; L <= max_code_len; L++) {
            uint32_t nb = huff_read_bits(huff_stream, bit_pos + (int64_t)(L - 1), 1);
            code = (code << 1) | nb;

            // sl_first_code is int32: canonical codes fit in 24 bits (<2^24).
            // Sentinel -1 stays representable in int32.
            int32_t fc = sl_first_code[L];
            if (fc >= 0) {
                int32_t delta = (int32_t)code - fc;
                if (delta >= 0) {
                    int32_t base  = sl_base_offset[L];
                    int32_t count = sl_base_offset[L + 1] - base;
                    if (delta < count) {
                        out_sym = sl_sym[base + delta];
                        out_len = L;
                        break;
                    }
                }
            }
        }
        out_row[k] = out_sym;
        bit_pos   += (int64_t)out_len;
    }
}

// ── int32-indexed codebook linear (no bit-unpacking) ─────────────────────────
// Grid(T*M), Block(BLOCK)
// out[tok, ofeat] = Σ_j  x[tok, j] * codebook[ i32_indices[ofeat*K + j] ]
// Identical structure to compressed_linear_kernel but reads int32 directly.
template<int BLOCK>
__global__ void huffman_i32_linear_kernel(
    const float*   __restrict__ x,           // [T, K]  float32
    const int32_t* __restrict__ i32_indices, // [M, K]  int32 (codebook indices)
    const float*   __restrict__ codebook,    // [C]     float32
    float*         __restrict__ out,         // [T, M]  float32
    int T, int M, int K, int C)
{
    int idx   = (int)blockIdx.x;
    int tok   = idx / M;
    int ofeat = idx - tok * M;
    int tid   = (int)threadIdx.x;

    __shared__ float sh[BLOCK];

    const float*   xt = x           + (int64_t)tok   * K;
    const int32_t* wt = i32_indices + (int64_t)ofeat * K;
    float acc = 0.f;

    for (int j = tid; j < K; j += BLOCK) {
        int32_t ci = wt[j];
        if (ci < 0 || ci >= C) ci = C - 1;
        acc += xt[j] * codebook[ci];
    }
    sh[tid] = acc;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[(int64_t)tok * M + ofeat] = sh[0];
}

// ── int32-indexed codebook embedding (no bit-unpacking) ──────────────────────
// Grid(T, ceil(H/BLOCK)), Block(BLOCK)
// out[tok, h] = codebook[ i32_indices[token_id * H + h] ]
template<int BLOCK>
__global__ void huffman_i32_embedding_kernel(
    const int32_t* __restrict__ token_ids,   // [T]
    const int32_t* __restrict__ i32_indices, // [vocab, H]  int32
    const float*   __restrict__ codebook,    // [C]         float32
    float*         __restrict__ out,         // [T, H]      float32
    int T, int H, int C)
{
    int tok = (int)blockIdx.x;
    int h   = (int)blockIdx.y * BLOCK + (int)threadIdx.x;
    if (h >= H) return;

    int32_t token_id = token_ids[tok];
    int32_t ci       = i32_indices[(int64_t)token_id * H + h];
    if (ci < 0 || ci >= C) ci = C - 1;
    out[(int64_t)tok * H + h] = codebook[ci];
}

// ── Diagnostic: direct stream read test ──────────────────────────────────────
// One thread reads the first N int32 words from the stream and writes to out.
// Used to verify that the stream pointer is valid and readable from GPU code.
__global__ void diag_stream_read_kernel(
    const int32_t* stream,
    int32_t* out,
    int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            out[i] = stream[i];
        }
    }
}

// ── Python-facing C++ entry points ───────────────────────────────────────────

// NOTE: row_bit_start and sl_first_code are now int32 (not int64) tensors.
// This avoids 64-bit pointer arithmetic bugs in ROCm/HIP JIT hipification.
// The Python side converts these arrays to int32 before uploading.
torch::Tensor huffman_decode_to_i32(
    torch::Tensor huff_stream,    // [ceil(N_B/4)+1] int32 (stream bytes packed as words)
    torch::Tensor row_bit_start,  // [M]       int32 (bit offsets < 2^31)
    torch::Tensor lut_sym,        // [LUT_SIZE] int32
    torch::Tensor lut_len,        // [LUT_SIZE] int32  (was uint8; byte reads broken on ROCm/JIT)
    torch::Tensor sl_first_code,  // [max+2]   int32 (canonical codes fit in 24 bits)
    torch::Tensor sl_base_offset, // [max+2]   int32
    torch::Tensor sl_sym,         // [N]       int32
    int64_t M, int64_t K, int64_t max_code_len)
{
    huff_stream    = huff_stream.contiguous();
    row_bit_start  = row_bit_start.contiguous();
    lut_sym        = lut_sym.contiguous();
    lut_len        = lut_len.contiguous();
    sl_first_code  = sl_first_code.contiguous();
    sl_base_offset = sl_base_offset.contiguous();
    sl_sym         = sl_sym.contiguous();

    // Allocate int32 output tensor
    auto out = torch::empty(
        {M * K},
        torch::TensorOptions().dtype(torch::kInt32).device(huff_stream.device()));

    const int BLOCK = DECODE_BLK;
    dim3 grid((unsigned)((M + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    huffman_decode_to_i32_kernel<<<grid, blk>>>(
        huff_stream.data_ptr<int32_t>(),
        row_bit_start.data_ptr<int32_t>(),   // int32 (not int64)
        lut_sym.data_ptr<int32_t>(),
        lut_len.data_ptr<int32_t>(),
        sl_first_code.data_ptr<int32_t>(),   // int32 (not int64)
        sl_base_offset.data_ptr<int32_t>(),
        sl_sym.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(),
        (int)M, (int)K, (int)max_code_len);

    return out;  // int32 tensor of decoded codebook indices
}

torch::Tensor huffman_i32_linear(
    torch::Tensor x,            // [*, K]      any float
    torch::Tensor i32_indices,  // [M * K]     int32
    torch::Tensor codebook,     // [C]         float32
    int64_t M, int64_t K)
{
    auto orig = x.sizes().vec();
    int T = (int)(x.numel() / K);
    int C = (int)codebook.size(0);

    x           = x.reshape({T, K}).to(torch::kFloat32).contiguous();
    i32_indices = i32_indices.reshape({M, K}).contiguous();
    codebook    = codebook.to(torch::kFloat32).contiguous();

    auto out = torch::zeros(
        {T, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    const int BLOCK = 256;
    dim3 grid((unsigned)(T * M));
    dim3 blk(BLOCK);

    huffman_i32_linear_kernel<BLOCK><<<grid, blk>>>(
        x.data_ptr<float>(),
        i32_indices.data_ptr<int32_t>(),
        codebook.data_ptr<float>(),
        out.data_ptr<float>(),
        T, (int)M, (int)K, C);

    std::vector<int64_t> out_sz(orig.begin(), orig.end() - 1);
    out_sz.push_back(M);
    return out.reshape(out_sz);
}

torch::Tensor huffman_i32_embedding(
    torch::Tensor token_ids,    // [*]       int64 or int32
    torch::Tensor i32_indices,  // [vocab*H] int32
    torch::Tensor codebook,     // [C]       float32
    int64_t vocab, int64_t hidden)
{
    auto orig = token_ids.sizes().vec();
    int T = (int)token_ids.numel();
    int H = (int)hidden;
    int C = (int)codebook.size(0);

    auto ids = token_ids.reshape({T}).to(torch::kInt32).contiguous();
    i32_indices = i32_indices.reshape({vocab, H}).contiguous();
    codebook    = codebook.to(torch::kFloat32).contiguous();

    auto out = torch::zeros(
        {T, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(codebook.device()));

    const int BLOCK = 128;
    dim3 grid((unsigned)T, (unsigned)((H + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    huffman_i32_embedding_kernel<BLOCK><<<grid, blk>>>(
        ids.data_ptr<int32_t>(),
        i32_indices.data_ptr<int32_t>(),
        codebook.data_ptr<float>(),
        out.data_ptr<float>(),
        T, H, C);

    std::vector<int64_t> out_sz(orig.begin(), orig.end());
    out_sz.push_back(H);
    return out.reshape(out_sz);
}

// ── Diagnostic: copy first n int32 words from stream to output ───────────────
torch::Tensor diag_stream_read(torch::Tensor stream, int64_t n) {
    stream = stream.contiguous();
    auto out = torch::zeros(
        {n}, torch::TensorOptions().dtype(torch::kInt32).device(stream.device()));
    dim3 grid(1), blk(1);
    diag_stream_read_kernel<<<grid, blk>>>(
        stream.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(),
        (int)n);
    return out;
}
"""

_HUFFMAN_CPP_SRC = r"""
torch::Tensor diag_stream_read(torch::Tensor stream, int64_t n);

torch::Tensor huffman_decode_to_i32(
    torch::Tensor huff_stream,     // int32
    torch::Tensor row_bit_start,   // int32 (was int64)
    torch::Tensor lut_sym,         // int32
    torch::Tensor lut_len,         // int32
    torch::Tensor sl_first_code,   // int32 (was int64)
    torch::Tensor sl_base_offset,  // int32
    torch::Tensor sl_sym,          // int32
    int64_t M, int64_t K, int64_t max_code_len);

torch::Tensor huffman_i32_linear(
    torch::Tensor x,
    torch::Tensor i32_indices,
    torch::Tensor codebook,
    int64_t M, int64_t K);

torch::Tensor huffman_i32_embedding(
    torch::Tensor token_ids,
    torch::Tensor i32_indices,
    torch::Tensor codebook,
    int64_t vocab, int64_t hidden);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diag_stream_read",      &diag_stream_read,
          "Diagnostic: copy first N int32 words from stream to output");
    m.def("huffman_decode_to_i32", &huffman_decode_to_i32,
          "GPU Huffman decode: stream → int32 index buffer");
    m.def("huffman_i32_linear",    &huffman_i32_linear,
          "Codebook linear using int32 indices (no bit-unpacking)");
    m.def("huffman_i32_embedding", &huffman_i32_embedding,
          "Codebook embedding using int32 indices (no bit-unpacking)");
}
"""

# ---------------------------------------------------------------------------
# ROCm ctypes wrapper (bypasses broken JIT hipify pipeline)
# ---------------------------------------------------------------------------

import ctypes as _ctypes


class _ROCmHuffmanWrapper:
    """
    ctypes wrapper for libhuffman_kernel.so (standalone HIP binary).

    PyTorch's JIT hipify pipeline (load_inline) produces incorrect kernel
    behaviour on gfx906 / ROCm 6-7: all global-memory reads from kernel
    code return 0.  Compiling with hipcc directly and loading via ctypes
    avoids the hipification step entirely.
    """

    def __init__(self, so_path: str):
        lib = _ctypes.CDLL(so_path)

        # int huff_decode_to_i32(stream, row_bit_start, lut_sym, lut_len,
        #   sl_first_code, sl_base_offset, sl_sym, out, M, K, max_code_len,
        #   hipStream_t)
        lib.huff_decode_to_i32.restype = _ctypes.c_int
        lib.huff_decode_to_i32.argtypes = [
            _ctypes.c_void_p,  # stream         uint8_t*
            _ctypes.c_void_p,  # row_bit_start  int64_t*
            _ctypes.c_void_p,  # lut_sym        int32_t*
            _ctypes.c_void_p,  # lut_len        int32_t*
            _ctypes.c_void_p,  # sl_first_code  int32_t*
            _ctypes.c_void_p,  # sl_base_offset int32_t*
            _ctypes.c_void_p,  # sl_sym         int32_t*
            _ctypes.c_void_p,  # out            int32_t*
            _ctypes.c_int,     # M
            _ctypes.c_int,     # K
            _ctypes.c_int,     # max_code_len
            _ctypes.c_void_p,  # hipStream_t (NULL = default)
        ]

        # int huff_i32_linear_f32(x, i32_indices, codebook, out,
        #   T, M, K, C, hipStream_t)
        lib.huff_i32_linear_f32.restype = _ctypes.c_int
        lib.huff_i32_linear_f32.argtypes = [
            _ctypes.c_void_p,  # x           float*
            _ctypes.c_void_p,  # i32_indices int32_t*
            _ctypes.c_void_p,  # codebook    float*
            _ctypes.c_void_p,  # out         float*
            _ctypes.c_int,     # T
            _ctypes.c_int,     # M
            _ctypes.c_int,     # K
            _ctypes.c_int,     # C
            _ctypes.c_void_p,  # hipStream_t
        ]

        # int huff_i32_embedding_f32(token_ids, i32_indices, codebook, out,
        #   T, H, C, hipStream_t)
        lib.huff_i32_embedding_f32.restype = _ctypes.c_int
        lib.huff_i32_embedding_f32.argtypes = [
            _ctypes.c_void_p,  # token_ids   int32_t*
            _ctypes.c_void_p,  # i32_indices int32_t*
            _ctypes.c_void_p,  # codebook    float*
            _ctypes.c_void_p,  # out         float*
            _ctypes.c_int,     # T
            _ctypes.c_int,     # H
            _ctypes.c_int,     # C
            _ctypes.c_void_p,  # hipStream_t
        ]

        self._lib = lib

    def _stream_ptr(self):
        try:
            return _ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        except Exception:
            return None

    def huffman_decode_to_i32(self, stream, row_bit_start, lut_sym, lut_len,
                               sl_first_code, sl_base_offset, sl_sym,
                               M, K, max_code_len):
        """
        Inputs: torch.Tensor on GPU
          stream        : uint8 (raw bytes)  — note: uint8, NOT int32!
          row_bit_start : int64
          lut_sym       : int32
          lut_len       : int32
          sl_first_code : int32
          sl_base_offset: int32
          sl_sym        : int32
        Returns: int32 tensor [M*K] on GPU
        """
        stream        = stream.contiguous()
        row_bit_start = row_bit_start.contiguous()
        lut_sym       = lut_sym.contiguous()
        lut_len       = lut_len.contiguous()
        sl_first_code = sl_first_code.contiguous()
        sl_base_offset= sl_base_offset.contiguous()
        sl_sym        = sl_sym.contiguous()

        out = torch.empty(int(M) * int(K), dtype=torch.int32,
                          device=stream.device)
        rc = self._lib.huff_decode_to_i32(
            _ctypes.c_void_p(stream.data_ptr()),
            _ctypes.c_void_p(row_bit_start.data_ptr()),
            _ctypes.c_void_p(lut_sym.data_ptr()),
            _ctypes.c_void_p(lut_len.data_ptr()),
            _ctypes.c_void_p(sl_first_code.data_ptr()),
            _ctypes.c_void_p(sl_base_offset.data_ptr()),
            _ctypes.c_void_p(sl_sym.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            int(M), int(K), int(max_code_len),
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"huff_decode_to_i32 returned error {rc}")
        return out

    def huffman_i32_linear(self, x, i32_indices, codebook, M, K):
        orig = x.shape
        T = int(x.numel() // K)
        C = int(codebook.size(0))

        x_f32       = x.reshape(T, int(K)).to(torch.float32).contiguous()
        i32_indices = i32_indices.reshape(int(M), int(K)).contiguous()
        cb_f32      = codebook.to(torch.float32).contiguous()
        out         = torch.zeros(T, int(M), dtype=torch.float32,
                                  device=x_f32.device)

        rc = self._lib.huff_i32_linear_f32(
            _ctypes.c_void_p(x_f32.data_ptr()),
            _ctypes.c_void_p(i32_indices.data_ptr()),
            _ctypes.c_void_p(cb_f32.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            T, int(M), int(K), C,
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"huff_i32_linear_f32 returned error {rc}")

        out_shape = list(orig[:-1]) + [int(M)]
        return out.reshape(out_shape)

    def huffman_i32_embedding(self, token_ids, i32_indices, codebook,
                               vocab, hidden):
        orig = token_ids.shape
        T = int(token_ids.numel())
        H = int(hidden)
        C = int(codebook.size(0))

        ids    = token_ids.reshape(T).to(torch.int32).contiguous()
        i32_idx= i32_indices.reshape(int(vocab), H).contiguous()
        cb_f32 = codebook.to(torch.float32).contiguous()
        out    = torch.zeros(T, H, dtype=torch.float32, device=cb_f32.device)

        rc = self._lib.huff_i32_embedding_f32(
            _ctypes.c_void_p(ids.data_ptr()),
            _ctypes.c_void_p(i32_idx.data_ptr()),
            _ctypes.c_void_p(cb_f32.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            T, H, C,
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"huff_i32_embedding_f32 returned error {rc}")

        out_shape = list(orig) + [H]
        return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# Extension loader
# ---------------------------------------------------------------------------

_huff_ext = None


def _load_huffman_extension():
    """
    Load the GPU Huffman extension (first call only; cached thereafter).

    On ROCm: loads libhuffman_kernel.so via ctypes (bypasses the broken
    PyTorch JIT hipify pipeline that causes all global-memory reads to
    return 0 on gfx906/ROCm 6-7).

    On NVIDIA CUDA: uses PyTorch load_inline (works correctly on NVCC).
    """
    global _huff_ext
    if _huff_ext is not None:
        return _huff_ext
    if not torch.cuda.is_available():
        return None

    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

    if is_rocm:
        # ── ROCm path: load pre-compiled libhuffman_kernel.so via ctypes ──
        # The JIT hipify path (load_inline) is known to produce incorrect
        # kernel behaviour on this system (all reads return 0).
        _this_dir = Path(__file__).resolve().parent
        _rocm_dir = _this_dir.parent / 'rocm'
        so_path   = _rocm_dir / 'libhuffman_kernel.so'
        if not so_path.exists():
            print(
                f"  [huffman_kernel] {so_path} not found.\n"
                f"  Run: cd {_rocm_dir} && make huffman "
                f"HIP_LIB_DIR=<path/to/torch/lib> HIP_COV=5\n"
                "  (HIP_COV=5 required when using ROCm 6.0 bundled with PyTorch)\n"
                "  Falling back to Phase 1 CPU decode.",
                flush=True,
            )
            return None
        try:
            _huff_ext = _ROCmHuffmanWrapper(str(so_path))
            print(f"  [huffman_kernel] Loaded {so_path.name} (ROCm ctypes path).",
                  flush=True)
        except Exception as e:
            print(f"  [huffman_kernel] Failed to load {so_path}: {e}\n"
                  "  Falling back to Phase 1 CPU decode.", flush=True)
            _huff_ext = None
        return _huff_ext

    # ── NVIDIA CUDA path: JIT compile via load_inline ─────────────────────
    try:
        from torch.utils.cpp_extension import load_inline
        print("  [huffman_kernel] Compiling GPU Huffman kernel (first run ~30-60s)...",
              flush=True)
        _huff_ext = load_inline(
            name="huffman_codebook_v7",
            cpp_sources=[_HUFFMAN_CPP_SRC],
            cuda_sources=[_HUFFMAN_CUDA_SRC],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        print("  [huffman_kernel] GPU Huffman kernel ready.", flush=True)
    except Exception as e:
        print(f"  [huffman_kernel] Compile failed ({e}); using Phase 1 CPU decode.")
        _huff_ext = None

    return _huff_ext


def _has_phase2_data(data: dict) -> bool:
    """True if all Phase 2 GPU tables were stored in this NPZ."""
    return all(k in data for k in (
        'huff_row_bit_starts', 'huff_lut_sym', 'huff_lut_len',
        'huff_sl_first_code', 'huff_sl_base_offset', 'huff_sl_sym',
    ))


def _np_to_i32_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """Upload a numpy array to GPU as int32."""
    return torch.from_numpy(np.asarray(arr, dtype=np.int32)).to(device)


# ---------------------------------------------------------------------------
# Phase 1 & 2 wrapper classes
# ---------------------------------------------------------------------------

class HuffmanCodebookLinear:
    """
    Linear layer using a Huffman-encoded index stream.

    Phase 1 (CPU decode at load time):
      Decodes Huffman → LCM-packed → uploads to GPU → reuses existing kernel.
      VRAM: same as standard LCM-packed (no VRAM savings; disk/RAM savings only).

    Phase 2 (GPU decode per forward pass):
      Uploads the Huffman stream to GPU VRAM (~18% smaller than LCM-packed).
      Each forward call decodes to a transient uint16 buffer, then does the matmul.
      VRAM at rest: ~18% smaller than Phase 1.

    Phase 2 is used when:
      (a) CUDA is available,
      (b) the Phase 2 GPU tables were stored at compression time, and
      (c) the GPU Huffman kernel compiles successfully.
    """

    def __init__(
        self,
        name: str,
        huff_stream: np.ndarray,
        huff_lengths: np.ndarray,
        huff_n: int,
        codebook: torch.Tensor,
        shape: Tuple[int, int],
        bits: int,
        # Phase 2 data — present when compressed with --entropy-code and shape provided
        huff_row_bit_starts: Optional[np.ndarray] = None,
        huff_lut_sym:        Optional[np.ndarray] = None,
        huff_lut_len:        Optional[np.ndarray] = None,
        huff_sl_first_code:  Optional[np.ndarray] = None,
        huff_sl_base_offset: Optional[np.ndarray] = None,
        huff_sl_sym:         Optional[np.ndarray] = None,
    ):
        self.name  = name
        self.shape = tuple(int(s) for s in shape)
        self.bits  = bits
        M, K = self.shape

        phase2_data = all(x is not None for x in (
            huff_row_bit_starts, huff_lut_sym, huff_lut_len,
            huff_sl_first_code, huff_sl_base_offset, huff_sl_sym,
        ))

        if torch.cuda.is_available() and phase2_data:
            ext = _load_huffman_extension()
        else:
            ext = None

        if ext is not None:
            # ── Phase 2: store Huffman stream on GPU; decode per forward pass ──
            device = codebook.device if codebook.is_cuda else torch.device('cuda')

            is_rocm_ext = isinstance(ext, _ROCmHuffmanWrapper)
            if is_rocm_ext:
                # ROCm ctypes path: native HIP kernel reads raw uint8 bytes
                # and int64_t row_bit_starts.  No packing needed.
                stream_np  = np.asarray(huff_stream, dtype=np.uint8)
                rbs_dtype  = np.int64
            else:
                # NVIDIA load_inline path: kernel reads stream as int32 words
                # (with shift/mask byte extraction) and int32 row_bit_starts.
                raw_len   = len(huff_stream)
                pad_len   = ((raw_len + 11) // 4) * 4
                padded_u8 = np.zeros(pad_len, dtype=np.uint8)
                padded_u8[:raw_len] = huff_stream
                stream_np  = padded_u8.view(np.int32)
                rbs_dtype  = np.int32

            self._phase           = 2
            self._ext             = ext
            self._huff_stream     = torch.from_numpy(stream_np.copy()).to(device)
            self._row_bit_start   = torch.from_numpy(
                                        np.asarray(huff_row_bit_starts, dtype=rbs_dtype)
                                    ).to(device)
            self._lut_sym         = _np_to_i32_tensor(huff_lut_sym, device)
            self._lut_len         = _np_to_i32_tensor(huff_lut_len, device)
            self._sl_first_code   = _np_to_i32_tensor(huff_sl_first_code, device)
            self._sl_base_offset  = _np_to_i32_tensor(huff_sl_base_offset, device)
            self._sl_sym          = _np_to_i32_tensor(huff_sl_sym, device)
            self._max_code_len    = int(huff_lengths.max()) if huff_lengths.max() > 0 else 1
            self._codebook        = codebook.to(device=device, dtype=torch.float32)
            self._gpu             = None

            n_stream = len(huff_stream)
            n_lcm    = (M * K * bits + 7) // 8
            pct      = (1.0 - n_stream / n_lcm) * 100 if n_lcm else 0.0
            print(
                f"  [HuffmanLinear P2] {name}: "
                f"stream={n_stream/1e3:.1f}KB  LCM={n_lcm/1e3:.1f}KB  "
                f"VRAM saved={pct:.1f}%",
                flush=True,
            )

        elif torch.cuda.is_available():
            # ── Phase 1: CPU decode → LCM-packed → existing GPU kernel ──
            self._phase = 1
            self._ext   = None
            print(
                f"  [HuffmanLinear P1] decoding {name}: "
                f"{huff_n:,} symbols → {bits}-bit",
                flush=True,
            )
            raw_indices = huffman_decode_indices(huff_stream, huff_lengths, huff_n)
            packed = pack_any_bits(raw_indices, bits)
            from gpu_accelerated_functions import GPUAcceleratedLinear
            self._gpu = GPUAcceleratedLinear(name, packed, codebook, shape, bits)

        else:
            # ── CPU fallback ──
            self._phase = 0
            self._ext   = None
            self._gpu   = None
            raw_indices = huffman_decode_indices(huff_stream, huff_lengths, huff_n)
            self._packed_cpu   = pack_any_bits(raw_indices, bits)
            self._codebook_cpu = codebook

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        if self._phase == 2:
            M, K = self.shape
            device = self._codebook.device
            # Decode Huffman stream → transient int32 index buffer
            i32_buf = self._ext.huffman_decode_to_i32(
                self._huff_stream,
                self._row_bit_start,
                self._lut_sym,
                self._lut_len,
                self._sl_first_code,
                self._sl_base_offset,
                self._sl_sym,
                M, K, self._max_code_len,
            )
            # Codebook matmul using int32 indices
            out = self._ext.huffman_i32_linear(
                x.to(device), i32_buf, self._codebook, M, K
            )
            return out.to(dtype=in_dtype)

        if self._phase == 1:
            return self._gpu(x)

        # CPU fallback
        M, K = self.shape
        from compressed_matmul_cpu import compressed_matmul as _c_matmul
        x_np  = x.cpu().reshape(-1, K).float().numpy()
        cb_np = self._codebook_cpu.cpu().float().numpy()
        out_np = _c_matmul(x_np, self._packed_cpu, cb_np, M, K, self.bits, C=len(cb_np))
        return torch.from_numpy(out_np).reshape(*x.shape[:-1], M).to(x.device)


class HuffmanCodebookEmbedding:
    """
    Embedding layer using a Huffman-encoded index stream.

    Phase 1: CPU decode at load time → existing GPU embedding kernel.
    Phase 2: GPU decode per forward pass (stream in VRAM, decode → uint16 → lookup).
    """

    def __init__(
        self,
        name: str,
        huff_stream: np.ndarray,
        huff_lengths: np.ndarray,
        huff_n: int,
        codebook: torch.Tensor,
        shape: Tuple[int, int],
        bits: int,
        huff_row_bit_starts: Optional[np.ndarray] = None,
        huff_lut_sym:        Optional[np.ndarray] = None,
        huff_lut_len:        Optional[np.ndarray] = None,
        huff_sl_first_code:  Optional[np.ndarray] = None,
        huff_sl_base_offset: Optional[np.ndarray] = None,
        huff_sl_sym:         Optional[np.ndarray] = None,
    ):
        self.name  = name
        self.shape = tuple(int(s) for s in shape)
        self.bits  = bits
        vocab, hidden = self.shape

        phase2_data = all(x is not None for x in (
            huff_row_bit_starts, huff_lut_sym, huff_lut_len,
            huff_sl_first_code, huff_sl_base_offset, huff_sl_sym,
        ))

        if torch.cuda.is_available() and phase2_data:
            ext = _load_huffman_extension()
        else:
            ext = None

        if ext is not None:
            device = codebook.device if codebook.is_cuda else torch.device('cuda')

            is_rocm_ext = isinstance(ext, _ROCmHuffmanWrapper)
            if is_rocm_ext:
                # ROCm ctypes path: native HIP kernel reads raw uint8 bytes
                # and int64_t row_bit_starts.  No packing needed.
                stream_np = np.asarray(huff_stream, dtype=np.uint8)
                rbs_dtype = np.int64
            else:
                # NVIDIA load_inline path: kernel reads stream as int32 words
                raw_len   = len(huff_stream)
                pad_len   = ((raw_len + 11) // 4) * 4
                padded_u8 = np.zeros(pad_len, dtype=np.uint8)
                padded_u8[:raw_len] = huff_stream
                stream_np = padded_u8.view(np.int32)
                rbs_dtype = np.int32

            self._phase           = 2
            self._ext             = ext
            self._huff_stream     = torch.from_numpy(stream_np.copy()).to(device)
            self._row_bit_start   = torch.from_numpy(
                                        np.asarray(huff_row_bit_starts, dtype=rbs_dtype)
                                    ).to(device)
            self._lut_sym         = _np_to_i32_tensor(huff_lut_sym, device)
            self._lut_len         = _np_to_i32_tensor(huff_lut_len, device)
            self._sl_first_code   = _np_to_i32_tensor(huff_sl_first_code, device)
            self._sl_base_offset  = _np_to_i32_tensor(huff_sl_base_offset, device)
            self._sl_sym          = _np_to_i32_tensor(huff_sl_sym, device)
            self._max_code_len    = int(huff_lengths.max()) if huff_lengths.max() > 0 else 1
            self._codebook        = codebook.to(device=device, dtype=torch.float32)
            self._gpu             = None

            print(f"  [HuffmanEmbed P2] {name}: {huff_n:,} symbols → {bits}-bit", flush=True)

        elif torch.cuda.is_available():
            self._phase = 1
            self._ext   = None
            print(f"  [HuffmanEmbed P1] decoding {name}: {huff_n:,} symbols → {bits}-bit",
                  flush=True)
            raw_indices = huffman_decode_indices(huff_stream, huff_lengths, huff_n)
            packed = pack_any_bits(raw_indices, bits)
            from gpu_accelerated_functions import GPUAcceleratedEmbedding
            self._gpu = GPUAcceleratedEmbedding(name, packed, codebook, shape, bits)

        else:
            self._phase = 0
            self._ext   = None
            self._gpu   = None
            raw_indices = huffman_decode_indices(huff_stream, huff_lengths, huff_n)
            self._packed_cpu   = pack_any_bits(raw_indices, bits)
            self._codebook_cpu = codebook

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._phase == 2:
            vocab, hidden = self.shape
            device = self._codebook.device
            # Decode: treat embedding as [vocab, hidden] — row per vocab entry
            i32_buf = self._ext.huffman_decode_to_i32(
                self._huff_stream,
                self._row_bit_start,
                self._lut_sym,
                self._lut_len,
                self._sl_first_code,
                self._sl_base_offset,
                self._sl_sym,
                vocab, hidden, self._max_code_len,
            )
            out = self._ext.huffman_i32_embedding(
                x.to(device), i32_buf, self._codebook, vocab, hidden
            )
            return out

        if self._phase == 1:
            return self._gpu(x)

        # CPU fallback via fast index manager
        from fast_index_manager import get_index_manager
        hidden = self.shape[1]
        cb_cpu = self._codebook_cpu.cpu()
        index_manager = get_index_manager('cpu')
        if self.name not in index_manager.lookup_tables:
            index_manager.prepare_lookup_table(
                self.name,
                torch.from_numpy(self._packed_cpu),
                self.bits,
            )
        unique_ids, inverse = torch.unique(x.reshape(-1).cpu(), return_inverse=True)
        rows = []
        for tid in unique_ids.tolist():
            start = int(tid) * hidden
            row_indices = index_manager.fast_index_lookup(self.name, hidden, start)
            rows.append(cb_cpu[row_indices.long()])
        weight = torch.stack(rows, dim=0)
        return weight[inverse].reshape(*x.shape, hidden).to(x.device)
