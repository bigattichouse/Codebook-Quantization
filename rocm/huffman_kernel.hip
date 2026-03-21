/*
 * huffman_kernel.hip
 *
 * Standalone HIP implementation of GPU Huffman decode + codebook matmul/embedding.
 * No PyTorch or other ML framework dependency.
 *
 * Build with (from this directory):
 *   make libhuffman_kernel.so
 * or manually:
 *   hipcc -O3 --offload-arch=<arch> -fPIC -fno-gpu-rdc -shared \
 *         -o libhuffman_kernel.so huffman_kernel.hip
 *
 * Why standalone?
 *   PyTorch's JIT hipify pipeline (load_inline) produces incorrect kernel
 *   behaviour on gfx906 / ROCm 6-7: all global memory reads return 0.
 *   Compiling with hipcc directly avoids the hipification step entirely.
 */

#include "huffman_kernel.h"
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* ── Constants ────────────────────────────────────────────────────────────── */
#define HUFF_LUT_BITS  12
#define HUFF_LUT_SIZE  (1 << HUFF_LUT_BITS)   /* 4096 */
#define HUFF_NO_SYM    (-1)
#define DECODE_BLK     64     /* threads per block for decode kernel */

/* ── Helper: read n bits at bit-position bit_pos in an MSB-first bitstream ─ */
/*
 * The stream is stored as raw uint8 bytes.  In native hipcc (not JIT
 * hipified), uint8_t global reads work correctly.  We read 3 consecutive
 * bytes and combine into a 24-bit window, then shift-and-mask.
 *
 * For n <= 12 and bit_off <= 7, at most 3 bytes are needed (7+12 = 19 bits).
 * bit_pos is int64 for large tensors (> 256 MB of bits).
 */
__device__ __forceinline__ uint32_t huff_read_bits(
    const uint8_t* __restrict__ stream,
    int64_t bit_pos,
    int n)
{
    int64_t byte_pos = bit_pos >> 3;
    int     bit_off  = (int)(bit_pos & 7);

    uint32_t b0 = (uint32_t)stream[byte_pos];
    uint32_t b1 = (uint32_t)stream[byte_pos + 1];
    uint32_t b2 = (uint32_t)stream[byte_pos + 2];

    uint32_t w24 = (b0 << 16) | (b1 << 8) | b2;
    return (w24 >> (24 - bit_off - n)) & ((1u << n) - 1u);
}

/* ── Huffman decode kernel ─────────────────────────────────────────────────── */
/*
 * Grid: ceil(M / DECODE_BLK)
 * Block: DECODE_BLK threads
 * Each thread decodes one row of K symbols.
 */
__global__ void huffman_decode_kernel(
    const uint8_t* __restrict__ stream,       /* raw Huffman bitstream  */
    const int64_t* __restrict__ row_bit_start,/* [M] bit offset per row */
    const int32_t* __restrict__ lut_sym,      /* [4096] symbol LUT      */
    const int32_t* __restrict__ lut_len,      /* [4096] code length LUT */
    const int32_t* __restrict__ sl_first_code,/* [max_len+2] slow path  */
    const int32_t* __restrict__ sl_base_off,  /* [max_len+2]            */
    const int32_t* __restrict__ sl_sym,       /* [N_long] slow symbols  */
    int32_t*       __restrict__ out,          /* [M*K] output indices   */
    int M, int K, int max_code_len)
{
    int row = (int)blockIdx.x * DECODE_BLK + (int)threadIdx.x;
    if (row >= M) return;

    int64_t  bit_pos = row_bit_start[row];
    int32_t* out_row = out + (int64_t)row * K;

    for (int k = 0; k < K; ++k) {
        /* ── Fast path: 12-bit LUT ──────────────────────────────────────── */
        uint32_t key = huff_read_bits(stream, bit_pos, HUFF_LUT_BITS);
        int32_t  sym = lut_sym[key];
        int32_t  len = lut_len[key];

        if (sym != HUFF_NO_SYM) {
            out_row[k] = sym;
            bit_pos   += (int64_t)len;
            continue;
        }

        /* ── Slow path: codes > 12 bits (rare, < 1% for typical LLMs) ─── */
        uint32_t code    = key;
        int32_t  out_sym = 0;
        int      out_len = 1;

        for (int L = HUFF_LUT_BITS + 1; L <= max_code_len; ++L) {
            uint32_t nb = huff_read_bits(stream, bit_pos + (int64_t)(L - 1), 1);
            code = (code << 1) | nb;

            int32_t fc = sl_first_code[L];
            if (fc >= 0) {
                int32_t delta = (int32_t)code - fc;
                if (delta >= 0) {
                    int32_t base  = sl_base_off[L];
                    int32_t count = sl_base_off[L + 1] - base;
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

/* ── Codebook linear with int32 indices ───────────────────────────────────── */
/*
 * Grid: T*M blocks, Block: 256 threads
 * out[tok, ofeat] = Σ_k  x[tok,k] * codebook[ i32[ofeat*K + k] ]
 */
#define LIN_BLK 256
__global__ void huff_i32_linear_kernel(
    const float*   __restrict__ x,        /* [T, K]  float32 */
    const int32_t* __restrict__ i32_idx,  /* [M, K]  int32   */
    const float*   __restrict__ codebook, /* [C]     float32 */
    float*         __restrict__ out,      /* [T, M]  float32 */
    int T, int M, int K, int C)
{
    int idx   = (int)blockIdx.x;
    int tok   = idx / M;
    int ofeat = idx - tok * M;
    int tid   = (int)threadIdx.x;

    __shared__ float sh[LIN_BLK];

    const float*   xt = x       + (int64_t)tok   * K;
    const int32_t* wt = i32_idx + (int64_t)ofeat * K;
    float acc = 0.f;

    for (int j = tid; j < K; j += LIN_BLK) {
        int32_t ci = wt[j];
        if (ci < 0 || ci >= C) ci = C - 1;
        acc += xt[j] * codebook[ci];
    }
    sh[tid] = acc;
    __syncthreads();

    for (int s = LIN_BLK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[(int64_t)tok * M + ofeat] = sh[0];
}

/* ── Codebook embedding with int32 indices ───────────────────────────────── */
/*
 * Grid(T, ceil(H/128)), Block(128)
 * out[tok, h] = codebook[ i32_idx[token_id * H + h] ]
 */
#define EMB_BLK 128
__global__ void huff_i32_embedding_kernel(
    const int32_t* __restrict__ token_ids, /* [T]        int32  */
    const int32_t* __restrict__ i32_idx,   /* [vocab, H] int32  */
    const float*   __restrict__ codebook,  /* [C]        float32 */
    float*         __restrict__ out,       /* [T, H]     float32 */
    int T, int H, int C)
{
    int tok = (int)blockIdx.x;
    int h   = (int)blockIdx.y * EMB_BLK + (int)threadIdx.x;
    if (h >= H) return;

    int32_t token_id = token_ids[tok];
    int32_t ci       = i32_idx[(int64_t)token_id * H + h];
    if (ci < 0 || ci >= C) ci = C - 1;
    out[(int64_t)tok * H + h] = codebook[ci];
}

/* ============================================================================
 * C entry points (no name mangling)
 * ============================================================================ */

extern "C" {

/* --------------------------------------------------------------------------
 * huff_decode_to_i32
 *
 * Decode a per-row Huffman bitstream into a flat int32 index array.
 *
 * stream          : uint8_t* MSB-first Huffman bitstream (4+ extra pad bytes)
 * row_bit_start   : int64_t* [M]        bit offset for start of each row
 * lut_sym         : int32_t* [4096]     12-bit LUT symbols  (-1 = no match)
 * lut_len         : int32_t* [4096]     12-bit LUT lengths
 * sl_first_code   : int32_t* [max+2]    slow-path first canonical codes
 * sl_base_offset  : int32_t* [max+2]    slow-path symbol array offsets
 * sl_sym          : int32_t* [N]        slow-path symbols
 * out             : int32_t* [M*K]      decoded index output
 * M, K            : weight matrix dimensions
 * max_code_len    : maximum code length (for slow-path bound)
 * stream_h        : hipStream_t (NULL = default stream)
 * -------------------------------------------------------------------------- */
int huff_decode_to_i32(
    const void*  stream,
    const void*  row_bit_start,
    const void*  lut_sym,
    const void*  lut_len,
    const void*  sl_first_code,
    const void*  sl_base_offset,
    const void*  sl_sym,
    void*        out,
    int M, int K, int max_code_len,
    hipStream_t  stream_h)
{
    if (!stream || !row_bit_start || !lut_sym || !lut_len ||
        !sl_first_code || !sl_base_offset || !sl_sym || !out)
        return HK_ERR_ARG;
    if (M <= 0 || K <= 0) return HK_ERR_ARG;

    dim3 grid((unsigned)((M + DECODE_BLK - 1) / DECODE_BLK));
    dim3 blk(DECODE_BLK);

    huffman_decode_kernel<<<grid, blk, 0, stream_h>>>(
        (const uint8_t*)stream,
        (const int64_t*)row_bit_start,
        (const int32_t*)lut_sym,
        (const int32_t*)lut_len,
        (const int32_t*)sl_first_code,
        (const int32_t*)sl_base_offset,
        (const int32_t*)sl_sym,
        (int32_t*)out,
        M, K, max_code_len);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[huffman_kernel] decode kernel error: %s\n",
                hipGetErrorString(err));
        return HK_ERR_HIP;
    }
    return HK_SUCCESS;
}

/* --------------------------------------------------------------------------
 * huff_i32_linear_f32
 *
 * Codebook matmul using int32 indices (no bit-unpacking).
 * x          : float* [T, K]
 * i32_indices: int32_t* [M*K]
 * codebook   : float* [C]
 * out        : float* [T, M]  (caller allocates, initialised to 0)
 * -------------------------------------------------------------------------- */
int huff_i32_linear_f32(
    const void*  x,
    const void*  i32_indices,
    const void*  codebook,
    void*        out,
    int T, int M, int K, int C,
    hipStream_t  stream_h)
{
    if (!x || !i32_indices || !codebook || !out) return HK_ERR_ARG;
    if (T <= 0 || M <= 0 || K <= 0) return HK_ERR_ARG;

    dim3 grid((unsigned)(T * M));
    dim3 blk(LIN_BLK);

    huff_i32_linear_kernel<<<grid, blk, 0, stream_h>>>(
        (const float*)x,
        (const int32_t*)i32_indices,
        (const float*)codebook,
        (float*)out,
        T, M, K, C);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[huffman_kernel] linear kernel error: %s\n",
                hipGetErrorString(err));
        return HK_ERR_HIP;
    }
    return HK_SUCCESS;
}

/* --------------------------------------------------------------------------
 * huff_i32_embedding_f32
 * -------------------------------------------------------------------------- */
int huff_i32_embedding_f32(
    const void*  token_ids,
    const void*  i32_indices,
    const void*  codebook,
    void*        out,
    int T, int H, int C,
    hipStream_t  stream_h)
{
    if (!token_ids || !i32_indices || !codebook || !out) return HK_ERR_ARG;
    if (T <= 0 || H <= 0) return HK_ERR_ARG;

    dim3 grid((unsigned)T, (unsigned)((H + EMB_BLK - 1) / EMB_BLK));
    dim3 blk(EMB_BLK);

    huff_i32_embedding_kernel<<<grid, blk, 0, stream_h>>>(
        (const int32_t*)token_ids,
        (const int32_t*)i32_indices,
        (const float*)codebook,
        (float*)out,
        T, H, C);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[huffman_kernel] embedding kernel error: %s\n",
                hipGetErrorString(err));
        return HK_ERR_HIP;
    }
    return HK_SUCCESS;
}

} /* extern "C" */
