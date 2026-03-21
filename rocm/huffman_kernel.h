/*
 * huffman_kernel.h
 *
 * Public C API for standalone HIP Huffman decode + codebook matmul kernels.
 * No PyTorch or other framework dependency — link against libhuffman_kernel.so.
 *
 * All device pointers must be allocated with hipMalloc (or equivalent).
 *
 * Build:
 *   cd rocm && make libhuffman_kernel.so
 *
 * Link:
 *   -L<path> -lhuffman_kernel -lhip
 */

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes ─────────────────────────────────────────────────────────── */
#define HK_SUCCESS   0   /* no error                */
#define HK_ERR_HIP   1   /* HIP runtime error       */
#define HK_ERR_ARG   2   /* invalid argument        */

/* ── LUT constants ───────────────────────────────────────────────────────── */
#define HUFF_GPU_LUT_BITS  12
#define HUFF_GPU_LUT_SIZE  4096    /* 1 << HUFF_GPU_LUT_BITS */
#define HUFF_GPU_NO_SYM   (-1)    /* sentinel: no code ends at this prefix */

/* ── huff_decode_to_i32 ──────────────────────────────────────────────────── */
/*
 * Decode an MSB-first Huffman bitstream into a flat int32 index buffer.
 *
 * All device pointers must be on the same GPU.
 *
 * stream         : uint8_t* [N_bytes + 3]  MSB-first bitstream (+3 pad bytes)
 * row_bit_start  : int64_t* [M]            bit offset for start of row i
 * lut_sym        : int32_t* [4096]         12-bit LUT — decoded symbol
 *                                          (-1 = no short code at this prefix)
 * lut_len        : int32_t* [4096]         12-bit LUT — bits consumed
 * sl_first_code  : int32_t* [max_len+2]    slow-path: first code per length
 *                                          (-1 if no symbols at that length)
 * sl_base_offset : int32_t* [max_len+2]    slow-path: offset into sl_sym[]
 * sl_sym         : int32_t* [N_long]       slow-path: symbols for codes > 12b
 * out            : int32_t* [M * K]        decoded codebook index output
 * M, K           : weight matrix rows and columns
 * max_code_len   : maximum Huffman code length (bound for slow-path loop)
 * stream_h       : HIP stream (NULL = default)
 *
 * Returns HK_SUCCESS on success, HK_ERR_* on error.
 */
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
    hipStream_t  stream_h);

/* ── huff_i32_linear_f32 ─────────────────────────────────────────────────── */
/*
 * Codebook matmul using int32 indices.
 * out[tok, ofeat] = Σ_k  x[tok,k] * codebook[ i32_indices[ofeat*K + k] ]
 *
 * x            : float*   [T, K]   input activations
 * i32_indices  : int32_t* [M * K]  codebook indices
 * codebook     : float*   [C]      float32 codebook
 * out          : float*   [T, M]   output (caller must initialise to 0)
 * T, M, K, C   : tensor dimensions
 * stream_h     : HIP stream (NULL = default)
 */
int huff_i32_linear_f32(
    const void*  x,
    const void*  i32_indices,
    const void*  codebook,
    void*        out,
    int T, int M, int K, int C,
    hipStream_t  stream_h);

/* ── huff_i32_embedding_f32 ──────────────────────────────────────────────── */
/*
 * Codebook embedding using int32 indices.
 * out[tok, h] = codebook[ i32_indices[token_id * H + h] ]
 *
 * token_ids    : int32_t* [T]          input token IDs
 * i32_indices  : int32_t* [vocab * H]  codebook indices
 * codebook     : float*   [C]          float32 codebook
 * out          : float*   [T, H]       output
 * T, H, C      : tensor dimensions
 * stream_h     : HIP stream (NULL = default)
 */
int huff_i32_embedding_f32(
    const void*  token_ids,
    const void*  i32_indices,
    const void*  codebook,
    void*        out,
    int T, int H, int C,
    hipStream_t  stream_h);

#ifdef __cplusplus
}
#endif
