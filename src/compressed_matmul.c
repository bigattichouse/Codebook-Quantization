/*
 * compressed_matmul.c
 *
 * Compressed linear layer forward pass — no weight matrix ever materialised.
 *
 * For each output element (t, r):
 *   out[t, r] = sum_k  x[t, k] * codebook[ unpack(packed, r*K + k, bits) ]
 *
 * The weight value `w` is a scalar register; only the packed bytes and codebook
 * (already resident in L2/L3) are read.  No intermediate float array is created.
 *
 * Compile:
 *   gcc -O3 -march=native -shared -fPIC -o compressed_matmul.so compressed_matmul.c
 */

#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

/* ---------------------------------------------------------------------------
 * Bit unpacking helpers
 * ------------------------------------------------------------------------- */

/* Generic: extract `bits`-wide index at logical position `elem` from
 * a tightly-packed byte array.  Works for any bits in [1, 16].
 * Reads at most 3 bytes starting at byte_pos — caller must ensure the packed
 * array has at least ceil(N * bits / 8) + 2 bytes (2 byte padding at end).
 * elem and byte_pos use int64_t to avoid overflow for large tensors
 * (e.g. lm_head: 248320×1024×13 bits = 3.3B > INT32_MAX). */
static inline int
unpack_idx(const uint8_t *packed, int64_t elem, int bits)
{
    int64_t  bit_pos  = elem * bits;
    int64_t  byte_pos = bit_pos >> 3;    /* / 8  */
    int      shift    = (int)(bit_pos & 7); /* % 8  */
    // Use uint64_t or 4 bytes to avoid overflow and ensure enough bits
    // for indices up to 16 bits starting at any shift.
    uint32_t window = (uint32_t)packed[byte_pos]
                    | ((uint32_t)packed[byte_pos + 1] << 8)
                    | ((uint32_t)packed[byte_pos + 2] << 16)
                    | ((uint32_t)packed[byte_pos + 3] << 24);
    return (int)((window >> shift) & (uint32_t)((1u << bits) - 1u));
}

/* 8-bit fast path — direct byte index, no bit arithmetic. */
static inline int
unpack_idx_8(const uint8_t *packed, int64_t elem)
{
    return (int)packed[elem];
}

/* ---------------------------------------------------------------------------
 * Main kernel
 * ------------------------------------------------------------------------- */

/*
 * compressed_matmul_f32
 *
 *   x        : (T, K)  float32, row-major — input activations
 *   packed   : bit-packed weight indices, ceil(M*K*bits/8) + 2 pad bytes
 *   codebook : (C,)    float32 — codebook values
 *   out      : (T, M)  float32, row-major — output (caller must zero-init)
 *   T        : batch size (tokens)
 *   M        : output features (weight rows)
 *   K        : input features  (weight cols)
 *   bits     : index bit-width (8 or 13 for lossless)
 *
 * With -O3 gcc auto-vectorises the inner `t` loop (SIMD over batch).
 * For T=1 (autoregressive inference), the k-loop reduces to:
 *   out[r] += x[k] * codebook[idx]   (scalar FMA chain, ~1 cycle/iter at ILP).
 */
void
compressed_matmul_f32(
        const float   *x,
        const uint8_t *packed,
        const float   *codebook,
        float         *out,
        int T, int M, int K, int C, int bits)
{
    if (bits == 8) {
        for (int r = 0; r < M; r++) {
            for (int k = 0; k < K; k++) {
                int idx = unpack_idx_8(packed, (int64_t)r * K + k);
                if (idx >= C) idx = C - 1;  /* clamp, matches GPU kernel */
                float w = codebook[idx];
                for (int t = 0; t < T; t++)
                    out[t * M + r] += x[t * K + k] * w;
            }
        }
    } else {
        for (int r = 0; r < M; r++) {
            for (int k = 0; k < K; k++) {
                int idx = unpack_idx(packed, (int64_t)r * K + k, bits);
                if (idx >= C) idx = C - 1;  /* clamp, matches GPU kernel */
                float w = codebook[idx];
                for (int t = 0; t < T; t++)
                    out[t * M + r] += x[t * K + k] * w;
            }
        }
    }
}

/*
 * compressed_matmul_f32_chunk
 *
 * Same as above but processes only rows [r_start, r_end).
 * `out` is still indexed as (T, M) — caller allocates full (T, M) output
 * and passes the same pointer regardless of which chunk is being filled.
 * Allows the Python side to stream through a large layer in small chunks
 * without any additional allocation.
 */
void
compressed_matmul_f32_chunk(
        const float   *x,
        const uint8_t *packed,
        const float   *codebook,
        float         *out,
        int T, int M, int K, int C, int bits,
        int r_start, int r_end)
{
    if (r_end > M) r_end = M;

    if (bits == 8) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int r = r_start; r < r_end; r++) {
            for (int k = 0; k < K; k++) {
                int idx = unpack_idx_8(packed, (int64_t)r * K + k);
                if (idx >= C) idx = C - 1;
                float w = codebook[idx];
                for (int t = 0; t < T; t++)
                    out[t * M + r] += x[t * K + k] * w;
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int r = r_start; r < r_end; r++) {
            for (int k = 0; k < K; k++) {
                int idx = unpack_idx(packed, (int64_t)r * K + k, bits);
                if (idx >= C) idx = C - 1;
                float w = codebook[idx];
                for (int t = 0; t < T; t++)
                    out[t * M + r] += x[t * K + k] * w;
            }
        }
    }
}
