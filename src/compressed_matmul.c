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
 * Raw (uncompressed) matmul
 * ---------------------------------------------------------------------------
 *
 * raw_matmul_f32
 *
 *   x      : (T, K)  float32, row-major — input activations
 *   weight : (M, K)  float32, row-major — plain float weight matrix
 *   out    : (T, M)  float32, row-major — output (caller must zero-init)
 *   T, M, K: dimensions
 *
 * Equivalent to: out = x @ weight.T
 * No codebook, no bit-packing. With -O3 -march=native gcc auto-vectorises.
 */
void
raw_matmul_f32(
        const float *x,
        const float *weight,
        float       *out,
        int T, int M, int K)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < M; r++) {
        const float *wr = weight + (int64_t)r * K;
        for (int k = 0; k < K; k++) {
            float w = wr[k];
            for (int t = 0; t < T; t++)
                out[t * M + r] += x[t * K + k] * w;
        }
    }
}

/*
 * raw_matmul_f32_chunk
 *
 * Same as above but processes only rows [r_start, r_end).
 * Allows streaming large layers with a fixed working set.
 */
void
raw_matmul_f32_chunk(
        const float *x,
        const float *weight,
        float       *out,
        int T, int M, int K,
        int r_start, int r_end)
{
    if (r_end > M) r_end = M;

    #pragma omp parallel for schedule(static)
    for (int r = r_start; r < r_end; r++) {
        const float *wr = weight + (int64_t)r * K;
        for (int k = 0; k < K; k++) {
            float w = wr[k];
            for (int t = 0; t < T; t++)
                out[t * M + r] += x[t * K + k] * w;
        }
    }
}

/*
 * raw_embedding_f32
 *
 *   token_ids : (T,)        int32  — token IDs
 *   weight    : (vocab, H)  float32 row-major — full embedding table
 *   out       : (T, H)      float32 row-major — output (caller allocs)
 *
 * Equivalent to: out = weight[token_ids]
 */
void
raw_embedding_f32(
        const int32_t *token_ids,
        const float   *weight,
        float         *out,
        int T, int H)
{
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        int32_t tid = token_ids[t];
        memcpy(out + (int64_t)t * H, weight + (int64_t)tid * H, (size_t)H * sizeof(float));
    }
}

/* ---------------------------------------------------------------------------
 * Huffman decode helpers (MSB-first bitstream)
 * ---------------------------------------------------------------------------
 *
 * The Huffman bitstream is MSB-first within each byte:
 *   bit_pos 0  → byte 0, bit 7 (MSB of first byte)
 *   bit_pos 7  → byte 0, bit 0 (LSB of first byte)
 *   bit_pos 8  → byte 1, bit 7
 *
 * The stream always has 4 zero-pad bytes appended at encode time, so reading
 * a 4-byte window starting at any valid bit position is always safe.
 */

/*
 * huff_read_bits — peek n bits (n ≤ 25) at absolute bit_pos without advancing.
 * Returns the bits as the low-order bits of a uint32_t, MSB first.
 */
static inline uint32_t
huff_read_bits(const uint8_t *stream, int64_t bit_pos, int n)
{
    int64_t  byte_pos = bit_pos >> 3;
    int      bit_off  = (int)(bit_pos & 7);  /* bits already consumed in this byte */
    uint32_t w = ((uint32_t)stream[byte_pos    ] << 24)
               | ((uint32_t)stream[byte_pos + 1] << 16)
               | ((uint32_t)stream[byte_pos + 2] <<  8)
               |  (uint32_t)stream[byte_pos + 3];
    return (w << bit_off) >> (32 - n);
}

/*
 * huff_decode_one — decode one Huffman symbol, advancing *bit_pos.
 *
 * Fast path: 12-bit LUT (covers >99% of symbols in practice).
 *   lut_sym[key] : uint16, codebook symbol (0xFFFF unused since lut_len check comes first)
 *   lut_len[key] : uint8,  code length (0 = no match → take slow path)
 *
 * Slow path: extend bit by bit for codes longer than 12 bits.
 *   sl_first_code[L]  : int64, first canonical code at length L (-1 = none)
 *   sl_base_offset[L] : int32, offset into sl_sym[] for length L
 *   sl_sym[]          : uint16, symbols sorted by (length, code order)
 *   sl_max_len        : maximum code length in this table
 */
#define HUFF_LUT_BITS 12

static inline int
huff_decode_one(const uint8_t   *stream,
                int64_t         *bit_pos,
                const uint16_t  *lut_sym,
                const uint8_t   *lut_len,
                const int64_t   *sl_first_code,
                const int32_t   *sl_base_offset,
                const uint16_t  *sl_sym,
                int              sl_max_len)
{
    /* --- Fast path: 12-bit LUT --- */
    uint32_t key = huff_read_bits(stream, *bit_pos, HUFF_LUT_BITS);
    int      len = (int)lut_len[key];
    if (len > 0) {
        *bit_pos += len;
        return (int)lut_sym[key];
    }

    /* --- Slow path: extend bit-by-bit beyond 12 bits --- */
    uint32_t code = key;   /* already have HUFF_LUT_BITS bits */
    for (int L = HUFF_LUT_BITS + 1; L <= sl_max_len; L++) {
        uint32_t nb = huff_read_bits(stream, *bit_pos + (int64_t)(L - 1), 1);
        code = (code << 1) | nb;
        int64_t fc = sl_first_code[L];
        if (fc >= 0) {
            int64_t delta = (int64_t)code - fc;
            if (delta >= 0) {
                int32_t cnt = sl_base_offset[L + 1] - sl_base_offset[L];
                if (delta < (int64_t)cnt) {
                    *bit_pos += L;
                    return (int)sl_sym[sl_base_offset[L] + (int)delta];
                }
            }
        }
    }
    /* Corrupt stream guard: skip 1 bit and return 0 */
    *bit_pos += 1;
    return 0;
}

/*
 * huffman_matmul_f32_chunk
 *
 * Like compressed_matmul_f32_chunk but weights are stored as a Huffman
 * bitstream.  Decodes K symbols per row on-the-fly — no unpacked index buffer
 * is ever created.  Rows are independent → parallelised with OpenMP.
 *
 *   x              : (T, K)  float32, row-major
 *   huff_stream    : uint8[] MSB-first Huffman bitstream (+4 pad bytes at end)
 *   lut_sym        : uint16[4096] 12-bit LUT symbols
 *   lut_len        : uint8[4096]  12-bit LUT code lengths (0 = no LUT match)
 *   sl_first_code  : int64[sl_max_len+2] first canonical code per length > 12
 *   sl_base_offset : int32[sl_max_len+2] offsets into sl_sym; [max+1] = sentinel
 *   sl_sym         : uint16[N] slow-path symbols
 *   row_bit_starts : int64[M]  bit offset of row r in huff_stream
 *   codebook       : float32[C]
 *   out            : (T, M)  float32, row-major — caller must zero-initialise
 *   r_start,r_end  : row range [r_start, r_end)
 *   sl_max_len     : maximum code length (slow-path loop bound)
 */
void
huffman_matmul_f32_chunk(
        const float    *x,
        const uint8_t  *huff_stream,
        const uint16_t *lut_sym,
        const uint8_t  *lut_len,
        const int64_t  *sl_first_code,
        const int32_t  *sl_base_offset,
        const uint16_t *sl_sym,
        const int64_t  *row_bit_starts,
        const float    *codebook,
        float          *out,
        int T, int M, int K, int C,
        int r_start, int r_end, int sl_max_len)
{
    if (r_end > M) r_end = M;

    #pragma omp parallel for schedule(dynamic, 64)
    for (int r = r_start; r < r_end; r++) {
        int64_t bit_pos = row_bit_starts[r];
        for (int k = 0; k < K; k++) {
            int sym = huff_decode_one(huff_stream, &bit_pos,
                                      lut_sym, lut_len,
                                      sl_first_code, sl_base_offset, sl_sym,
                                      sl_max_len);
            if (sym >= C) sym = C - 1;
            float w = codebook[sym];
            for (int t = 0; t < T; t++)
                out[t * M + r] += x[t * K + k] * w;
        }
    }
}

/*
 * huffman_embedding_f32_rows
 *
 * Embedding lookup for Huffman-compressed tables.  Decodes H symbols per
 * token row — only the rows actually requested are decoded, not the full vocab.
 *
 *   token_ids      : int32[T]
 *   (huff fields)  : same layout as huffman_matmul_f32_chunk
 *   row_bit_starts : int64[vocab]  bit offset for token id t
 *   out            : (T, H) float32, row-major — overwritten (not accumulated)
 *   T, H, C        : dimensions
 */
void
huffman_embedding_f32_rows(
        const int32_t  *token_ids,
        const uint8_t  *huff_stream,
        const uint16_t *lut_sym,
        const uint8_t  *lut_len,
        const int64_t  *sl_first_code,
        const int32_t  *sl_base_offset,
        const uint16_t *sl_sym,
        const int64_t  *row_bit_starts,
        const float    *codebook,
        float          *out,
        int T, int H, int C, int sl_max_len)
{
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        int32_t tok     = token_ids[t];
        int64_t bit_pos = row_bit_starts[(int64_t)tok];
        float  *out_row = out + (int64_t)t * H;
        for (int h = 0; h < H; h++) {
            int sym = huff_decode_one(huff_stream, &bit_pos,
                                      lut_sym, lut_len,
                                      sl_first_code, sl_base_offset, sl_sym,
                                      sl_max_len);
            if (sym >= C) sym = C - 1;
            out_row[h] = codebook[sym];
        }
    }
}

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
