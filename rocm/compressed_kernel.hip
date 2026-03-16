/*
 * compressed_kernel.hip
 *
 * Standalone HIP implementation of compressed matmul and embedding.
 * No PyTorch or other ML framework dependency.
 *
 * Build with:
 *   hipcc -O3 --offload-arch=<arch> -fPIC -shared \
 *         -o libcompressed_kernel.so compressed_kernel.hip
 *
 * See Makefile for automatic arch detection.
 */

#include "compressed_kernel.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


/* ── Bit unpacking ───────────────────────────────────────────────────────────
 *
 * Reads 4 bytes at byte_pos and extracts `bits` bits starting at bit_shift.
 * The caller must ensure packed has at least 4 bytes of padding at the end
 * so the 4-byte window read is always safe.
 *
 * Uses int64_t for bit/byte positions to avoid overflow on large tensors
 * (e.g. lm_head: 248320×1024 × 13 bits = 3.3 billion > INT32_MAX).
 */
__device__ __forceinline__ uint32_t unpack_idx(
    const uint8_t* __restrict__ packed,
    int64_t logical_idx,
    int bits)
{
    int64_t  bit_pos  = logical_idx * (int64_t)bits;
    int64_t  byte_pos = bit_pos >> 3;
    int      bit_shft = (int)(bit_pos & 7);
    uint32_t mask     = (1u << bits) - 1u;

    uint32_t w = (uint32_t)packed[byte_pos]
               | ((uint32_t)packed[byte_pos + 1] << 8)
               | ((uint32_t)packed[byte_pos + 2] << 16)
               | ((uint32_t)packed[byte_pos + 3] << 24);
    return (w >> bit_shft) & mask;
}


/* ── Linear kernel (templated on scalar I/O type) ───────────────────────────
 *
 * Grid : 1-D, T*M blocks  (avoids the 65535 limit on grid.y for large M)
 * Block: BLOCK threads
 *
 * Each block owns one (token, output-feature) pair and reduces over K.
 *
 * T: scalar type for x and out (float or hip_bfloat16)
 */
template <int BLOCK, typename T>
__global__ void compressed_linear_kernel(
    const T*       __restrict__ x,
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ codebook,
    T*             __restrict__ out,
    int total_T, int M, int K, int C, int bits)
{
    int idx   = (int)blockIdx.x;
    int tok   = idx / M;
    int ofeat = idx - tok * M;
    int tid   = (int)threadIdx.x;

    __shared__ float sh[BLOCK];

    const T* xt = x + (int64_t)tok * K;
    float acc = 0.f;

    for (int j = tid; j < K; j += BLOCK) {
        int64_t  li = (int64_t)ofeat * K + j;
        uint32_t ci = unpack_idx(packed, li, bits);
        if ((int)ci >= C) ci = (uint32_t)(C - 1);
        acc += (float)xt[j] * codebook[ci];
    }
    sh[tid] = acc;
    __syncthreads();

    /* parallel reduction */
    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[(int64_t)tok * M + ofeat] = (T)sh[0];
}


/* ── Embedding kernel ────────────────────────────────────────────────────────
 *
 * Grid : (T, ceil(H/BLOCK))
 * Block: BLOCK threads
 *
 * Each thread writes one output element — embarrassingly parallel.
 * Always outputs float32 (callers cast if needed).
 */
template <int BLOCK>
__global__ void compressed_embedding_kernel(
    const int32_t* __restrict__ token_ids,
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ codebook,
    float*         __restrict__ out,
    int T, int H, int C, int bits)
{
    int tok = (int)blockIdx.x;
    int h   = (int)blockIdx.y * BLOCK + (int)threadIdx.x;
    if (h >= H) return;

    int32_t  token_id = token_ids[tok];
    int64_t  li  = (int64_t)token_id * H + h;
    uint32_t ci  = unpack_idx(packed, li, bits);
    if ((int)ci >= C) ci = (uint32_t)(C - 1);
    out[(int64_t)tok * H + h] = codebook[ci];
}


/* ── Raw (uncompressed) linear kernel ───────────────────────────────────────
 *
 * Grid : 1-D, T*M blocks  (same layout as compressed_linear_kernel)
 * Block: BLOCK threads
 *
 * out[tok, ofeat] = Σ_j  x[tok, j] * weight[ofeat, j]
 * weight is row-major [M, K]; row ofeat starts at weight + ofeat*K.
 */
template <int BLOCK, typename T>
__global__ void raw_linear_kernel(
    const T* __restrict__ x,        /* [T, K] */
    const T* __restrict__ weight,   /* [M, K] row-major */
    T*       __restrict__ out,      /* [T, M] */
    int total_T, int M, int K)
{
    int idx   = (int)blockIdx.x;
    int tok   = idx / M;
    int ofeat = idx - tok * M;
    int tid   = (int)threadIdx.x;

    __shared__ float sh[BLOCK];

    const T* xt = x      + (int64_t)tok   * K;
    const T* wt = weight + (int64_t)ofeat * K;
    float acc = 0.f;

    for (int j = tid; j < K; j += BLOCK)
        acc += (float)xt[j] * (float)wt[j];

    sh[tid] = acc;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[(int64_t)tok * M + ofeat] = (T)sh[0];
}


/* ── Raw (uncompressed) embedding kernel ─────────────────────────────────────
 *
 * Grid : (T, ceil(H/BLOCK))
 * Block: BLOCK threads
 *
 * out[tok, h] = weight[token_ids[tok], h]
 * weight is [vocab, H]; each row is a full embedding vector.
 */
template <int BLOCK, typename T>
__global__ void raw_embedding_kernel(
    const int32_t* __restrict__ token_ids,
    const T*       __restrict__ weight,   /* [vocab, H] */
    T*             __restrict__ out,      /* [T, H] */
    int T_tok, int H)
{
    int tok = (int)blockIdx.x;
    int h   = (int)blockIdx.y * BLOCK + (int)threadIdx.x;
    if (h >= H) return;

    int32_t tid = token_ids[tok];
    out[(int64_t)tok * H + h] = weight[(int64_t)tid * H + h];
}


/* ── Causal depthwise conv1d kernel (prefill) ────────────────────────────────
 *
 * Depthwise causal 1-D convolution with SiLU activation.
 * Replaces nn.Conv1d(groups=C, padding=kernel_size-1)[:, :, :seq_len].
 *
 * Grid : (B * C)      — one block per (batch, channel) strip
 * Block: BLOCK threads — each handles one or more time positions
 *
 * x      : [B, C, L]  input (float32)
 * weight : [C, ksz]   depthwise weights
 * out    : [B, C, L]  output with SiLU applied
 */
template <int BLOCK>
__global__ void causal_conv1d_prefill_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float*       __restrict__ out,
    int B, int C, int L, int ksz)
{
    int bc  = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    const float* xbc = x      + (int64_t)bc * L;
    const float* wc  = weight + (bc % C) * ksz;
    float*       obc = out    + (int64_t)bc * L;

    /* Load weights into registers — ksz is small (typically 4). */
    float w0 = (ksz > 0) ? wc[0] : 0.f;
    float w1 = (ksz > 1) ? wc[1] : 0.f;
    float w2 = (ksz > 2) ? wc[2] : 0.f;
    float w3 = (ksz > 3) ? wc[3] : 0.f;

    for (int t = tid; t < L; t += BLOCK) {
        float acc = xbc[t] * w0;
        if (ksz > 1 && t >= 1) acc += xbc[t - 1] * w1;
        if (ksz > 2 && t >= 2) acc += xbc[t - 2] * w2;
        if (ksz > 3 && t >= 3) acc += xbc[t - 3] * w3;
        obc[t] = acc / (1.f + expf(-acc));  /* SiLU */
    }
}


/* ── Causal conv1d decode update kernel ──────────────────────────────────────
 *
 * Single-step depthwise conv1d state update for autoregressive decoding.
 * Mirrors torch_causal_conv1d_update: roll state left, insert x_new, dot with w.
 *
 * Grid : ceil(B*C / BLOCK)
 * Block: BLOCK — each thread owns one (b, c) channel
 *
 * x_new      : [B, C]       new input token (float32)
 * conv_state : [B, C, ksz]  rolling conv buffer (updated in-place)
 * weight     : [C, ksz]
 * out        : [B, C]       SiLU(conv_output)
 */
template <int BLOCK>
__global__ void causal_conv1d_update_kernel(
    const float* __restrict__ x_new,
    float*                    conv_state,
    const float* __restrict__ weight,
    float*       __restrict__ out,
    int B, int C, int ksz)
{
    int bc = (int)blockIdx.x * BLOCK + (int)threadIdx.x;
    if (bc >= B * C) return;

    float  x_cur  = x_new[bc];
    float* state  = conv_state + (int64_t)bc * ksz;
    const float* w = weight + (bc % C) * ksz;

    /* Roll state left, accumulate dot product, insert x_cur at end. */
    float acc = 0.f;
    for (int k = 0; k < ksz - 1; k++) {
        float s  = state[k + 1];
        state[k] = s;
        acc += s * w[k];
    }
    state[ksz - 1] = x_cur;
    acc += x_cur * w[ksz - 1];

    out[bc] = acc / (1.f + expf(-acc));  /* SiLU */
}


/* ── Gated Delta Rule — single decode step ───────────────────────────────────
 *
 * One timestep of the Gated DeltaNet recurrence (seq_len = 1).
 * State H[h] ∈ ℝ^{KD × VD} per head h.
 *
 * Algorithm (per head h):
 *   H   *= exp(log_g[h])               decay
 *   kv   = H @ k[h]                    project state → value space [VD]
 *   δ    = (v[h] - kv) * β[h]          delta correction
 *   H   += outer(k[h], δ)              rank-1 state update
 *   y[h] = H @ q[h]                    readout [VD]
 *
 * Grid : (B × H)   — one block per (batch, head)
 * Block: BLOCK=64  — one wavefront; threads span the VD dimension
 *
 * q, k : [B, H, KD]  float32  (caller applies L2-norm and 1/sqrt(KD) scaling)
 * v    : [B, H, VD]  float32
 * log_g: [B, H]      float32  log-decay (negative → decay < 1)
 * beta : [B, H]      float32
 * state: [B, H, KD, VD]  float32  (updated in-place)
 * out  : [B, H, VD]  float32
 *
 * Dynamic shared memory required: KD * VD * sizeof(float) bytes (<= 64 KB).
 */
#define GDR_MAX_VI 16   /* max iterations over VD: supports VD up to BLOCK*16=1024 */

template <int BLOCK>
__global__ void gdr_decode_step_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ log_g,
    const float* __restrict__ beta,
    float*                    state,
    float*       __restrict__ out,
    int B, int H, int KD, int VD)
{
    extern __shared__ float sh[];   /* [KD, VD] — state for this (b, h) */

    int bh  = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    float g_val  = expf(log_g[bh]);
    float beta_h = beta[bh];

    const float* q_h = q     + bh * KD;
    const float* k_h = k     + bh * KD;
    const float* v_h = v     + bh * VD;
    float*       s_h = state + (int64_t)bh * KD * VD;
    float*       o_h = out   + bh * VD;

    /* Load state into shared memory for the 3 passes below. */
    int total = KD * VD;
    for (int i = tid; i < total; i += BLOCK)
        sh[i] = s_h[i];
    __syncthreads();

    /* Each thread owns VD positions: vd = tid, tid+BLOCK, tid+2*BLOCK, ... */
    int n_vi = (VD + BLOCK - 1) / BLOCK;

    /* Pass 1: decay + kv accumulation */
    float kv_acc[GDR_MAX_VI] = {};
    for (int kd = 0; kd < KD; kd++) {
        float k_kd = k_h[kd];
        float* row = sh + kd * VD;
        for (int vi = 0; vi < n_vi; vi++) {
            int vd = tid + vi * BLOCK;
            if (vd < VD) {
                row[vd] *= g_val;
                kv_acc[vi] += row[vd] * k_kd;
            }
        }
    }

    /* Pass 2: delta = (v - kv) * beta, then rank-1 update */
    float delta[GDR_MAX_VI] = {};
    for (int vi = 0; vi < n_vi; vi++) {
        int vd = tid + vi * BLOCK;
        if (vd < VD)
            delta[vi] = (v_h[vd] - kv_acc[vi]) * beta_h;
    }
    for (int kd = 0; kd < KD; kd++) {
        float k_kd = k_h[kd];
        float* row = sh + kd * VD;
        for (int vi = 0; vi < n_vi; vi++) {
            int vd = tid + vi * BLOCK;
            if (vd < VD)
                row[vd] += k_kd * delta[vi];
        }
    }

    /* Pass 3: readout y = H @ q */
    float y[GDR_MAX_VI] = {};
    for (int kd = 0; kd < KD; kd++) {
        float q_kd = q_h[kd];
        float* row = sh + kd * VD;
        for (int vi = 0; vi < n_vi; vi++) {
            int vd = tid + vi * BLOCK;
            if (vd < VD)
                y[vi] += row[vd] * q_kd;
        }
    }

    /* Write updated state and output */
    __syncthreads();
    for (int i = tid; i < total; i += BLOCK)
        s_h[i] = sh[i];
    for (int vi = 0; vi < n_vi; vi++) {
        int vd = tid + vi * BLOCK;
        if (vd < VD)
            o_h[vd] = y[vi];
    }
}

#undef GDR_MAX_VI


/* ── GDR sequential prefill kernel ──────────────────────────────────────────
 *
 * Processes a full input sequence in a single kernel launch.
 * The state [KD, VD] lives in shared memory (LDS) across ALL tokens —
 * no round-trip to global memory between steps.
 *
 * Input shapes (all float32, contiguous):
 *   q, k  : [B, SL, H, KD]
 *   v     : [B, SL, H, VD]
 *   log_g : [B, SL, H]        log-decay (will be exp'd)
 *   beta  : [B, SL, H]
 *   state : [B, H, KD, VD]   (updated in-place — pass zeros or prior state)
 *   out   : [B, SL, H, VD]   (caller allocates; can be uninitialised)
 *
 * Grid : (B * H)  blocks — one block per (batch, head)
 * Block: BLOCK threads
 * LDS  : KD * VD * sizeof(float) bytes  (<= 64 KB)
 *
 * Each thread owns columns  { vd = tid + vi*BLOCK : vi in 0..n_vi-1 }
 * of the KD×VD state matrix.  Threads are entirely independent — no
 * inter-thread shared-memory communication within the token loop.
 */
#define GDR_MAX_VI 16   /* ceil(VD/BLOCK): supports VD up to BLOCK*16=1024 */

template <int BLOCK>
__global__ void gdr_sequential_prefill_kernel(
    const float* __restrict__ q,      /* [B, SL, H, KD] */
    const float* __restrict__ k,      /* [B, SL, H, KD] */
    const float* __restrict__ v,      /* [B, SL, H, VD] */
    const float* __restrict__ log_g,  /* [B, SL, H]     */
    const float* __restrict__ beta,   /* [B, SL, H]     */
    float*                    state,  /* [B, H, KD, VD] (in-place) */
    float*       __restrict__ out,    /* [B, SL, H, VD] */
    int B, int SL, int H, int KD, int VD)
{
    extern __shared__ float sh[];   /* [KD, VD] — state for this (b, h) */

    int bh  = (int)blockIdx.x;
    int b   = bh / H;
    int h   = bh % H;
    int tid = (int)threadIdx.x;

    float* s_h    = state + (int64_t)bh * KD * VD;
    int    total  = KD * VD;
    int    n_vi   = (VD + BLOCK - 1) / BLOCK;

    /* Load initial state into shared memory once. */
    for (int i = tid; i < total; i += BLOCK)
        sh[i] = s_h[i];
    __syncthreads();

    /* Per-token loop — state stays in LDS between iterations. */
    for (int t = 0; t < SL; t++) {
        int64_t qk_off = ((int64_t)b * SL * H + (int64_t)t * H + h) * KD;
        int64_t v_off  = ((int64_t)b * SL * H + (int64_t)t * H + h) * VD;
        int64_t sc_off =  (int64_t)b * SL * H + (int64_t)t * H + h;

        const float* q_t  = q    + qk_off;
        const float* k_t  = k    + qk_off;
        const float* v_t  = v    + v_off;
        float g_val  = expf(log_g[sc_off]);
        float beta_h = beta[sc_off];

        /* Pass 1: decay state + accumulate kv = H @ k */
        float kv_acc[GDR_MAX_VI];
        for (int vi = 0; vi < n_vi; vi++) kv_acc[vi] = 0.0f;

        for (int kd = 0; kd < KD; kd++) {
            float k_kd = k_t[kd];
            float* row = sh + kd * VD;
            for (int vi = 0; vi < n_vi; vi++) {
                int vd = tid + vi * BLOCK;
                if (vd < VD) {
                    row[vd] *= g_val;
                    kv_acc[vi] += row[vd] * k_kd;
                }
            }
        }

        /* Pass 2: delta = (v - kv) * beta, then rank-1 update H += k ⊗ delta */
        float delta[GDR_MAX_VI];
        for (int vi = 0; vi < n_vi; vi++) {
            int vd = tid + vi * BLOCK;
            delta[vi] = (vd < VD) ? (v_t[vd] - kv_acc[vi]) * beta_h : 0.0f;
        }
        for (int kd = 0; kd < KD; kd++) {
            float k_kd = k_t[kd];
            float* row = sh + kd * VD;
            for (int vi = 0; vi < n_vi; vi++) {
                int vd = tid + vi * BLOCK;
                if (vd < VD)
                    row[vd] += k_kd * delta[vi];
            }
        }

        /* Pass 3: readout y = H @ q, write output token */
        int64_t out_off = ((int64_t)b * SL * H + (int64_t)t * H + h) * VD;
        float*       o_t = out + out_off;
        float y[GDR_MAX_VI];
        for (int vi = 0; vi < n_vi; vi++) y[vi] = 0.0f;

        for (int kd = 0; kd < KD; kd++) {
            float q_kd = q_t[kd];
            float* row = sh + kd * VD;
            for (int vi = 0; vi < n_vi; vi++) {
                int vd = tid + vi * BLOCK;
                if (vd < VD)
                    y[vi] += row[vd] * q_kd;
            }
        }
        for (int vi = 0; vi < n_vi; vi++) {
            int vd = tid + vi * BLOCK;
            if (vd < VD)
                o_t[vd] = y[vi];
        }
    } /* end token loop */

    /* Write final state back to global memory. */
    __syncthreads();
    for (int i = tid; i < total; i += BLOCK)
        s_h[i] = sh[i];
}

#undef GDR_MAX_VI


/* ── Internal launch helpers ─────────────────────────────────────────────── */

template <typename T>
static int launch_linear(
    const T*       x,
    const uint8_t* packed,
    const float*   codebook,
    T*             out,
    int T_tok, int M, int K, int C, int bits,
    hipStream_t stream)
{
    if (!x || !packed || !codebook || !out)          return CK_ERR_ARG;
    if (T_tok <= 0 || M <= 0 || K <= 0 || C <= 0)   return CK_ERR_ARG;
    if (bits < 1 || bits > 16)                       return CK_ERR_ARG;

    const int BLOCK = 256;
    dim3 grid((unsigned)(T_tok * M));
    dim3 blk(BLOCK);

    compressed_linear_kernel<BLOCK, T><<<grid, blk, 0, stream>>>(
        x, packed, codebook, out, T_tok, M, K, C, bits);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_linear] kernel error: %s\n", hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}


template <typename T>
static int launch_raw_linear(
    const T* x, const T* weight, T* out,
    int T_tok, int M, int K,
    hipStream_t stream)
{
    if (!x || !weight || !out)               return CK_ERR_ARG;
    if (T_tok <= 0 || M <= 0 || K <= 0)     return CK_ERR_ARG;

    const int BLOCK = 256;
    dim3 grid((unsigned)(T_tok * M));
    dim3 blk(BLOCK);

    raw_linear_kernel<BLOCK, T><<<grid, blk, 0, stream>>>(x, weight, out, T_tok, M, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_linear_raw] kernel error: %s\n", hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}

template <typename T>
static int launch_raw_embedding(
    const int32_t* token_ids, const T* weight, T* out,
    int T_tok, int H,
    hipStream_t stream)
{
    if (!token_ids || !weight || !out)   return CK_ERR_ARG;
    if (T_tok <= 0 || H <= 0)           return CK_ERR_ARG;

    const int BLOCK = 128;
    dim3 grid((unsigned)T_tok, (unsigned)((H + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    raw_embedding_kernel<BLOCK, T><<<grid, blk, 0, stream>>>(token_ids, weight, out, T_tok, H);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_embedding_raw] kernel error: %s\n", hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}


/* ── C API implementation ────────────────────────────────────────────────── */

const char* ck_error_string(int err)
{
    switch (err) {
        case CK_SUCCESS:  return "success";
        case CK_ERR_HIP:  return "HIP runtime error";
        case CK_ERR_ARG:  return "invalid argument";
        default:          return "unknown error";
    }
}

extern "C" {

int ck_linear_f32(
    const float*    x,
    const uint8_t*  packed,
    const float*    codebook,
    float*          out,
    int T, int M, int K, int C, int bits,
    hipStream_t stream)
{
    return launch_linear<float>(x, packed, codebook, out, T, M, K, C, bits, stream);
}

int ck_linear_bf16(
    const hip_bfloat16* x,
    const uint8_t*        packed,
    const float*          codebook,
    hip_bfloat16*       out,
    int T, int M, int K, int C, int bits,
    hipStream_t stream)
{
    return launch_linear<hip_bfloat16>(x, packed, codebook, out, T, M, K, C, bits, stream);
}

int ck_embedding_f32(
    const int32_t*  token_ids,
    const uint8_t*  packed,
    const float*    codebook,
    float*          out,
    int T, int H, int C, int bits,
    hipStream_t stream)
{
    if (!token_ids || !packed || !codebook || !out)  return CK_ERR_ARG;
    if (T <= 0 || H <= 0 || C <= 0)                 return CK_ERR_ARG;
    if (bits < 1 || bits > 16)                       return CK_ERR_ARG;

    const int BLOCK = 128;
    dim3 grid((unsigned)T, (unsigned)((H + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    compressed_embedding_kernel<BLOCK><<<grid, blk, 0, stream>>>(
        token_ids, packed, codebook, out, T, H, C, bits);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_embedding] kernel error: %s\n", hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}

uint8_t* ck_upload_packed(const uint8_t* host, size_t nbytes)
{
    uint8_t* dev = NULL;
    if (hipMalloc(&dev, nbytes) != hipSuccess) return NULL;
    if (hipMemcpy(dev, host, nbytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(dev);
        return NULL;
    }
    return dev;
}

float* ck_upload_codebook(const float* host, int C)
{
    float* dev = NULL;
    size_t nbytes = (size_t)C * sizeof(float);
    if (hipMalloc(&dev, nbytes) != hipSuccess) return NULL;
    if (hipMemcpy(dev, host, nbytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(dev);
        return NULL;
    }
    return dev;
}

/* ── Raw kernel C API ─────────────────────────────────────────────────────── */

int ck_linear_raw_f32(
    const float* x, const float* weight, float* out,
    int T, int M, int K, hipStream_t stream)
{
    return launch_raw_linear<float>(x, weight, out, T, M, K, stream);
}

int ck_linear_raw_bf16(
    const hip_bfloat16* x, const hip_bfloat16* weight, hip_bfloat16* out,
    int T, int M, int K, hipStream_t stream)
{
    return launch_raw_linear<hip_bfloat16>(x, weight, out, T, M, K, stream);
}

int ck_embedding_raw_f32(
    const int32_t* token_ids, const float* weight, float* out,
    int T, int H, hipStream_t stream)
{
    return launch_raw_embedding<float>(token_ids, weight, out, T, H, stream);
}

int ck_embedding_raw_bf16(
    const int32_t* token_ids, const hip_bfloat16* weight, hip_bfloat16* out,
    int T, int H, hipStream_t stream)
{
    return launch_raw_embedding<hip_bfloat16>(token_ids, weight, out, T, H, stream);
}

/* ── Raw weight upload helpers ────────────────────────────────────────────── */

float* ck_upload_weights_f32(const float* host, int rows, int cols)
{
    float* dev = NULL;
    size_t nbytes = (size_t)rows * cols * sizeof(float);
    if (hipMalloc(&dev, nbytes) != hipSuccess) return NULL;
    if (hipMemcpy(dev, host, nbytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(dev);
        return NULL;
    }
    return dev;
}

hip_bfloat16* ck_upload_weights_bf16(const hip_bfloat16* host, int rows, int cols)
{
    hip_bfloat16* dev = NULL;
    size_t nbytes = (size_t)rows * cols * sizeof(hip_bfloat16);
    if (hipMalloc(&dev, nbytes) != hipSuccess) return NULL;
    if (hipMemcpy(dev, host, nbytes, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(dev);
        return NULL;
    }
    return dev;
}

/* ── SSM kernel C API ─────────────────────────────────────────────────────── */

int ck_causal_conv1d_prefill_f32(
    const float* x, const float* weight, float* out,
    int B, int C, int L, int ksz,
    hipStream_t stream)
{
    if (!x || !weight || !out)           return CK_ERR_ARG;
    if (B <= 0 || C <= 0 || L <= 0)     return CK_ERR_ARG;
    if (ksz < 1 || ksz > 4)             return CK_ERR_ARG;

    const int BLOCK = 64;
    dim3 grid((unsigned)(B * C));
    dim3 blk(BLOCK);

    causal_conv1d_prefill_kernel<BLOCK><<<grid, blk, 0, stream>>>(
        x, weight, out, B, C, L, ksz);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_causal_conv1d_prefill] kernel error: %s\n",
                hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}

int ck_causal_conv1d_update_f32(
    const float* x_new, float* conv_state, const float* weight, float* out,
    int B, int C, int ksz,
    hipStream_t stream)
{
    if (!x_new || !conv_state || !weight || !out)  return CK_ERR_ARG;
    if (B <= 0 || C <= 0 || ksz < 1)              return CK_ERR_ARG;

    const int BLOCK = 128;
    int total = B * C;
    dim3 grid((unsigned)((total + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    causal_conv1d_update_kernel<BLOCK><<<grid, blk, 0, stream>>>(
        x_new, conv_state, weight, out, B, C, ksz);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_causal_conv1d_update] kernel error: %s\n",
                hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}

int ck_gdr_decode_step_f32(
    const float* q, const float* k, const float* v,
    const float* log_g, const float* beta,
    float* state, float* out,
    int B, int H, int KD, int VD,
    hipStream_t stream)
{
    if (!q || !k || !v || !log_g || !beta || !state || !out) return CK_ERR_ARG;
    if (B <= 0 || H <= 0 || KD <= 0 || VD <= 0)             return CK_ERR_ARG;

    size_t smem = (size_t)KD * VD * sizeof(float);
    if (smem > 65536) {
        /* State too large for LDS — fall back to caller's PyTorch path */
        fprintf(stderr, "[ck_gdr_decode_step] state %zu B > 64 KB LDS limit, "
                "refusing launch\n", smem);
        return CK_ERR_ARG;
    }

    const int BLOCK = 64;
    dim3 grid((unsigned)(B * H));
    dim3 blk(BLOCK);

    gdr_decode_step_kernel<BLOCK><<<grid, blk, smem, stream>>>(
        q, k, v, log_g, beta, state, out, B, H, KD, VD);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_gdr_decode_step] kernel error: %s\n",
                hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}

int ck_gdr_prefill_sequential_f32(
    const float* q, const float* k, const float* v,
    const float* log_g, const float* beta,
    float* state, float* out,
    int B, int SL, int H, int KD, int VD,
    hipStream_t stream)
{
    if (!q || !k || !v || !log_g || !beta || !state || !out) return CK_ERR_ARG;
    if (B <= 0 || SL <= 0 || H <= 0 || KD <= 0 || VD <= 0)  return CK_ERR_ARG;

    size_t smem = (size_t)KD * VD * sizeof(float);
    if (smem > 65536) {
        fprintf(stderr, "[ck_gdr_prefill_sequential] state %zu B > 64 KB LDS limit\n",
                smem);
        return CK_ERR_ARG;
    }

    const int BLOCK = 64;
    dim3 grid((unsigned)(B * H));
    dim3 blk(BLOCK);

    gdr_sequential_prefill_kernel<BLOCK><<<grid, blk, smem, stream>>>(
        q, k, v, log_g, beta, state, out, B, SL, H, KD, VD);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[ck_gdr_prefill_sequential] kernel error: %s\n",
                hipGetErrorString(err));
        return CK_ERR_HIP;
    }
    return CK_SUCCESS;
}

} /* extern "C" */
