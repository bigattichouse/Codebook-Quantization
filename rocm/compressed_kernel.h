/*
 * compressed_kernel.h
 *
 * Public C API for standalone HIP compressed-matmul and embedding kernels.
 * No PyTorch or other framework dependency — link against libcompressed_kernel.so.
 *
 * Precision model:
 *   - Packed indices  : uint8_t  (bit-packed, 1–16 bits per index)
 *   - Codebook        : float32  (small, ~30 KB, full precision)
 *   - Input / output  : float32 or bfloat16 (choose at call time via the _bf16 variants)
 *   - Accumulation    : always float32 internally
 *
 * All device pointers must be allocated with hipMalloc (or equivalent).
 * The helper functions ck_upload_* allocate device memory and copy from host;
 * the caller is responsible for hipFree()-ing the returned pointers.
 *
 * Build:
 *   cd proofofconcept/rocm && make
 *
 * Link:
 *   -L<path> -lcompressed_kernel -lhip
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes ─────────────────────────────────────────────────────────── */

#define CK_SUCCESS   0  /* no error                   */
#define CK_ERR_HIP   1  /* HIP runtime error          */
#define CK_ERR_ARG   2  /* invalid argument (NULL ptr, bad dims, …) */

const char* ck_error_string(int err);


/* ── Compressed linear — float32 I/O ─────────────────────────────────────
 *
 * Computes:
 *   out[t, m] = sum_k  x[t, k] * codebook[ unpack(packed, m*K+k, bits) ]
 *
 * All pointers: HIP device memory.
 * out must be zeroed by the caller (or freshly hipMalloc'd — zeroed by default).
 */
int ck_linear_f32(
    const float*    x,        /* [T, K]   float32 input activations      */
    const uint8_t*  packed,   /* bit-packed weight indices + 4 pad bytes  */
    const float*    codebook, /* [C]      float32 codebook                */
    float*          out,      /* [T, M]   float32 output (caller allocs)  */
    int T, int M, int K, int C, int bits,
    hipStream_t stream        /* pass 0 for default stream                */
);

/* ── Compressed linear — bfloat16 I/O ───────────────────────────────────
 *
 * Same computation as ck_linear_f32 but x and out use __hip_bfloat16.
 * Codebook stays float32. Accumulation is float32 internally.
 */
int ck_linear_bf16(
    const hip_bfloat16* x,        /* [T, K]  bfloat16 input             */
    const uint8_t*      packed,   /* bit-packed weight indices + 4 pad  */
    const float*        codebook, /* [C]     float32 codebook           */
    hip_bfloat16*       out,      /* [T, M]  bfloat16 output            */
    int T, int M, int K, int C, int bits,
    hipStream_t stream
);


/* ── Compressed embedding — float32 output ───────────────────────────────
 *
 * Computes:
 *   out[t, h] = codebook[ unpack(packed, token_ids[t]*H + h, bits) ]
 *
 * All pointers: HIP device memory.
 */
int ck_embedding_f32(
    const int32_t*  token_ids, /* [T]      int32 token IDs               */
    const uint8_t*  packed,    /* bit-packed embedding indices + 4 pad   */
    const float*    codebook,  /* [C]      float32 codebook              */
    float*          out,       /* [T, H]   float32 output (caller allocs)*/
    int T, int H, int C, int bits,
    hipStream_t stream
);

/* ── Raw (uncompressed) linear — float32 ─────────────────────────────────
 *
 * Standard matrix multiply (no codebook, no bit-packing):
 *   out[t, m] = sum_k  x[t, k] * weight[m, k]
 *
 * weight is row-major [M, K].  All pointers: HIP device memory.
 * out must be zeroed by the caller.
 */
int ck_linear_raw_f32(
    const float* x,       /* [T, K]  float32 input             */
    const float* weight,  /* [M, K]  float32 weight (row-major)*/
    float*       out,     /* [T, M]  float32 output            */
    int T, int M, int K,
    hipStream_t stream
);

/* ── Raw (uncompressed) linear — bfloat16 ────────────────────────────────
 *
 * Same as ck_linear_raw_f32 but with bfloat16 I/O.
 * Accumulation is always float32 internally.
 */
int ck_linear_raw_bf16(
    const hip_bfloat16* x,       /* [T, K]  bfloat16 input    */
    const hip_bfloat16* weight,  /* [M, K]  bfloat16 weights  */
    hip_bfloat16*       out,     /* [T, M]  bfloat16 output   */
    int T, int M, int K,
    hipStream_t stream
);

/* ── Raw (uncompressed) embedding — float32 ──────────────────────────────
 *
 * Standard embedding lookup:
 *   out[t, h] = weight[token_ids[t], h]
 *
 * weight is the full embedding table [vocab, H].
 */
int ck_embedding_raw_f32(
    const int32_t* token_ids,  /* [T]        int32 token IDs      */
    const float*   weight,     /* [vocab, H] float32 embed table  */
    float*         out,        /* [T, H]     float32 output       */
    int T, int H,
    hipStream_t stream
);

/* ── Raw (uncompressed) embedding — bfloat16 ─────────────────────────────
 *
 * Same as ck_embedding_raw_f32 but with bfloat16 I/O.
 */
int ck_embedding_raw_bf16(
    const int32_t*      token_ids,  /* [T]        int32 token IDs      */
    const hip_bfloat16* weight,     /* [vocab, H] bfloat16 embed table */
    hip_bfloat16*       out,        /* [T, H]     bfloat16 output      */
    int T, int H,
    hipStream_t stream
);

/* ── Convenience upload helpers ──────────────────────────────────────────
 *
 * Allocate device memory and copy from host.  Returns NULL on failure.
 * Caller must hipFree() the returned pointer when done.
 *
 * ck_upload_packed      : uploads the bit-packed index array (+ 4 pad bytes)
 * ck_upload_codebook    : uploads the float32 codebook
 * ck_upload_weights_f32 : uploads a plain float32 weight matrix [rows × cols]
 * ck_upload_weights_bf16: uploads a bfloat16 weight matrix [rows × cols]
 */
uint8_t*        ck_upload_packed      (const uint8_t*        host_packed,   size_t nbytes);
float*          ck_upload_codebook    (const float*           host_codebook, int C);
float*          ck_upload_weights_f32 (const float*           host,          int rows, int cols);
hip_bfloat16*   ck_upload_weights_bf16(const hip_bfloat16*    host,          int rows, int cols);


/* ── SSM kernels ─────────────────────────────────────────────────────────────
 *
 * ck_causal_conv1d_prefill_f32
 *   Depthwise causal 1-D convolution + SiLU for prefill (full sequence).
 *   x, weight, out: float32 device pointers.
 *   x      [B, C, L]   input
 *   weight [C, ksz]    filter weights (ksz ≤ 4)
 *   out    [B, C, L]   SiLU(conv(x)) output
 *
 * ck_causal_conv1d_update_f32
 *   Single-step decode update.  Rolls conv_state left, inserts x_new, dots with w.
 *   x_new      [B, C]       new input token
 *   conv_state [B, C, ksz]  rolling buffer (updated in-place)
 *   weight     [C, ksz]
 *   out        [B, C]       SiLU(conv_output)
 *
 * ck_gdr_decode_step_f32
 *   One step of Gated DeltaNet recurrence (seq_len = 1).
 *   q, k   [B, H, KD]   query/key (caller applies L2-norm and scale 1/sqrt(KD))
 *   v      [B, H, VD]   value
 *   log_g  [B, H]       log-decay (negative; will be exp'd inside kernel)
 *   beta   [B, H]       beta (sigmoid output from model)
 *   state  [B, H, KD, VD]  recurrent state (updated in-place)
 *   out    [B, H, VD]   output activations
 *   Requires KD*VD*4 bytes of device LDS (≤ 64 KB).
 */
int ck_causal_conv1d_prefill_f32(
    const float* x, const float* weight, float* out,
    int B, int C, int L, int ksz,
    hipStream_t stream
);

int ck_causal_conv1d_update_f32(
    const float* x_new, float* conv_state, const float* weight, float* out,
    int B, int C, int ksz,
    hipStream_t stream
);

int ck_gdr_decode_step_f32(
    const float* q, const float* k, const float* v,
    const float* log_g, const float* beta,
    float* state, float* out,
    int B, int H, int KD, int VD,
    hipStream_t stream
);

/* ── GDR sequential prefill ──────────────────────────────────────────────────
 *
 * Processes an entire input sequence (SL tokens) in a single kernel launch.
 * The recurrent state [KD, VD] stays in GPU shared memory (LDS) across all
 * tokens — no global-memory round-trip between steps.
 *
 * q, k  : [B, SL, H, KD]   float32 (L2-normed + scaled by caller)
 * v     : [B, SL, H, VD]   float32
 * log_g : [B, SL, H]       float32  log-decay (exp'd inside kernel)
 * beta  : [B, SL, H]       float32
 * state : [B, H, KD, VD]   float32  (updated in-place; pass zeros for fresh start)
 * out   : [B, SL, H, VD]   float32  (caller allocates)
 *
 * Requires KD*VD*4 bytes of LDS per block (<= 64 KB).
 */
int ck_gdr_prefill_sequential_f32(
    const float* q, const float* k, const float* v,
    const float* log_g, const float* beta,
    float* state, float* out,
    int B, int SL, int H, int KD, int VD,
    hipStream_t stream
);

#ifdef __cplusplus
}
#endif
