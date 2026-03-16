/*
 * test_kernel.cpp
 *
 * Sanity-check for compressed_kernel.so.
 * Builds a tiny compressed layer with known weights, runs forward pass,
 * and checks output against a CPU reference.
 *
 * Build & run:  make test
 */

#include "compressed_kernel.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ── Helpers ──────────────────────────────────────────────────────────── */

#define CHECK_HIP(call) do {                                        \
    hipError_t _e = (call);                                         \
    if (_e != hipSuccess) {                                         \
        fprintf(stderr, "HIP error %s:%d — %s\n",                  \
                __FILE__, __LINE__, hipGetErrorString(_e));         \
        exit(1);                                                    \
    }                                                               \
} while(0)

/* Pack an array of `n` indices (each < 2^bits) into a byte array.
 * Matches the Python bitpack.pack_any_bits() layout exactly. */
static void pack_bits(const uint16_t* idx, int n, int bits, uint8_t* out_bytes, int out_len)
{
    memset(out_bytes, 0, out_len);
    for (int i = 0; i < n; i++) {
        int64_t bit_pos  = (int64_t)i * bits;
        int64_t byte_pos = bit_pos >> 3;
        int     shift    = (int)(bit_pos & 7);
        uint32_t val = (uint32_t)idx[i] & ((1u << bits) - 1u);
        /* write up to 3 bytes */
        out_bytes[byte_pos]     |= (uint8_t)(val << shift);
        if (shift + bits > 8)   out_bytes[byte_pos + 1] |= (uint8_t)(val >> (8  - shift));
        if (shift + bits > 16)  out_bytes[byte_pos + 2] |= (uint8_t)(val >> (16 - shift));
    }
}

/* CPU reference: out[t,m] = sum_k x[t,k] * codebook[idx[m,k]] */
static void cpu_linear(const float* x, const uint16_t* idx,
                        const float* cb, float* out,
                        int T, int M, int K)
{
    for (int t = 0; t < T; t++)
        for (int m = 0; m < M; m++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += x[t * K + k] * cb[idx[m * K + k]];
            out[t * M + m] = acc;
        }
}

static float cosine(const float* a, const float* b, int n)
{
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

static float max_abs_diff(const float* a, const float* b, int n)
{
    float m = 0.f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ── Test cases ───────────────────────────────────────────────────────── */

static int test_linear_f32(int T, int M, int K, int bits, int C)
{
    printf("  linear f32  T=%d M=%d K=%d bits=%d C=%d ... ", T, M, K, bits, C);
    fflush(stdout);

    int n_idx = M * K;
    int packed_bytes = (n_idx * bits + 7) / 8 + 4;  /* +4 pad */

    uint16_t* idx      = (uint16_t*)malloc(n_idx    * sizeof(uint16_t));
    float*    codebook = (float*)   malloc(C         * sizeof(float));
    float*    x_host   = (float*)   malloc(T * K     * sizeof(float));
    float*    ref      = (float*)   malloc(T * M     * sizeof(float));
    float*    gpu_out  = (float*)   malloc(T * M     * sizeof(float));
    uint8_t*  packed_h = (uint8_t*) calloc(packed_bytes, 1);

    /* deterministic fill */
    for (int i = 0; i < C;     i++) codebook[i] = (i - C/2) * 0.01f;
    for (int i = 0; i < n_idx; i++) idx[i]      = (uint16_t)(i % C);
    for (int i = 0; i < T * K; i++) x_host[i]   = (i % 7 - 3) * 0.1f;

    pack_bits(idx, n_idx, bits, packed_h, packed_bytes);
    cpu_linear(x_host, idx, codebook, ref, T, M, K);

    /* upload */
    float*    d_x  = NULL;
    float*    d_cb = ck_upload_codebook(codebook, C);
    uint8_t*  d_pk = ck_upload_packed(packed_h, packed_bytes);
    float*    d_out = NULL;

    CHECK_HIP(hipMalloc(&d_x,   T * K * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_out, T * M * sizeof(float)));
    CHECK_HIP(hipMemset(d_out, 0, T * M * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_x, x_host, T * K * sizeof(float), hipMemcpyHostToDevice));

    int rc = ck_linear_f32(d_x, d_pk, d_cb, d_out, T, M, K, C, bits, 0);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(gpu_out, d_out, T * M * sizeof(float), hipMemcpyDeviceToHost));

    (void)hipFree(d_x); (void)hipFree(d_cb); (void)hipFree(d_pk); (void)hipFree(d_out);

    float cos = cosine(ref, gpu_out, T * M);
    float mad = max_abs_diff(ref, gpu_out, T * M);
    int pass  = (rc == CK_SUCCESS) && (cos > 0.999f) && (mad < 1e-3f);
    printf("%s  cos=%.6f  max_diff=%.2e\n", pass ? "PASS" : "FAIL", cos, mad);

    free(idx); free(codebook); free(x_host); free(ref); free(gpu_out); free(packed_h);
    return pass;
}

static int test_linear_bf16(int T, int M, int K, int bits, int C)
{
    printf("  linear bf16 T=%d M=%d K=%d bits=%d C=%d ... ", T, M, K, bits, C);
    fflush(stdout);

    int n_idx = M * K;
    int packed_bytes = (n_idx * bits + 7) / 8 + 4;

    uint16_t*           idx      = (uint16_t*)           malloc(n_idx    * sizeof(uint16_t));
    float*              codebook = (float*)               malloc(C         * sizeof(float));
    float*              x_f32    = (float*)               malloc(T * K     * sizeof(float));
    hip_bfloat16*     x_bf16   = (hip_bfloat16*)      malloc(T * K     * sizeof(hip_bfloat16));
    float*              ref      = (float*)               malloc(T * M     * sizeof(float));
    hip_bfloat16*     gpu_bf16 = (hip_bfloat16*)      malloc(T * M     * sizeof(hip_bfloat16));
    float*              gpu_f32  = (float*)               malloc(T * M     * sizeof(float));
    uint8_t*            packed_h = (uint8_t*)             calloc(packed_bytes, 1);

    for (int i = 0; i < C;     i++) codebook[i] = (i - C/2) * 0.01f;
    for (int i = 0; i < n_idx; i++) idx[i]      = (uint16_t)(i % C);
    for (int i = 0; i < T * K; i++) {
        x_f32[i]  = (i % 7 - 3) * 0.1f;
        x_bf16[i] = (hip_bfloat16)x_f32[i];
    }

    pack_bits(idx, n_idx, bits, packed_h, packed_bytes);
    cpu_linear(x_f32, idx, codebook, ref, T, M, K);

    float*           d_cb  = ck_upload_codebook(codebook, C);
    uint8_t*         d_pk  = ck_upload_packed(packed_h, packed_bytes);
    hip_bfloat16*  d_x   = NULL;
    hip_bfloat16*  d_out = NULL;

    CHECK_HIP(hipMalloc(&d_x,   T * K * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_out, T * M * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMemset(d_out, 0, T * M * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMemcpy(d_x, x_bf16, T * K * sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    int rc = ck_linear_bf16(d_x, d_pk, d_cb, d_out, T, M, K, C, bits, 0);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(gpu_bf16, d_out, T * M * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));

    (void)hipFree(d_x); (void)hipFree(d_cb); (void)hipFree(d_pk); (void)hipFree(d_out);

    for (int i = 0; i < T * M; i++) gpu_f32[i] = (float)gpu_bf16[i];

    /* bfloat16 has 7 mantissa bits (~0.78% relative error).
     * Check cosine similarity only — absolute max_diff is not meaningful
     * when codebook values span a wide range (values up to ~C/2 * scale). */
    float cos = cosine(ref, gpu_f32, T * M);
    float mad = max_abs_diff(ref, gpu_f32, T * M);
    int pass  = (rc == CK_SUCCESS) && (cos > 0.9999f);
    printf("%s  cos=%.6f  max_diff=%.2e\n", pass ? "PASS" : "FAIL", cos, mad);

    free(idx); free(codebook); free(x_f32); free(x_bf16); free(ref);
    free(gpu_bf16); free(gpu_f32); free(packed_h);
    return pass;
}

static int test_embedding(int T, int vocab, int H, int bits, int C)
{
    printf("  embedding   T=%d vocab=%d H=%d bits=%d C=%d ... ", T, vocab, H, bits, C);
    fflush(stdout);

    int n_idx = vocab * H;
    int packed_bytes = (n_idx * bits + 7) / 8 + 4;

    uint16_t* idx      = (uint16_t*)malloc(n_idx    * sizeof(uint16_t));
    float*    codebook = (float*)   malloc(C         * sizeof(float));
    int32_t*  tids_h   = (int32_t*) malloc(T         * sizeof(int32_t));
    float*    ref      = (float*)   malloc(T * H     * sizeof(float));
    float*    gpu_out  = (float*)   malloc(T * H     * sizeof(float));
    uint8_t*  packed_h = (uint8_t*) calloc(packed_bytes, 1);

    for (int i = 0; i < C;     i++) codebook[i] = (i - C/2) * 0.02f;
    for (int i = 0; i < n_idx; i++) idx[i]      = (uint16_t)(i % C);
    for (int t = 0; t < T;     t++) tids_h[t]   = t % vocab;

    pack_bits(idx, n_idx, bits, packed_h, packed_bytes);

    /* CPU reference */
    for (int t = 0; t < T; t++) {
        int32_t tid = tids_h[t];
        for (int h = 0; h < H; h++) {
            int ci = idx[(int64_t)tid * H + h] % C;
            ref[t * H + h] = codebook[ci];
        }
    }

    float*   d_cb   = ck_upload_codebook(codebook, C);
    uint8_t* d_pk   = ck_upload_packed(packed_h, packed_bytes);
    int32_t* d_tids = NULL;
    float*   d_out  = NULL;

    CHECK_HIP(hipMalloc(&d_tids, T * sizeof(int32_t)));
    CHECK_HIP(hipMalloc(&d_out,  T * H * sizeof(float)));
    CHECK_HIP(hipMemset(d_out, 0, T * H * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_tids, tids_h, T * sizeof(int32_t), hipMemcpyHostToDevice));

    int rc = ck_embedding_f32(d_tids, d_pk, d_cb, d_out, T, H, C, bits, 0);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(gpu_out, d_out, T * H * sizeof(float), hipMemcpyDeviceToHost));

    (void)hipFree(d_tids); (void)hipFree(d_cb); (void)hipFree(d_pk); (void)hipFree(d_out);

    float cos = cosine(ref, gpu_out, T * H);
    float mad = max_abs_diff(ref, gpu_out, T * H);
    int pass  = (rc == CK_SUCCESS) && (cos > 0.999f) && (mad < 1e-5f);
    printf("%s  cos=%.6f  max_diff=%.2e\n", pass ? "PASS" : "FAIL", cos, mad);

    free(idx); free(codebook); free(tids_h); free(ref); free(gpu_out); free(packed_h);
    return pass;
}

/* ── Raw (uncompressed) test cases ────────────────────────────────────── */

static int test_raw_linear_f32(int T, int M, int K)
{
    printf("  raw linear f32  T=%d M=%d K=%d ... ", T, M, K);
    fflush(stdout);

    float* weight_h = (float*)malloc((size_t)M * K * sizeof(float));
    float* x_h      = (float*)malloc((size_t)T * K * sizeof(float));
    float* ref      = (float*)calloc((size_t)T * M, sizeof(float));
    float* gpu_out  = (float*)malloc((size_t)T * M * sizeof(float));

    for (int i = 0; i < M * K; i++) weight_h[i] = (i % 7 - 3) * 0.1f;
    for (int i = 0; i < T * K; i++) x_h[i]      = (i % 5 - 2) * 0.2f;

    /* CPU reference */
    for (int t = 0; t < T; t++)
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                ref[t * M + m] += x_h[t * K + k] * weight_h[m * K + k];

    float* d_x   = ck_upload_weights_f32(x_h,      T, K);
    float* d_w   = ck_upload_weights_f32(weight_h,  M, K);
    float* d_out = NULL;
    CHECK_HIP(hipMalloc(&d_out, (size_t)T * M * sizeof(float)));
    CHECK_HIP(hipMemset(d_out, 0, (size_t)T * M * sizeof(float)));

    int rc = ck_linear_raw_f32(d_x, d_w, d_out, T, M, K, 0);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(gpu_out, d_out, (size_t)T * M * sizeof(float), hipMemcpyDeviceToHost));

    (void)hipFree(d_x); (void)hipFree(d_w); (void)hipFree(d_out);

    float c = cosine(ref, gpu_out, T * M);
    float d = max_abs_diff(ref, gpu_out, T * M);
    int pass = (rc == CK_SUCCESS) && (c > 0.9999f) && (d < 1e-3f);
    printf("%s  cos=%.6f  max_diff=%.2e\n", pass ? "PASS" : "FAIL", c, d);

    free(weight_h); free(x_h); free(ref); free(gpu_out);
    return pass;
}

static int test_raw_linear_bf16(int T, int M, int K)
{
    printf("  raw linear bf16 T=%d M=%d K=%d ... ", T, M, K);
    fflush(stdout);

    float*        weight_f32 = (float*)        malloc((size_t)M * K * sizeof(float));
    float*        x_f32      = (float*)        malloc((size_t)T * K * sizeof(float));
    hip_bfloat16* weight_bf  = (hip_bfloat16*) malloc((size_t)M * K * sizeof(hip_bfloat16));
    hip_bfloat16* x_bf       = (hip_bfloat16*) malloc((size_t)T * K * sizeof(hip_bfloat16));
    float*        ref        = (float*)        calloc((size_t)T * M, sizeof(float));
    hip_bfloat16* gpu_bf     = (hip_bfloat16*) malloc((size_t)T * M * sizeof(hip_bfloat16));
    float*        gpu_f32    = (float*)        malloc((size_t)T * M * sizeof(float));

    for (int i = 0; i < M * K; i++) { weight_f32[i] = (i % 7 - 3) * 0.1f; weight_bf[i] = (hip_bfloat16)weight_f32[i]; }
    for (int i = 0; i < T * K; i++) { x_f32[i]      = (i % 5 - 2) * 0.2f; x_bf[i]      = (hip_bfloat16)x_f32[i]; }

    for (int t = 0; t < T; t++)
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K; k++)
                ref[t * M + m] += x_f32[t * K + k] * weight_f32[m * K + k];

    hip_bfloat16* d_x   = ck_upload_weights_bf16(x_bf,     T, K);
    hip_bfloat16* d_w   = ck_upload_weights_bf16(weight_bf, M, K);
    hip_bfloat16* d_out = NULL;
    CHECK_HIP(hipMalloc(&d_out, (size_t)T * M * sizeof(hip_bfloat16)));
    CHECK_HIP(hipMemset(d_out, 0, (size_t)T * M * sizeof(hip_bfloat16)));

    int rc = ck_linear_raw_bf16(d_x, d_w, d_out, T, M, K, 0);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(gpu_bf, d_out, (size_t)T * M * sizeof(hip_bfloat16), hipMemcpyDeviceToHost));

    (void)hipFree(d_x); (void)hipFree(d_w); (void)hipFree(d_out);

    for (int i = 0; i < T * M; i++) gpu_f32[i] = (float)gpu_bf[i];

    float c = cosine(ref, gpu_f32, T * M);
    float d = max_abs_diff(ref, gpu_f32, T * M);
    int pass = (rc == CK_SUCCESS) && (c > 0.9999f);
    printf("%s  cos=%.6f  max_diff=%.2e\n", pass ? "PASS" : "FAIL", c, d);

    free(weight_f32); free(x_f32); free(weight_bf); free(x_bf);
    free(ref); free(gpu_bf); free(gpu_f32);
    return pass;
}

static int test_raw_embedding_f32(int T, int vocab, int H)
{
    printf("  raw embedding   T=%d vocab=%d H=%d ... ", T, vocab, H);
    fflush(stdout);

    float*   weight_h = (float*)  malloc((size_t)vocab * H * sizeof(float));
    int32_t* tids_h   = (int32_t*)malloc((size_t)T      * sizeof(int32_t));
    float*   ref      = (float*)  malloc((size_t)T * H  * sizeof(float));
    float*   gpu_out  = (float*)  malloc((size_t)T * H  * sizeof(float));

    for (int i = 0; i < vocab * H; i++) weight_h[i] = (i % 11 - 5) * 0.05f;
    for (int t = 0; t < T; t++)         tids_h[t]   = t % vocab;

    for (int t = 0; t < T; t++)
        memcpy(ref + t * H, weight_h + (size_t)tids_h[t] * H, H * sizeof(float));

    float*   d_w    = ck_upload_weights_f32(weight_h, vocab, H);
    int32_t* d_tids = NULL;
    float*   d_out  = NULL;
    CHECK_HIP(hipMalloc(&d_tids, T * sizeof(int32_t)));
    CHECK_HIP(hipMalloc(&d_out,  (size_t)T * H * sizeof(float)));
    CHECK_HIP(hipMemset(d_out, 0, (size_t)T * H * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_tids, tids_h, T * sizeof(int32_t), hipMemcpyHostToDevice));

    int rc = ck_embedding_raw_f32(d_tids, d_w, d_out, T, H, 0);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(gpu_out, d_out, (size_t)T * H * sizeof(float), hipMemcpyDeviceToHost));

    (void)hipFree(d_tids); (void)hipFree(d_w); (void)hipFree(d_out);

    float c = cosine(ref, gpu_out, T * H);
    float d = max_abs_diff(ref, gpu_out, T * H);
    int pass = (rc == CK_SUCCESS) && (c > 0.9999f) && (d < 1e-6f);
    printf("%s  cos=%.6f  max_diff=%.2e\n", pass ? "PASS" : "FAIL", c, d);

    free(weight_h); free(tids_h); free(ref); free(gpu_out);
    return pass;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void)
{
    /* Print device info */
    int dev_count = 0;
    (void)hipGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "No HIP devices found.\n");
        return 1;
    }
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);
    printf("Device: %s\n\n", prop.name);

    int pass = 1;

    printf("Linear f32 (8-bit indices):\n");
    pass &= test_linear_f32(1,   64,  32,  8, 256);   /* autoregressive T=1   */
    pass &= test_linear_f32(4,   64,  32,  8, 256);   /* small batch          */
    pass &= test_linear_f32(1,  512, 256, 13, 8192);  /* 13-bit, larger layer */

    printf("\nLinear bf16 (13-bit indices):\n");
    pass &= test_linear_bf16(1,  64,  32, 13, 8192);
    pass &= test_linear_bf16(4, 256, 128, 13, 8192);

    printf("\nEmbedding f32:\n");
    pass &= test_embedding(1,  256,  64,  8, 256);
    pass &= test_embedding(4, 1024, 128, 13, 8192);

    printf("\nRaw (uncompressed) linear f32:\n");
    pass &= test_raw_linear_f32(1,  64,  32);
    pass &= test_raw_linear_f32(4, 128,  64);
    pass &= test_raw_linear_f32(1, 512, 256);

    printf("\nRaw (uncompressed) linear bf16:\n");
    pass &= test_raw_linear_bf16(1,  64,  32);
    pass &= test_raw_linear_bf16(4, 128,  64);

    printf("\nRaw (uncompressed) embedding f32:\n");
    pass &= test_raw_embedding_f32(1, 256, 64);
    pass &= test_raw_embedding_f32(4, 512, 128);

    printf("\n%s\n", pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return pass ? 0 : 1;
}
