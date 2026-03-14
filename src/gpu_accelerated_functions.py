"""
GPU-Accelerated Compressed Inference via Kernel-Fused Matmul

Instead of decompressing the full weight matrix on CPU then running F.linear,
these kernels compute the matmul directly from compressed indices:

    out[tok, i] = Σ_j  x[tok, j] * codebook[ unpack_index(packed, i*K + j, bits) ]

No weight materialization. No CPU↔GPU round-trips for weight data.
Codebook (~30 KB) fits in L2 cache on all modern GPUs.

Kernels are JIT-compiled by nvcc on first import and cached by PyTorch.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# CUDA kernel source
# ──────────────────────────────────────────────────────────────────────────────

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

// ── Bit-unpacking (matches _unpack_bits_np exactly) ─────────────────────────
// Reads 3 bytes from `packed` at the byte position corresponding to
// logical_idx * bits, then extracts `bits` bits starting at bit_shift.
// The 4-byte padding added by the Python side ensures byte_pos+2 is safe.
__device__ __forceinline__ uint32_t unpack_idx(
    const uint8_t* __restrict__ packed,
    int64_t logical_idx,
    int bits)
{
    int64_t bit_pos  = logical_idx * (int64_t)bits;
    int64_t byte_pos = bit_pos >> 3;
    int     bit_shft = (int)(bit_pos & 7);
    uint32_t mask    = (1u << bits) - 1u;

    uint32_t w = (uint32_t)packed[byte_pos]
               | ((uint32_t)packed[byte_pos + 1] << 8)
               | ((uint32_t)packed[byte_pos + 2] << 16)
               | ((uint32_t)packed[byte_pos + 3] << 24);
    return (w >> bit_shft) & mask;
}

// ── Linear kernel ────────────────────────────────────────────────────────────
// Grid(T*M), Block(BLOCK)  — 1D grid avoids the 65535 limit on grid.y (Pascal).
// out[tok, ofeat] = Σ_j  x[tok, j] * codebook[ idx(ofeat*K + j) ]
// Each block reduces over the K (input-feature) dimension.
template<int BLOCK>
__global__ void compressed_linear_kernel(
    const float* __restrict__ x,         // [T, K]  float32
    const uint8_t* __restrict__ packed,  // bit-packed weight indices, logical [M, K]
    const float* __restrict__ codebook,  // [C]  float32
    float*       __restrict__ out,       // [T, M]  float32
    int T, int M, int K, int C, int bits)
{
    int idx   = (int)blockIdx.x;
    int tok   = idx / M;
    int ofeat = idx - tok * M;   // idx % M
    int tid   = (int)threadIdx.x;

    __shared__ float sh[BLOCK];

    const float* xt = x + (int64_t)tok * K;
    float acc = 0.f;

    // Each thread handles K/BLOCK elements; accumulate partial dot-product.
    for (int j = tid; j < K; j += BLOCK) {
        int64_t li  = (int64_t)ofeat * K + j;
        uint32_t ci = unpack_idx(packed, li, bits);
        if ((int)ci >= C) ci = (uint32_t)(C - 1);
        acc += xt[j] * codebook[ci];
    }
    sh[tid] = acc;
    __syncthreads();

    // Parallel reduction in shared memory.
    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[(int64_t)tok * M + ofeat] = sh[0];
}

// ── Embedding kernel ─────────────────────────────────────────────────────────
// Grid(T, ceil(H/BLOCK)), Block(BLOCK)
// out[tok, h] = codebook[ idx(token_id * H + h) ]
// One thread per output element – embarrassingly parallel.
template<int BLOCK>
__global__ void compressed_embedding_kernel(
    const int32_t* __restrict__ token_ids,  // [T]
    const uint8_t* __restrict__ packed,     // bit-packed embedding indices
    const float*   __restrict__ codebook,   // [C]
    float*         __restrict__ out,        // [T, H]
    int T, int H, int C, int bits)
{
    int tok = (int)blockIdx.x;
    int h   = (int)blockIdx.y * BLOCK + (int)threadIdx.x;
    if (h >= H) return;

    int32_t token_id = token_ids[tok];
    int64_t li  = (int64_t)token_id * H + h;
    uint32_t ci = unpack_idx(packed, li, bits);
    if ((int)ci >= C) ci = (uint32_t)(C - 1);
    out[(int64_t)tok * H + h] = codebook[ci];
}

// ── Python-facing entry points ───────────────────────────────────────────────

torch::Tensor fused_compressed_linear(
    torch::Tensor x,         // [*, K]  any float dtype
    torch::Tensor packed,    // [N_bytes] uint8, with 4 padding bytes at end
    torch::Tensor codebook,  // [C] float32
    int64_t M,
    int64_t K,
    int bits)
{
    auto orig = x.sizes().vec();
    int T  = (int)(x.numel() / K);
    int C  = (int)codebook.size(0);

    x        = x.reshape({T, K}).to(torch::kFloat32).contiguous();
    packed   = packed.contiguous();
    codebook = codebook.to(torch::kFloat32).contiguous();

    auto out = torch::zeros(
        {T, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    const int BLOCK = 256;
    // Use 1D grid (T*M) to avoid the 65535 limit on grid.y for large M (e.g. lm_head M=248320).
    // grid.x supports up to 2^31-1 on all CUDA-capable devices.
    dim3 grid((unsigned)(T * M));
    dim3 blk(BLOCK);

    compressed_linear_kernel<BLOCK><<<grid, blk>>>(
        x.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        codebook.data_ptr<float>(),
        out.data_ptr<float>(),
        T, (int)M, (int)K, C, bits);

    std::vector<int64_t> out_sz(orig.begin(), orig.end() - 1);
    out_sz.push_back(M);
    return out.reshape(out_sz);
}

torch::Tensor fused_compressed_embedding(
    torch::Tensor token_ids,  // [*] int64 or int32
    torch::Tensor packed,     // [N_bytes] uint8, with 4 padding bytes at end
    torch::Tensor codebook,   // [C] float32
    int64_t hidden,
    int bits)
{
    auto orig = token_ids.sizes().vec();
    int T = (int)token_ids.numel();
    int H = (int)hidden;
    int C = (int)codebook.size(0);

    auto ids_flat = token_ids.reshape({T}).to(torch::kInt32).contiguous();
    packed   = packed.contiguous();
    codebook = codebook.to(torch::kFloat32).contiguous();

    auto out = torch::zeros(
        {T, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(codebook.device()));

    const int BLOCK = 128;
    dim3 grid((unsigned)T, (unsigned)((H + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    compressed_embedding_kernel<BLOCK><<<grid, blk>>>(
        ids_flat.data_ptr<int32_t>(),
        packed.data_ptr<uint8_t>(),
        codebook.data_ptr<float>(),
        out.data_ptr<float>(),
        T, H, C, bits);

    std::vector<int64_t> out_sz(orig.begin(), orig.end());
    out_sz.push_back(H);
    return out.reshape(out_sz);
}
"""

_CPP_SRC = r"""
torch::Tensor fused_compressed_linear(
    torch::Tensor x,
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t M,
    int64_t K,
    int bits);

torch::Tensor fused_compressed_embedding(
    torch::Tensor token_ids,
    torch::Tensor packed,
    torch::Tensor codebook,
    int64_t hidden,
    int bits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compressed_linear",    &fused_compressed_linear,
          "Kernel-fused compressed linear forward (CUDA)");
    m.def("fused_compressed_embedding", &fused_compressed_embedding,
          "Kernel-fused compressed embedding forward (CUDA)");
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Load / compile the extension
# ──────────────────────────────────────────────────────────────────────────────

import os as _os

_ext = None

def _load_extension():
    global _ext
    if _ext is not None:
        return _ext

    if not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load_inline
        print("  [compressed_kernel] Compiling CUDA extension (first run ~30-60s)...")
        _ext = load_inline(
            name="compressed_matmul_v3",   # bumped: removed tiled kernel
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        print("  [compressed_kernel] CUDA extension ready.")
    except Exception as e:
        print(f"  [compressed_kernel] Failed to compile CUDA extension: {e}")
        _ext = None

    return _ext


def _pad_packed(indices_np: np.ndarray) -> torch.Tensor:
    """Append 4 zero-bytes so the kernel can safely read 3-byte windows at the end."""
    padded = np.zeros(len(indices_np) + 4, dtype=np.uint8)
    padded[:len(indices_np)] = indices_np
    return torch.from_numpy(padded)


# ──────────────────────────────────────────────────────────────────────────────
# Public wrappers used by compressed_modules.py
# ──────────────────────────────────────────────────────────────────────────────

class GPUAcceleratedLinear:
    """
    Drop-in replacement for the CPU-decode path in AdaptiveCodebookLinear.

    Stores packed indices on GPU, runs fused compressed matmul at call time.
    Never materializes the full weight matrix.
    """

    def __init__(self, name: str, indices_np: np.ndarray,
                 codebook: torch.Tensor, shape: tuple, bits: int):
        self.name    = name
        self.shape   = tuple(int(s) for s in shape)   # (out_features, in_features)
        self.bits    = bits
        self.M, self.K = self.shape[0], self.shape[1]

        ext = _load_extension()
        self._ext = ext

        device = codebook.device if codebook.is_cuda else torch.device('cuda')

        # Store codebook in float32 on GPU
        self.codebook = codebook.to(device=device, dtype=torch.float32)

        # Pad and store packed indices on GPU
        packed_padded = _pad_packed(
            np.ascontiguousarray(indices_np) if isinstance(indices_np, np.ndarray)
            else indices_np.cpu().numpy()
        )
        self.packed = packed_padded.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [..., K]  (any float dtype, usually bfloat16)
        returns: [..., M]  same dtype as x
        """
        in_dtype = x.dtype
        device   = x.device

        if self._ext is None:
            raise RuntimeError("CUDA extension not available for GPUAcceleratedLinear")

        out = self._ext.fused_compressed_linear(
            x.to(device), self.packed, self.codebook, self.M, self.K, self.bits
        )
        # Cast back to model dtype (e.g., bfloat16)
        return out.to(dtype=in_dtype)


class GPUAcceleratedEmbedding:
    """
    Drop-in replacement for the CPU-decode path in AdaptiveCodebookEmbedding.

    Looks up embedding rows directly from compressed representation on GPU.
    """

    def __init__(self, name: str, indices_np: np.ndarray,
                 codebook: torch.Tensor, shape: tuple, bits: int):
        self.name   = name
        self.shape  = tuple(int(s) for s in shape)  # (vocab, hidden)
        self.bits   = bits
        self.vocab, self.hidden = self.shape[0], self.shape[1]

        ext = _load_extension()
        self._ext = ext

        device = codebook.device if codebook.is_cuda else torch.device('cuda')

        self.codebook = codebook.to(device=device, dtype=torch.float32)

        packed_padded = _pad_packed(
            np.ascontiguousarray(indices_np) if isinstance(indices_np, np.ndarray)
            else indices_np.cpu().numpy()
        )
        self.packed = packed_padded.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [...] int64 token IDs
        returns: [..., hidden]  float32 (cast to model dtype by caller if needed)
        """
        if self._ext is None:
            raise RuntimeError("CUDA extension not available for GPUAcceleratedEmbedding")

        device = self.codebook.device
        out = self._ext.fused_compressed_embedding(
            x.to(device),
            self.packed,
            self.codebook,
            self.hidden,
            self.bits,
        )
        return out
