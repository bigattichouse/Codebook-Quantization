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
# CUDA/HIP kernel source (compatible with both NVCC and hipcc)
# ──────────────────────────────────────────────────────────────────────────────

_CUDA_SRC = r"""
// ROCm/HIP compatibility headers
#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <torch/extension.h>
    #include <stdint.h>
    // Define CUDA functions as HIP equivalents for ROCm
    #define cudaSuccess hipSuccess
    #define cudaError_t hipError_t
    #define cudaGetErrorString hipGetErrorString
#else
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <stdint.h>
#endif

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

// ── Raw (uncompressed) kernels ───────────────────────────────────────────────
// Standard matmul / embedding lookup — no codebook, no bit-packing.

template<int BLOCK>
__global__ void raw_linear_kernel(
    const float* __restrict__ x,       // [T, K]
    const float* __restrict__ weight,  // [M, K] row-major
    float*       __restrict__ out,     // [T, M]
    int T, int M, int K)
{
    int idx   = (int)blockIdx.x;
    int tok   = idx / M;
    int ofeat = idx - tok * M;
    int tid   = (int)threadIdx.x;

    __shared__ float sh[BLOCK];

    const float* xt = x      + (int64_t)tok   * K;
    const float* wt = weight + (int64_t)ofeat * K;
    float acc = 0.f;

    for (int j = tid; j < K; j += BLOCK)
        acc += xt[j] * wt[j];

    sh[tid] = acc;
    __syncthreads();

    for (int s = BLOCK >> 1; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[(int64_t)tok * M + ofeat] = sh[0];
}

template<int BLOCK>
__global__ void raw_embedding_kernel(
    const int32_t* __restrict__ token_ids,
    const float*   __restrict__ weight,   // [vocab, H]
    float*         __restrict__ out,      // [T, H]
    int T, int H)
{
    int tok = (int)blockIdx.x;
    int h   = (int)blockIdx.y * BLOCK + (int)threadIdx.x;
    if (h >= H) return;

    int32_t tid = token_ids[tok];
    out[(int64_t)tok * H + h] = weight[(int64_t)tid * H + h];
}

torch::Tensor fused_raw_linear(
    torch::Tensor x,       // [*, K] any float
    torch::Tensor weight,  // [M, K] float32
    int64_t M,
    int64_t K)
{
    auto orig = x.sizes().vec();
    int T = (int)(x.numel() / K);

    x      = x.reshape({T, K}).to(torch::kFloat32).contiguous();
    weight = weight.to(torch::kFloat32).contiguous();

    auto out = torch::zeros(
        {T, M},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    const int BLOCK = 256;
    dim3 grid((unsigned)(T * M));
    dim3 blk(BLOCK);

    raw_linear_kernel<BLOCK><<<grid, blk>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(),
        T, (int)M, (int)K);

    std::vector<int64_t> out_sz(orig.begin(), orig.end() - 1);
    out_sz.push_back(M);
    return out.reshape(out_sz);
}

torch::Tensor fused_raw_embedding(
    torch::Tensor token_ids,  // [*] int64 or int32
    torch::Tensor weight,     // [vocab, H] float32
    int64_t hidden)
{
    auto orig = token_ids.sizes().vec();
    int T = (int)token_ids.numel();
    int H = (int)hidden;

    auto ids_flat = token_ids.reshape({T}).to(torch::kInt32).contiguous();
    weight = weight.to(torch::kFloat32).contiguous();

    auto out = torch::zeros(
        {T, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(weight.device()));

    const int BLOCK = 128;
    dim3 grid((unsigned)T, (unsigned)((H + BLOCK - 1) / BLOCK));
    dim3 blk(BLOCK);

    raw_embedding_kernel<BLOCK><<<grid, blk>>>(
        ids_flat.data_ptr<int32_t>(), weight.data_ptr<float>(), out.data_ptr<float>(),
        T, H);

    std::vector<int64_t> out_sz(orig.begin(), orig.end());
    out_sz.push_back(H);
    return out.reshape(out_sz);
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

torch::Tensor fused_raw_linear(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t M,
    int64_t K);

torch::Tensor fused_raw_embedding(
    torch::Tensor token_ids,
    torch::Tensor weight,
    int64_t hidden);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compressed_linear",    &fused_compressed_linear,
          "Kernel-fused compressed linear forward (CUDA/HIP)");
    m.def("fused_compressed_embedding", &fused_compressed_embedding,
          "Kernel-fused compressed embedding forward (CUDA/HIP)");
    m.def("fused_raw_linear",           &fused_raw_linear,
          "Kernel-fused raw (uncompressed) linear forward (CUDA/HIP)");
    m.def("fused_raw_embedding",        &fused_raw_embedding,
          "Kernel-fused raw (uncompressed) embedding forward (CUDA/HIP)");
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Load / compile the extension
# ──────────────────────────────────────────────────────────────────────────────

import ctypes as _ctypes
import os as _os
import re as _re
import subprocess as _subprocess

_ext = None

# Path to the standalone ROCm shared library (relative to this file).
_ROCM_SO = Path(__file__).parent.parent / 'rocm' / 'libcompressed_kernel.so'
_ROCM_DIR = Path(__file__).parent.parent / 'rocm'


def _find_rocm_path() -> str:
    """Return the ROCm installation root, or '/opt/rocm' as a last resort."""
    env = _os.environ.get('ROCM_PATH', '')
    if env and Path(env).is_dir():
        return env
    candidates = sorted(Path('/opt').glob('rocm*'), reverse=True)
    for c in candidates:
        if c.is_dir():
            return str(c)
    return '/opt/rocm'


def _detect_rocm_arch() -> str:
    """Detect the AMD GPU GCN architecture string (e.g. 'gfx906', 'gfx1100').

    Tries two methods:
    1. torch.cuda.get_device_properties — available in some PyTorch-ROCm builds.
    2. rocminfo subprocess — present on any ROCm installation.

    Returns the arch string on success, or '' if it cannot be determined
    (hipcc will then compile for whatever GPU is present at runtime).
    """
    # Method 1: PyTorch device properties
    try:
        props = torch.cuda.get_device_properties(0)
        if hasattr(props, 'gcnArchName'):
            # Format: "gfx906:sramecc+:xnack-"  →  "gfx906"
            arch = props.gcnArchName.split(':')[0].strip()
            if arch.startswith('gfx'):
                return arch
    except Exception:
        pass

    # Method 2: rocminfo
    try:
        result = _subprocess.run(
            ['rocminfo'], capture_output=True, text=True, timeout=10
        )
        match = _re.search(r'amdgcn-amd-amdhsa--(gfx\w+)', result.stdout)
        if match:
            return match.group(1).split(':')[0]
    except Exception:
        pass

    return ''


class _ROCmExtWrapper:
    """
    Drop-in replacement for the PyTorch load_inline extension on ROCm.

    Loads libcompressed_kernel.so directly via ctypes, bypassing PyTorch's
    JIT hipify pipeline which produces incorrect results on ROCm (the
    standalone .hip kernel gives cos=1.0 on the same hardware where the
    hipified CUDA kernel gives cos=0.0).

    Exposes the same interface as the load_inline extension so
    GPUAcceleratedLinear and GPUAcceleratedEmbedding need no changes.
    """

    def __init__(self, so_path: str):
        lib = _ctypes.CDLL(so_path)

        # ck_linear_f32(x, packed, codebook, out, T, M, K, C, bits, stream)
        lib.ck_linear_f32.restype  = _ctypes.c_int
        lib.ck_linear_f32.argtypes = [
            _ctypes.c_void_p,  # x        [T, K] float32
            _ctypes.c_void_p,  # packed   uint8
            _ctypes.c_void_p,  # codebook [C] float32
            _ctypes.c_void_p,  # out      [T, M] float32
            _ctypes.c_int,     # T
            _ctypes.c_int,     # M
            _ctypes.c_int,     # K
            _ctypes.c_int,     # C
            _ctypes.c_int,     # bits
            _ctypes.c_void_p,  # hipStream_t (NULL = default stream)
        ]

        # ck_embedding_f32(token_ids, packed, codebook, out, T, H, C, bits, stream)
        lib.ck_embedding_f32.restype  = _ctypes.c_int
        lib.ck_embedding_f32.argtypes = [
            _ctypes.c_void_p,  # token_ids [T] int32
            _ctypes.c_void_p,  # packed    uint8
            _ctypes.c_void_p,  # codebook  [C] float32
            _ctypes.c_void_p,  # out       [T, H] float32
            _ctypes.c_int,     # T
            _ctypes.c_int,     # H
            _ctypes.c_int,     # C
            _ctypes.c_int,     # bits
            _ctypes.c_void_p,  # hipStream_t (NULL = default stream)
        ]

        # ck_linear_raw_f32(x, weight, out, T, M, K, stream)
        lib.ck_linear_raw_f32.restype  = _ctypes.c_int
        lib.ck_linear_raw_f32.argtypes = [
            _ctypes.c_void_p,  # x      [T, K] float32
            _ctypes.c_void_p,  # weight [M, K] float32
            _ctypes.c_void_p,  # out    [T, M] float32
            _ctypes.c_int,     # T
            _ctypes.c_int,     # M
            _ctypes.c_int,     # K
            _ctypes.c_void_p,  # hipStream_t
        ]

        # ck_linear_raw_bf16(x, weight, out, T, M, K, stream)
        lib.ck_linear_raw_bf16.restype  = _ctypes.c_int
        lib.ck_linear_raw_bf16.argtypes = [
            _ctypes.c_void_p,  # x      [T, K] bfloat16
            _ctypes.c_void_p,  # weight [M, K] bfloat16
            _ctypes.c_void_p,  # out    [T, M] bfloat16
            _ctypes.c_int,     # T
            _ctypes.c_int,     # M
            _ctypes.c_int,     # K
            _ctypes.c_void_p,  # hipStream_t
        ]

        # ck_embedding_raw_f32(token_ids, weight, out, T, H, stream)
        lib.ck_embedding_raw_f32.restype  = _ctypes.c_int
        lib.ck_embedding_raw_f32.argtypes = [
            _ctypes.c_void_p,  # token_ids [T]        int32
            _ctypes.c_void_p,  # weight    [vocab, H] float32
            _ctypes.c_void_p,  # out       [T, H]     float32
            _ctypes.c_int,     # T
            _ctypes.c_int,     # H
            _ctypes.c_void_p,  # hipStream_t
        ]

        # ck_embedding_raw_bf16(token_ids, weight, out, T, H, stream)
        lib.ck_embedding_raw_bf16.restype  = _ctypes.c_int
        lib.ck_embedding_raw_bf16.argtypes = [
            _ctypes.c_void_p,  # token_ids [T]        int32
            _ctypes.c_void_p,  # weight    [vocab, H] bfloat16
            _ctypes.c_void_p,  # out       [T, H]     bfloat16
            _ctypes.c_int,     # T
            _ctypes.c_int,     # H
            _ctypes.c_void_p,  # hipStream_t
        ]

        self._lib = lib

    def _stream_ptr(self):
        """Return the current PyTorch CUDA stream as a ctypes void pointer."""
        try:
            return _ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        except Exception:
            return None  # fall back to default stream

    def fused_compressed_linear(self, x, packed, codebook, M, K, bits):
        """
        x        : [..., K] any float tensor on GPU
        packed   : [N_bytes] uint8 on GPU (with 4 pad bytes at end)
        codebook : [C] float32 on GPU
        Returns  : [..., M] float32 on GPU
        """
        orig_shape = x.shape
        T = int(x.numel() // K)
        C = int(codebook.size(0))

        x_f32  = x.reshape(T, K).to(torch.float32).contiguous()
        pk     = packed.contiguous()
        cb_f32 = codebook.to(torch.float32).contiguous()
        out    = torch.zeros(T, M, dtype=torch.float32, device=x.device)

        rc = self._lib.ck_linear_f32(
            _ctypes.c_void_p(x_f32.data_ptr()),
            _ctypes.c_void_p(pk.data_ptr()),
            _ctypes.c_void_p(cb_f32.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            T, M, K, C, bits,
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"ck_linear_f32 returned error {rc}")

        out_shape = list(orig_shape[:-1]) + [M]
        return out.reshape(out_shape)

    def fused_raw_linear(self, x, weight, M, K):
        """
        x      : [..., K] any float tensor on GPU
        weight : [M, K]   float32 or bfloat16 on GPU (chooses kernel automatically)
        Returns: [..., M] float32 on GPU
        """
        orig_shape = x.shape
        T = int(x.numel() // K)

        if weight.dtype == torch.bfloat16:
            # Keep in bfloat16 end-to-end to save VRAM on large uncompressed models
            x_bf16 = x.reshape(T, K).to(torch.bfloat16).contiguous()
            w_bf16 = weight.contiguous()
            out    = torch.zeros(T, M, dtype=torch.bfloat16, device=x.device)
            rc = self._lib.ck_linear_raw_bf16(
                _ctypes.c_void_p(x_bf16.data_ptr()),
                _ctypes.c_void_p(w_bf16.data_ptr()),
                _ctypes.c_void_p(out.data_ptr()),
                T, M, K,
                self._stream_ptr(),
            )
            if rc != 0:
                raise RuntimeError(f"ck_linear_raw_bf16 returned error {rc}")
            return out.reshape(list(orig_shape[:-1]) + [M])

        x_f32  = x.reshape(T, K).to(torch.float32).contiguous()
        w_f32  = weight.to(torch.float32).contiguous()
        out    = torch.zeros(T, M, dtype=torch.float32, device=x.device)

        rc = self._lib.ck_linear_raw_f32(
            _ctypes.c_void_p(x_f32.data_ptr()),
            _ctypes.c_void_p(w_f32.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            T, M, K,
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"ck_linear_raw_f32 returned error {rc}")

        return out.reshape(list(orig_shape[:-1]) + [M])

    def fused_raw_embedding(self, token_ids, weight, hidden):
        """
        token_ids : [...] int tensor on GPU
        weight    : [vocab, H] float32 or bfloat16 on GPU
        Returns   : [..., H] same dtype as weight on GPU
        """
        orig_shape = token_ids.shape
        T = int(token_ids.numel())
        H = int(hidden)

        ids = token_ids.reshape(T).to(torch.int32).contiguous()

        if weight.dtype == torch.bfloat16:
            w_bf16 = weight.contiguous()
            out = torch.zeros(T, H, dtype=torch.bfloat16, device=weight.device)
            rc = self._lib.ck_embedding_raw_bf16(
                _ctypes.c_void_p(ids.data_ptr()),
                _ctypes.c_void_p(w_bf16.data_ptr()),
                _ctypes.c_void_p(out.data_ptr()),
                T, H,
                self._stream_ptr(),
            )
            if rc != 0:
                raise RuntimeError(f"ck_embedding_raw_bf16 returned error {rc}")
            return out.reshape(list(orig_shape) + [H])

        w_f32 = weight.to(torch.float32).contiguous()
        out   = torch.zeros(T, H, dtype=torch.float32, device=weight.device)

        rc = self._lib.ck_embedding_raw_f32(
            _ctypes.c_void_p(ids.data_ptr()),
            _ctypes.c_void_p(w_f32.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            T, H,
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"ck_embedding_raw_f32 returned error {rc}")

        return out.reshape(list(orig_shape) + [H])

    def fused_compressed_embedding(self, token_ids, packed, codebook, hidden, bits):
        """
        token_ids : [...] int tensor on GPU
        packed    : [N_bytes] uint8 on GPU (with 4 pad bytes at end)
        codebook  : [C] float32 on GPU
        Returns   : [..., hidden] float32 on GPU
        """
        orig_shape = token_ids.shape
        T = int(token_ids.numel())
        H = int(hidden)
        C = int(codebook.size(0))

        ids  = token_ids.reshape(T).to(torch.int32).contiguous()
        pk   = packed.contiguous()
        cb   = codebook.to(torch.float32).contiguous()
        out  = torch.zeros(T, H, dtype=torch.float32, device=codebook.device)

        rc = self._lib.ck_embedding_f32(
            _ctypes.c_void_p(ids.data_ptr()),
            _ctypes.c_void_p(pk.data_ptr()),
            _ctypes.c_void_p(cb.data_ptr()),
            _ctypes.c_void_p(out.data_ptr()),
            T, H, C, bits,
            self._stream_ptr(),
        )
        if rc != 0:
            raise RuntimeError(f"ck_embedding_f32 returned error {rc}")

        out_shape = list(orig_shape) + [H]
        return out.reshape(out_shape)


def _get_pytorch_hip_lib_dir() -> str:
    """Return PyTorch's bundled HIP library directory, or '' if not a ROCm build."""
    try:
        p = Path(torch.__file__).parent / 'lib'
        if (p / 'libamdhip64.so').exists():
            return str(p)
    except Exception:
        pass
    return ''


def _get_hip_code_object_version() -> str:
    """Return the HIP code-object version string that matches PyTorch's runtime.

    System hipcc may default to a newer code-object version than the ROCm
    runtime bundled with PyTorch.  Forcing the correct version avoids silent
    kernel-launch failures (the kernel runs but writes nothing).

    Mapping (approximate):
        ROCm 4.x  →  code object v4
        ROCm 5.x  →  code object v5
        ROCm 6.x  →  code object v5
        ROCm 7.x  →  code object v6
    """
    hip_ver = getattr(torch.version, 'hip', None)
    if hip_ver is None:
        return ''
    try:
        major = int(hip_ver.split('.')[0])
    except (ValueError, IndexError):
        return ''
    if major <= 4:
        return '4'
    elif major <= 6:
        return '5'
    else:
        return '6'


def _preload_hip_libs(pytorch_hip_dir: str) -> None:
    """Pre-open PyTorch's HSA/HIP runtime libs with RTLD_GLOBAL.

    Puts their symbols into the process-wide namespace so that when
    libcompressed_kernel.so loads its own libamdhip64 dependency, the
    dynamic linker reuses the already-loaded PyTorch versions instead of
    the system ROCm ones (which can be a different version and cause
    'undefined symbol … version ROCR_1' errors).
    """
    for name in ('libhsa-runtime64.so.1', 'libamdhip64.so'):
        path = Path(pytorch_hip_dir) / name
        if path.exists():
            try:
                _ctypes.CDLL(str(path), mode=_os.RTLD_GLOBAL)
            except Exception:
                pass  # best-effort; failures show up later when loading our .so


def _run_make(extra_args: list) -> bool:
    """Run make -C <rocm_dir> [extra_args]. Returns True on success."""
    try:
        result = _subprocess.run(
            ['make', '-C', str(_ROCM_DIR)] + extra_args,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  [compressed_kernel] make failed:\n{result.stderr[-600:]}")
            return False
        return True
    except Exception as e:
        print(f"  [compressed_kernel] make error: {e}")
        return False


def _build_and_load_rocm_so():
    """
    Build libcompressed_kernel.so (if needed) and load it via ctypes.
    Returns a _ROCmExtWrapper on success, None on failure.

    On ROCm, PyTorch bundles its own HIP runtime which can conflict with the
    system ROCm.  We handle this in two ways:
      1. Preload PyTorch's HIP/HSA libs globally so their symbols are available
         when our .so resolves its libamdhip64 dependency.
      2. If loading still fails, rebuild the .so against PyTorch's lib dir so
         its SONAME dependency becomes libamdhip64.so (no version suffix)
         instead of the system libamdhip64.so.7, which is already satisfied by
         PyTorch's loaded library.
    """
    so = _ROCM_SO
    pytorch_hip_dir = _get_pytorch_hip_lib_dir()

    # Step 1: Preload PyTorch's HIP runtime into the global symbol table.
    # This must happen before any dlopen that transitively loads libamdhip64.
    if pytorch_hip_dir:
        _preload_hip_libs(pytorch_hip_dir)

    # Step 2: Build the .so if it doesn't exist yet.
    cov = _get_hip_code_object_version()
    if not so.exists():
        print(f"  [compressed_kernel] Building ROCm kernel library ({_ROCM_DIR}/Makefile)...")
        make_args = []
        if pytorch_hip_dir:
            make_args.append(f'HIP_LIB_DIR={pytorch_hip_dir}')
        if cov:
            make_args.append(f'HIP_COV={cov}')
        if not _run_make(make_args):
            return None
        if not so.exists():
            print(f"  [compressed_kernel] {so} not found after build.")
            return None
        print("  [compressed_kernel] ROCm kernel library built successfully.")

    # Step 3: Try loading as-is (preloading may have resolved any conflict).
    try:
        wrapper = _ROCmExtWrapper(str(so))
        print(f"  [compressed_kernel] Loaded ROCm kernel library ({so.name}).")
        return wrapper
    except Exception as first_err:
        # Step 4: If loading failed and we know PyTorch's HIP dir, rebuild
        # against PyTorch's libs so the version dependency matches.
        if not pytorch_hip_dir:
            print(f"  [compressed_kernel] Failed to load {so}: {first_err}")
            return None

        print(
            f"  [compressed_kernel] Load failed ({str(first_err)[:100]})\n"
            f"  [compressed_kernel] Rebuilding against PyTorch HIP libs ({pytorch_hip_dir})..."
        )
        so.unlink(missing_ok=True)
        rebuild_args = [f'HIP_LIB_DIR={pytorch_hip_dir}']
        if cov:
            rebuild_args.append(f'HIP_COV={cov}')
        if not _run_make(rebuild_args):
            return None
        if not so.exists():
            print(f"  [compressed_kernel] {so} not found after rebuild.")
            return None

        try:
            wrapper = _ROCmExtWrapper(str(so))
            print(f"  [compressed_kernel] Loaded ROCm kernel library (PyTorch HIP build).")
            return wrapper
        except Exception as second_err:
            print(f"  [compressed_kernel] Failed to load even after rebuild: {second_err}")
            return None


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext

    if not torch.cuda.is_available():
        return None

    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

    # ── ROCm path: use standalone libcompressed_kernel.so ─────────────────
    # The PyTorch JIT hipify pipeline produces incorrect results on ROCm
    # (kernel compiles but outputs cos=0.0 vs reference).  The standalone
    # library compiled directly with hipcc is verified correct (cos=1.0).
    if is_rocm:
        arch = _detect_rocm_arch()
        print(f"  [compressed_kernel] ROCm backend detected"
              f"{f', targeting {arch}' if arch else ' (arch unknown)'}...")
        _ext = _build_and_load_rocm_so()
        if _ext is not None:
            return _ext
        print("  [compressed_kernel] Falling back to PyTorch JIT (results may be incorrect on ROCm).")

    # ── CUDA path (or ROCm fallback): PyTorch load_inline ─────────────────
    try:
        from torch.utils.cpp_extension import load_inline

        if is_rocm:
            extra_flags = ["-O3", "-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1", "-fno-gpu-rdc"]
            if arch:
                extra_flags.append(f"--offload-arch={arch}")
            cov = _get_hip_code_object_version()
            if cov:
                extra_flags.append(f"-mcode-object-version={cov}")
            backend_name = "ROCm/HIP (JIT fallback)"
        else:
            extra_flags = ["-O3", "--use_fast_math"]
            backend_name = "CUDA/NVCC"

        print(f"  [compressed_kernel] Compiling {backend_name} extension (first run ~30-60s)...")
        _ext = load_inline(
            name="compressed_matmul_v3",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            verbose=False,
            extra_cuda_cflags=extra_flags,
        )
        print(f"  [compressed_kernel] {backend_name} extension ready.")
    except Exception as e:
        print(f"  [compressed_kernel] Failed to compile extension: {e}")
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

    @classmethod
    def from_weight(cls, name: str, weight: torch.Tensor,
                    shape: tuple) -> 'GPUAcceleratedLinear':
        """Create from a plain (uncompressed) float weight matrix.

        weight : [M, K] float32 or bfloat16 tensor (CPU or GPU)
        shape  : (M, K)
        """
        obj = cls.__new__(cls)
        obj.name    = name
        obj.shape   = tuple(int(s) for s in shape)
        obj.M, obj.K = obj.shape[0], obj.shape[1]
        obj.bits    = 0       # sentinel: raw/uncompressed mode

        ext = _load_extension()
        obj._ext = ext

        device = weight.device if weight.is_cuda else torch.device('cuda')
        # Preserve the original dtype (bfloat16 for large models to save VRAM,
        # float32 for compressed-path codebook weights).
        store_dtype = weight.dtype if weight.dtype in (torch.float32, torch.bfloat16) \
                      else torch.float32
        obj.weight_gpu = weight.to(device=device, dtype=store_dtype)
        obj.codebook   = None
        obj.packed     = None
        return obj

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [..., K]  (any float dtype, usually bfloat16)
        returns: [..., M]  same dtype as x
        """
        in_dtype = x.dtype
        device   = x.device

        if self._ext is None:
            raise RuntimeError("CUDA/HIP extension not available for GPUAcceleratedLinear")

        if self.bits == 0:
            # Raw (uncompressed) path — kernel chosen by weight dtype (f32 or bf16)
            out = self._ext.fused_raw_linear(
                x.to(device), self.weight_gpu, self.M, self.K
            )
        else:
            # Compressed path
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

    @classmethod
    def from_weight(cls, name: str, weight: torch.Tensor,
                    shape: tuple) -> 'GPUAcceleratedEmbedding':
        """Create from a plain (uncompressed) float embedding table.

        weight : [vocab, H] float32 or bfloat16 tensor (CPU or GPU)
        shape  : (vocab, H)
        """
        obj = cls.__new__(cls)
        obj.name  = name
        obj.shape = tuple(int(s) for s in shape)
        obj.vocab, obj.hidden = obj.shape[0], obj.shape[1]
        obj.bits  = 0       # sentinel: raw/uncompressed mode

        ext = _load_extension()
        obj._ext = ext

        device = weight.device if weight.is_cuda else torch.device('cuda')
        store_dtype = weight.dtype if weight.dtype in (torch.float32, torch.bfloat16) \
                      else torch.float32
        obj.weight_gpu = weight.to(device=device, dtype=store_dtype)
        obj.codebook   = None
        obj.packed     = None
        return obj

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [...] int64 token IDs
        returns: [..., hidden]  float32 or bfloat16 depending on weight storage
        """
        if self._ext is None:
            raise RuntimeError("CUDA/HIP extension not available for GPUAcceleratedEmbedding")

        if self.bits == 0:
            # Raw (uncompressed) path
            device = self.weight_gpu.device
            out = self._ext.fused_raw_embedding(
                x.to(device), self.weight_gpu, self.hidden
            )
        else:
            # Compressed path
            device = self.codebook.device
            out = self._ext.fused_compressed_embedding(
                x.to(device),
                self.packed,
                self.codebook,
                self.hidden,
                self.bits,
            )
        return out
