"""
Tests for CUDA GPU kernels — skipped gracefully when no GPU is available.
"""

import pytest
import torch
import numpy as np

from bitpack import pack_any_bits

# Skip entire module if CUDA not available
pytestmark = pytest.mark.gpu
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@requires_cuda
class TestGPULinearKernel:
    """Test fused_compressed_linear against CPU F.linear reference."""

    def _setup(self, M=64, K=32, cb_size=256, bits=13):
        from gpu_accelerated_functions import GPUAcceleratedLinear
        codebook = np.sort(np.random.randn(cb_size).astype(np.float32) * 0.02)
        indices = np.random.randint(0, cb_size, size=M * K, dtype=np.uint16)
        packed = pack_any_bits(indices, bits)
        cb_tensor = torch.from_numpy(codebook).cuda()
        gpu_linear = GPUAcceleratedLinear("test", packed, cb_tensor, (M, K), bits)
        # Reference weight
        ref_weight = torch.from_numpy(codebook[indices].reshape(M, K)).cuda()
        return gpu_linear, ref_weight

    def test_correctness(self):
        gpu_linear, ref_weight = self._setup()
        x = torch.randn(2, 32, device="cuda")
        out_gpu = gpu_linear(x)
        out_ref = torch.nn.functional.linear(x, ref_weight)
        cos = torch.nn.functional.cosine_similarity(
            out_gpu.flatten().unsqueeze(0), out_ref.flatten().unsqueeze(0)
        )
        assert cos.item() > 0.999

    def test_output_shape(self):
        gpu_linear, _ = self._setup(M=128, K=64)
        x = torch.randn(4, 64, device="cuda")
        out = gpu_linear(x)
        assert out.shape == (4, 128)

    def test_dtype_cast(self):
        """Output should match input dtype (e.g. bfloat16)."""
        gpu_linear, _ = self._setup()
        x = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
        out = gpu_linear(x)
        assert out.dtype == torch.bfloat16


@requires_cuda
class TestGPUEmbeddingKernel:
    def test_correctness(self):
        from gpu_accelerated_functions import GPUAcceleratedEmbedding
        vocab, hidden, cb_size, bits = 100, 32, 256, 8
        codebook = np.sort(np.random.randn(cb_size).astype(np.float32) * 0.02)
        indices = np.random.randint(0, cb_size, size=vocab * hidden, dtype=np.uint8)
        cb_tensor = torch.from_numpy(codebook).cuda()
        gpu_emb = GPUAcceleratedEmbedding("test", indices, cb_tensor, (vocab, hidden), bits)
        token_ids = torch.tensor([0, 10, 50], dtype=torch.long, device="cuda")
        out = gpu_emb(token_ids)
        assert out.shape == (3, hidden)
        # Check against CPU reference
        ref_weight = codebook[indices].reshape(vocab, hidden)
        for i, tid in enumerate([0, 10, 50]):
            ref_row = torch.from_numpy(ref_weight[tid]).cuda()
            cos = torch.nn.functional.cosine_similarity(
                out[i].unsqueeze(0), ref_row.unsqueeze(0)
            )
            assert cos.item() > 0.99
