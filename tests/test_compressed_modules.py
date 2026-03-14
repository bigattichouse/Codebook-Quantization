"""
Tests for AdaptiveCodebookLinear and AdaptiveCodebookEmbedding using
synthetic compressed data (no real model needed).
"""

import pytest
import torch
import numpy as np

# Disable GPU acceleration in compressed_modules for these unit tests —
# we test the CPU decode path here; GPU kernels are tested in test_gpu_kernels.py.
import compressed_modules
compressed_modules.GPU_ACCELERATED_AVAILABLE = False
compressed_modules.HIP_AVAILABLE = False

from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
from bitpack import pack_any_bits


class TestAdaptiveCodebookLinearExact:
    """Test the 'exact' mode (uncompressed) path."""

    def test_forward_shape(self):
        shape = (64, 32)  # out_features=64, in_features=32
        data = {
            "mode": "exact",
            "shape": shape,
            "data": np.random.randn(64 * 32).astype(np.float32).view(np.uint32).astype(np.uint16),
        }
        layer = AdaptiveCodebookLinear.from_compressed("test.weight", data, {})
        x = torch.randn(2, 32)
        out = layer(x)
        assert out.shape == (2, 64)

    def test_output_dtype(self):
        shape = (16, 8)
        weight = np.random.randn(16 * 8).astype(np.float32)
        bf16 = (weight.view(np.uint32) >> 16).astype(np.uint16)
        data = {"mode": "exact", "shape": shape, "data": bf16}
        layer = AdaptiveCodebookLinear.from_compressed("test.weight", data, {})
        x = torch.randn(1, 8, dtype=torch.float32)
        out = layer(x)
        assert out.dtype == torch.float32


class TestAdaptiveCodebookLinearCodebook:
    """Test the 'direct_codebook' mode path."""

    def _make_codebook_layer(self, out_feat=64, in_feat=32, cb_size=256, bits=8):
        """Build an AdaptiveCodebookLinear from synthetic compressed data."""
        shape = (out_feat, in_feat)
        n = out_feat * in_feat
        codebook = np.sort(np.random.randn(cb_size).astype(np.float32) * 0.02)
        indices_raw = np.random.randint(0, cb_size, size=n, dtype=np.uint16)
        if bits != 8 and bits != 16:
            packed = pack_any_bits(indices_raw, bits)
        else:
            packed = indices_raw.astype(np.uint8) if bits == 8 else indices_raw
        data = {
            "mode": "direct_codebook",
            "shape": shape,
            "bits": bits,
            "indices": packed,
            "codebook": codebook,
        }
        return AdaptiveCodebookLinear.from_compressed("test.weight", data, {}), codebook, indices_raw

    def test_forward_shape(self):
        layer, _, _ = self._make_codebook_layer()
        x = torch.randn(2, 32)
        out = layer(x)
        assert out.shape == (2, 64)

    def test_cosine_similarity_to_reference(self):
        """Compressed forward should approximate F.linear with reconstructed weight."""
        out_feat, in_feat = 64, 32
        layer, codebook, indices = self._make_codebook_layer(out_feat, in_feat, cb_size=256, bits=8)
        # Reconstruct full weight for reference
        ref_weight = torch.from_numpy(codebook[indices].reshape(out_feat, in_feat))
        x = torch.randn(4, in_feat)
        out_compressed = layer(x)
        out_reference = torch.nn.functional.linear(x, ref_weight)
        cos = torch.nn.functional.cosine_similarity(
            out_compressed.flatten().unsqueeze(0),
            out_reference.flatten().unsqueeze(0),
        )
        assert cos.item() > 0.9

    def test_global_codebook_fallback(self):
        """Layer should work with a global codebook when no local one is provided."""
        shape = (32, 16)
        n = 32 * 16
        cb = np.sort(np.random.randn(128).astype(np.float32) * 0.02)
        indices = np.random.randint(0, 128, size=n, dtype=np.uint8)
        data = {
            "mode": "direct_codebook",
            "shape": shape,
            "bits": 8,
            "indices": indices,
            "codebook_type": "attention",
        }
        global_cbs = {"attention": torch.from_numpy(cb)}
        layer = AdaptiveCodebookLinear.from_compressed("test.weight", data, global_cbs)
        x = torch.randn(1, 16)
        out = layer(x)
        assert out.shape == (1, 32)


class TestAdaptiveCodebookEmbeddingExact:
    def test_forward_shape(self):
        shape = (100, 32)  # vocab=100, hidden=32
        weight = np.random.randn(100 * 32).astype(np.float32)
        bf16 = (weight.view(np.uint32) >> 16).astype(np.uint16)
        data = {"mode": "exact", "shape": shape, "data": bf16}
        layer = AdaptiveCodebookEmbedding.from_compressed("embed.weight", data, {})
        x = torch.tensor([0, 5, 99], dtype=torch.long)
        out = layer(x)
        assert out.shape == (3, 32)


class TestAdaptiveCodebookEmbeddingCodebook:
    def test_forward_shape(self):
        shape = (100, 32)
        n = 100 * 32
        cb = np.sort(np.random.randn(256).astype(np.float32) * 0.02)
        indices = np.random.randint(0, 256, size=n, dtype=np.uint8)
        data = {
            "mode": "direct_codebook",
            "shape": shape,
            "bits": 8,
            "indices": indices,
            "codebook": cb,
        }
        layer = AdaptiveCodebookEmbedding.from_compressed("embed.weight", data, {})
        x = torch.tensor([0, 10, 50], dtype=torch.long)
        out = layer(x)
        assert out.shape == (3, 32)
