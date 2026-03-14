"""
Tests for core compressor functions: kmeans, codebook assignment, dtype conversion,
tensor classification, and raw tensor loading.
"""

import pytest
import numpy as np
import struct
import json
import tempfile
from pathlib import Path

from compressor import (
    kmeans_1d,
    assign_to_codebook,
    classify_tensor,
    bfloat16_to_float32,
    float32_to_bfloat16,
    load_raw_tensor_data,
)


class TestKmeans1D:
    def test_returns_sorted(self):
        data = np.random.randn(5000).astype(np.float32)
        centroids = kmeans_1d(data, k=16)
        assert np.all(centroids[:-1] <= centroids[1:])

    def test_correct_count(self):
        data = np.random.randn(5000).astype(np.float32)
        centroids = kmeans_1d(data, k=32)
        assert len(centroids) == 32

    def test_centroids_within_range(self):
        data = np.random.randn(5000).astype(np.float32)
        centroids = kmeans_1d(data, k=16)
        assert centroids.min() >= data.min() - 0.1
        assert centroids.max() <= data.max() + 0.1

    def test_few_unique_values(self):
        """When data has fewer unique values than k, should still return k centroids."""
        data = np.array([1.0, 2.0, 3.0] * 1000, dtype=np.float32)
        centroids = kmeans_1d(data, k=8)
        assert len(centroids) == 8


class TestAssignToCodebook:
    def test_basic_assignment(self):
        codebook = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        data = np.array([0.1, 0.9, 2.1, 2.8], dtype=np.float32)
        indices = assign_to_codebook(data, codebook)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])

    def test_indices_in_range(self):
        codebook = np.sort(np.random.randn(256).astype(np.float32))
        data = np.random.randn(10000).astype(np.float32)
        indices = assign_to_codebook(data, codebook)
        assert indices.min() >= 0
        assert indices.max() < len(codebook)

    def test_reconstruction_close(self):
        """Reconstruction from codebook should be close to original."""
        data = np.random.randn(10000).astype(np.float32) * 0.02
        codebook = kmeans_1d(data, k=256)
        indices = assign_to_codebook(data, codebook)
        recon = codebook[indices]
        mse = np.mean((data - recon) ** 2)
        assert mse < 1e-4  # Should be very low for 256 centroids


class TestBFloat16Conversion:
    def test_roundtrip_preserves_values(self):
        """BF16 -> F32 -> BF16 roundtrip should be identity."""
        original_f32 = np.array([1.0, -0.5, 0.0, 3.14, -100.0], dtype=np.float32)
        bf16 = float32_to_bfloat16(original_f32)
        raw_bytes = bf16.tobytes()
        recovered = bfloat16_to_float32(raw_bytes)
        # BF16 truncates mantissa, so we compare within BF16 precision
        np.testing.assert_allclose(recovered, original_f32, rtol=1e-2)

    def test_zero(self):
        f32 = np.array([0.0], dtype=np.float32)
        bf16 = float32_to_bfloat16(f32)
        recovered = bfloat16_to_float32(bf16.tobytes())
        assert recovered[0] == 0.0

    def test_dtype(self):
        f32 = np.array([1.0], dtype=np.float32)
        bf16 = float32_to_bfloat16(f32)
        assert bf16.dtype == np.uint16


class TestClassifyTensor:
    """Test classify_tensor covers all major model families."""

    # Llama / Mistral patterns
    @pytest.mark.parametrize("name,expected", [
        ("model.layers.0.self_attn.q_proj.weight", "attention"),
        ("model.layers.0.self_attn.k_proj.weight", "attention"),
        ("model.layers.0.self_attn.v_proj.weight", "attention"),
        ("model.layers.0.self_attn.o_proj.weight", "attention"),
        ("model.layers.0.mlp.gate_proj.weight", "mlp_ffn"),
        ("model.layers.0.mlp.up_proj.weight", "mlp_ffn"),
        ("model.layers.0.mlp.down_proj.weight", "mlp_ffn"),
        ("model.layers.0.input_layernorm.weight", "layernorm"),
        ("model.layers.0.post_attention_layernorm.weight", "layernorm"),
        ("model.embed_tokens.weight", "embedding"),
        ("lm_head.weight", "embedding"),
    ])
    def test_llama_patterns(self, name, expected):
        assert classify_tensor(name) == expected

    # Gemma extra patterns
    @pytest.mark.parametrize("name,expected", [
        ("model.layers.0.pre_feedforward_layernorm.weight", "layernorm"),
        ("model.layers.0.post_feedforward_layernorm.weight", "layernorm"),
    ])
    def test_gemma_patterns(self, name, expected):
        assert classify_tensor(name) == expected

    # Qwen 3.5 patterns (language_model prefix)
    @pytest.mark.parametrize("name,expected", [
        ("model.language_model.layers.0.self_attn.q_proj.weight", "attention"),
        ("model.language_model.layers.0.mlp.gate_proj.weight", "mlp_ffn"),
        ("model.language_model.embed_tokens.weight", "embedding"),
        ("model.language_model.norm.weight", "layernorm"),
    ])
    def test_qwen_patterns(self, name, expected):
        assert classify_tensor(name) == expected

    # MoE patterns
    @pytest.mark.parametrize("name,expected", [
        ("model.layers.0.block_sparse_moe.gate.weight", "router"),
        ("model.layers.0.block_sparse_moe.experts.0.w1.weight", "moe_expert"),
    ])
    def test_moe_patterns(self, name, expected):
        assert classify_tensor(name) == expected


class TestLoadRawTensorData:
    def test_bf16_load(self, tmp_path):
        """Load BF16 data from a fake file with offset."""
        shape = (4, 8)
        n = 32
        original_f32 = np.random.randn(n).astype(np.float32)
        bf16_uint16 = (original_f32.view(np.uint32) >> 16).astype(np.uint16)
        raw = bf16_uint16.tobytes()

        fpath = tmp_path / "test.bin"
        # Write some junk prefix then the data
        prefix = b'\x00' * 100
        fpath.write_bytes(prefix + raw)

        with open(fpath, "rb") as f:
            result = load_raw_tensor_data(f, offset=100, size=len(raw),
                                          shape=shape, dtype_str="BF16")
        assert result.shape == shape
        assert result.dtype == np.float32
        # Values should be close (BF16 precision)
        np.testing.assert_allclose(result.flatten(), original_f32, rtol=1e-2)
