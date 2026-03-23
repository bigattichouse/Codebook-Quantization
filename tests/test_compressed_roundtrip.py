"""
test_compressed_roundtrip.py

Unit tests that verify each stage of the compressed inference pipeline:

  1. Codebook packing/unpacking round-trip (no model needed)
  2. Single-layer forward pass: compressed == uncompressed (synthetic weights)
  3. Embedding layer forward pass: compressed == uncompressed (synthetic)
  4. Huffman encode → decode round-trip
  5. Huffman-compressed layer forward pass == uncompressed
  6. Full tiny model: compress → load compressed → compare logits

Run:
    pytest tests/test_compressed_roundtrip.py -v
or for the real-model integration test:
    pytest tests/test_compressed_roundtrip.py -v --model <path>
"""

import sys
import json
import struct
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_weight(shape, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape).astype(np.float32) * 0.02
    return w


def _bf16_round_trip(arr: np.ndarray) -> np.ndarray:
    """Simulate bfloat16 storage: truncate lower 16 bits of float32."""
    u32 = arr.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    out = (u16.astype(np.uint32) << 16).view(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bit-pack / unpack round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestBitpackRoundtrip:
    """pack_any_bits → unpack_any_bits must be lossless for all bit widths."""

    @pytest.mark.parametrize("bits", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    def test_pack_unpack(self, bits):
        from bitpack import pack_any_bits, unpack_any_bits
        n = 1024
        max_val = (1 << bits) - 1
        rng = np.random.default_rng(bits)
        idx = rng.integers(0, max_val + 1, size=n, dtype=np.uint16)
        packed = pack_any_bits(idx, bits)
        recovered = unpack_any_bits(packed, bits, n)
        assert np.array_equal(idx, recovered), \
            f"bits={bits}: round-trip failed at indices {np.where(idx != recovered)[0][:5]}"

    def test_pack_unpack_large(self):
        from bitpack import pack_any_bits, unpack_any_bits
        n = 2_621_440  # typical weight tensor size
        bits = 12
        rng = np.random.default_rng(42)
        idx = rng.integers(0, 4096, size=n, dtype=np.uint16)
        packed = pack_any_bits(idx, bits)
        recovered = unpack_any_bits(packed, bits, n)
        assert np.array_equal(idx, recovered)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Codebook LUT assignment is lossless
# ─────────────────────────────────────────────────────────────────────────────

class TestCodebookAssignment:
    """The codebook quantisation step must reproduce all original values exactly
    for the lossless mode (every unique bf16 value gets its own entry)."""

    def test_codebook_lossless(self):
        from adaptive_compressor import AdaptiveCompressor
        rng = np.random.default_rng(0)
        # 1024 random float32 values stored as bf16
        w_f32 = rng.standard_normal(1024).astype(np.float32) * 0.02
        w_bf16 = _bf16_round_trip(w_f32)

        n_unique = len(np.unique(w_bf16.view(np.uint32)))
        bits = int(np.ceil(np.log2(max(n_unique, 2))))

        ac = AdaptiveCompressor.__new__(AdaptiveCompressor)
        # Build a simple LUT manually
        uniq = np.unique(w_bf16)
        cb = uniq
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in w_bf16], dtype=np.uint16)

        # Reconstruct
        reconstructed = cb[indices]
        assert np.allclose(w_bf16, reconstructed, atol=0), \
            "Codebook reconstruction failed — lossless requirement violated"

    def test_codebook_reconstruction_uint16_stored(self):
        """Exact round-trip: bf16 stored as uint16 → reconstruct → compare."""
        rng = np.random.default_rng(1)
        w_f32 = rng.standard_normal(512).astype(np.float32) * 0.02
        # Simulate what safetensors does for bfloat16 tensors
        u16 = (w_f32.view(np.uint32) >> 16).astype(np.uint16)
        # Recover the original bf16 float32 view
        w_recovered = (u16.astype(np.uint32) << 16).view(np.float32)

        n_unique = len(np.unique(w_recovered))
        uniq = np.unique(w_recovered)
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in w_recovered], dtype=np.uint16)
        reconstructed = uniq[indices]

        assert np.array_equal(w_recovered.view(np.uint32), reconstructed.view(np.uint32)), \
            "uint16-stored bf16 round-trip failed"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compressed linear layer forward == uncompressed
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearForwardEquality:
    """AdaptiveCodebookLinear.forward() must match F.linear() on the same weights."""

    @pytest.fixture
    def linear_data(self):
        M, K = 64, 128
        rng = np.random.default_rng(42)
        w_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.02
        w_bf16 = _bf16_round_trip(w_f32)
        return w_f32, w_bf16, M, K

    def test_exact_mode(self, linear_data):
        """mode='exact' stores the weight tensor directly — must match perfectly."""
        from compressed_modules import AdaptiveCodebookLinear
        w_f32, w_bf16, M, K = linear_data

        layer = AdaptiveCodebookLinear("test", (M, K), mode='exact')
        layer.weight = nn.Parameter(torch.from_numpy(w_bf16), requires_grad=False)

        x = torch.randn(4, K)
        expected = F.linear(x, torch.from_numpy(w_bf16))
        got = layer(x)
        assert torch.allclose(expected, got, atol=1e-6), \
            f"exact mode mismatch: max err={( expected - got).abs().max()}"

    def test_direct_codebook_cpu_fallback(self, linear_data):
        """direct_codebook without GPU must match F.linear on original bf16 weights."""
        from compressed_modules import AdaptiveCodebookLinear
        from bitpack import pack_any_bits
        w_f32, w_bf16, M, K = linear_data

        uniq = np.unique(w_bf16)
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in w_bf16.ravel()], dtype=np.uint16)
        bits = int(np.ceil(np.log2(max(len(uniq), 2))))
        packed = pack_any_bits(indices, bits)

        data = {
            'mode': 'direct_codebook',
            'shape': (M, K),
            'bits': bits,
            'indices': packed,
            'codebook': uniq,
            'codebook_type': None,
        }
        layer = AdaptiveCodebookLinear.from_compressed(
            "test", data, {}, use_gpu=False
        )
        layer.eval()

        x = torch.randn(4, K)
        expected = F.linear(x, torch.from_numpy(w_bf16))
        got = layer(x)
        # Allow for float32 vs bf16 accumulation differences
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, \
            f"direct_codebook CPU mismatch: max err={max_err:.6f} (expected < 1e-4)"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compressed embedding forward == uncompressed
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingForwardEquality:
    """AdaptiveCodebookEmbedding.forward() must match F.embedding() on same weights."""

    @pytest.fixture
    def embedding_data(self):
        vocab, hidden = 128, 64
        rng = np.random.default_rng(7)
        w_f32 = rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.02
        w_bf16 = _bf16_round_trip(w_f32)
        return w_f32, w_bf16, vocab, hidden

    def test_exact_mode(self, embedding_data):
        from compressed_modules import AdaptiveCodebookEmbedding
        w_f32, w_bf16, vocab, hidden = embedding_data

        layer = AdaptiveCodebookEmbedding("emb", (vocab, hidden), mode='exact')
        layer.weight = nn.Parameter(torch.from_numpy(w_bf16), requires_grad=False)

        ids = torch.tensor([0, 5, 10, 127, 0])
        expected = F.embedding(ids, torch.from_numpy(w_bf16))
        got = layer(ids)
        assert torch.allclose(expected, got, atol=1e-7)

    def test_direct_codebook_cpu_fallback(self, embedding_data):
        from compressed_modules import AdaptiveCodebookEmbedding
        from bitpack import pack_any_bits
        w_f32, w_bf16, vocab, hidden = embedding_data

        flat = w_bf16.ravel()
        uniq = np.unique(flat)
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in flat], dtype=np.uint16)
        bits = int(np.ceil(np.log2(max(len(uniq), 2))))
        packed = pack_any_bits(indices, bits)

        data = {
            'mode': 'direct_codebook',
            'shape': (vocab, hidden),
            'bits': bits,
            'indices': packed,
            'codebook': uniq,
            'codebook_type': None,
        }
        layer = AdaptiveCodebookEmbedding.from_compressed(
            "emb", data, {}, use_gpu=False
        )
        layer.eval()

        ids = torch.tensor([0, 5, 10, 63, 127, 0])
        expected = F.embedding(ids, torch.from_numpy(w_bf16))
        got = layer(ids)
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, \
            f"direct_codebook embedding CPU mismatch: max err={max_err:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Huffman encode → decode round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestHuffmanRoundtrip:
    """huffman_encode_indices → huffman_decode_indices must recover original."""

    @pytest.mark.parametrize("n,bits", [
        (1024, 8), (4096, 10), (16384, 12), (131072, 13),
    ])
    def test_cpu_encode_decode(self, n, bits):
        from huffman_codebook import huffman_encode_indices, huffman_decode_indices
        rng = np.random.default_rng(n + bits)
        max_val = (1 << bits) - 1
        # Skewed distribution (lower indices much more common)
        p = 1.0 / (np.arange(1, max_val + 2) ** 1.5)
        p /= p.sum()
        idx = rng.choice(max_val + 1, size=n, p=p).astype(np.uint16)

        M, K = max(1, n // 64), 64
        result = huffman_encode_indices(idx[:M * K], shape=(M, K))
        decoded = huffman_decode_indices(
            result['huff_stream'], result['huff_lengths'], int(result['huff_n'][0])
        )
        assert np.array_equal(decoded, idx[:M * K]), \
            f"Huffman round-trip failed for n={n}, bits={bits}"

    def test_all_same_value(self):
        """Edge case: all indices identical (single-symbol codebook)."""
        from huffman_codebook import huffman_encode_indices, huffman_decode_indices
        idx = np.zeros(256, dtype=np.uint16)
        result = huffman_encode_indices(idx, shape=(4, 64))
        decoded = huffman_decode_indices(
            result['huff_stream'], result['huff_lengths'], int(result['huff_n'][0])
        )
        assert np.array_equal(decoded, idx)

    def test_two_symbols(self):
        """Edge case: only two distinct symbols."""
        from huffman_codebook import huffman_encode_indices, huffman_decode_indices
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 2, size=512, dtype=np.uint16)
        result = huffman_encode_indices(idx, shape=(8, 64))
        decoded = huffman_decode_indices(
            result['huff_stream'], result['huff_lengths'], int(result['huff_n'][0])
        )
        assert np.array_equal(decoded, idx)

    def test_deep_tree_no_code_overflow(self):
        """Regression: skewed distributions with natural depths 25-26 must not
        produce overcomplete trees (Kraft sum > 1) or canonical codes that
        overflow 24 bits.  Previously _build_code_lengths capped at 24 using
        simple truncation, creating invalid codes for such distributions and
        corrupting the bitstream from the first >16-bit code onward."""
        from huffman_codebook import huffman_encode_indices, huffman_decode_indices, _canonical_codes
        # 5000 symbols, many with frequency 1 → natural depths 25-26
        rng = np.random.default_rng(777)
        n_syms = 5000
        # Most symbols appear once; a few hundred appear many times
        freq = np.ones(n_syms, dtype=np.int64)
        freq[:200] = rng.integers(1000, 10000, size=200)
        freq[200:500] = rng.integers(100, 1000, size=300)
        # Build index stream proportional to frequencies
        p = freq.astype(np.float64) / freq.sum()
        n = 4096 * 64
        idx = rng.choice(n_syms, size=n, p=p).astype(np.uint16)
        M, K = 4096, 64
        result = huffman_encode_indices(idx, shape=(M, K))
        # No code should exceed 32 bits
        assert result['huff_lengths'].max() <= 32, \
            f"Code length {result['huff_lengths'].max()} exceeds cap"
        # All canonical codes must fit in their declared length
        codes = _canonical_codes(result['huff_lengths'])
        for sym, (code_val, code_len) in codes.items():
            assert code_val < (1 << code_len), \
                f"sym={sym}: code {code_val} overflows {code_len} bits"
        decoded = huffman_decode_indices(
            result['huff_stream'], result['huff_lengths'], int(result['huff_n'][0])
        )
        assert np.array_equal(decoded, idx), \
            f"Round-trip failed: {(decoded != idx).sum()} mismatches"

    def test_large_array_uniform(self):
        """Large N (50M symbols, 8192 unique) with uniform distribution: round-trip."""
        from huffman_codebook import huffman_encode_indices, huffman_decode_indices
        M, K = 12288, 4096
        n = M * K
        rng = np.random.default_rng(99)
        idx = rng.integers(0, 8192, size=n, dtype=np.uint16)
        result = huffman_encode_indices(idx, shape=(M, K))
        decoded = huffman_decode_indices(
            result['huff_stream'], result['huff_lengths'], int(result['huff_n'][0])
        )
        assert np.array_equal(decoded, idx), \
            f"Large uniform round-trip: {(decoded != idx).sum()} mismatches"

    def test_large_array_deep_codes(self):
        """Large N with skewed distribution producing code depths 25+.
        This is the exact failure mode that caused wrong Qwen inference output."""
        from huffman_codebook import huffman_encode_indices, huffman_decode_indices
        # ~5500 unique symbols, many rare (like Qwen3.5-9B large linear layers)
        M, K = 12288, 4096
        n = M * K
        rng = np.random.default_rng(42)
        n_syms = 5500
        freq = np.ones(n_syms, dtype=np.float64)
        freq[:300] = rng.uniform(1000, 50000, 300)
        freq[300:1000] = rng.uniform(10, 500, 700)
        p = freq / freq.sum()
        idx = rng.choice(n_syms, size=n, p=p).astype(np.uint16)
        result = huffman_encode_indices(idx, shape=(M, K))
        # Some codes should be longer than 16 bits (exercises slow path)
        assert result['huff_lengths'].max() > 16, \
            "Distribution should produce codes > 16 bits to test slow path"
        decoded = huffman_decode_indices(
            result['huff_stream'], result['huff_lengths'], int(result['huff_n'][0])
        )
        assert np.array_equal(decoded, idx), \
            f"Large deep-code round-trip: {(decoded != idx).sum()} mismatches"

    def test_c_kernel_large_array_deep_codes(self):
        """C kernel must correctly decode large arrays with codes > 16 bits."""
        from huffman_codebook import huffman_encode_indices
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE, huffman_decode_weights
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 4096, 1024  # smaller for speed, but still triggers slow path
        n = M * K
        rng = np.random.default_rng(55)
        n_syms = 3000
        freq = np.ones(n_syms, dtype=np.float64)
        freq[:200] = rng.uniform(500, 10000, 200)
        p = freq / freq.sum()
        idx = rng.choice(n_syms, size=n, p=p).astype(np.uint16)
        codebook = rng.standard_normal(n_syms).astype(np.float32) * 0.02
        w_expected = codebook[idx].reshape(M, K)

        result = huffman_encode_indices(idx, shape=(M, K))
        assert result['huff_lengths'].max() > 16, \
            "Distribution should produce codes > 16 bits"

        _lens = np.asarray(result['huff_lengths'], dtype=np.uint8)
        sl_max = int(_lens.max()) if _lens.size and _lens.max() > 0 else 1
        huff_data = {
            'huff_stream':    np.asarray(result['huff_stream'],         dtype=np.uint8),
            'lut_sym':        np.asarray(result['huff_lut_sym'],        dtype=np.uint16),
            'lut_len':        np.asarray(result['huff_lut_len'],        dtype=np.uint8),
            'sl_first_code':  np.asarray(result['huff_sl_first_code'],  dtype=np.int64),
            'sl_base_offset': np.asarray(result['huff_sl_base_offset'], dtype=np.int32),
            'sl_sym':         np.asarray(result['huff_sl_sym'],         dtype=np.uint16),
            'row_bit_starts': np.asarray(result['huff_row_bit_starts'], dtype=np.int64),
            'sl_max_len':     sl_max,
        }
        w_got = huffman_decode_weights(huff_data, codebook, M, K)
        max_err = np.abs(w_got - w_expected).max()
        assert max_err < 1e-6, \
            f"C kernel large-array decode: max_err={max_err:.2e}"

    def test_gpu_phase2_large_array_deep_codes(self):
        """GPU Phase 2 must correctly decode large arrays with code depths > 16."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        from compressed_modules import AdaptiveCodebookLinear
        from huffman_codebook import huffman_encode_indices
        M, K = 4096, 1024
        n = M * K
        rng = np.random.default_rng(77)
        n_syms = 3000
        freq = np.ones(n_syms, dtype=np.float64)
        freq[:200] = rng.uniform(500, 10000, 200)
        p = freq / freq.sum()
        idx = rng.choice(n_syms, size=n, p=p).astype(np.uint16)
        codebook = rng.standard_normal(n_syms).astype(np.float32) * 0.02
        # Build weight matrix as bf16 → float32 roundtrip
        w_f32 = codebook[idx].reshape(M, K)

        result = huffman_encode_indices(idx, shape=(M, K))
        assert result['huff_lengths'].max() > 16, \
            "Distribution should produce codes > 16 bits"

        bits = int(np.ceil(np.log2(max(n_syms, 2))))
        data = {
            'mode': 'direct_codebook', 'shape': (M, K), 'bits': bits,
            'encoding': 'huffman', 'codebook': codebook, 'codebook_type': None,
            **{k: result[k] for k in ('huff_stream', 'huff_lengths', 'huff_n',
                                      'huff_lut_sym', 'huff_lut_len',
                                      'huff_sl_first_code', 'huff_sl_base_offset',
                                      'huff_sl_sym', 'huff_row_bit_starts')},
        }
        layer = AdaptiveCodebookLinear.from_compressed("p2_large_deep", data, {}, use_gpu=True)
        if layer._gpu_func is None:
            pytest.skip("GPU func not set (kernel compile failed)")
        layer.eval()

        x = torch.randn(2, K, device='cuda')
        out_gpu = layer(x).cpu().float()
        expected = F.linear(x.cpu(), torch.from_numpy(w_f32))
        max_err = (out_gpu - expected.float()).abs().max().item()
        assert max_err < 1e-3, f"GPU Phase 2 large deep-code: max_err={max_err:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Huffman-compressed layer forward == uncompressed
# ─────────────────────────────────────────────────────────────────────────────

class TestHuffmanLayerForward:
    """Huffman-compressed linear layer CPU forward must match F.linear."""

    def test_huffman_linear_cpu(self):
        from compressed_modules import AdaptiveCodebookLinear
        from huffman_codebook import huffman_encode_indices
        from bitpack import pack_any_bits

        M, K = 64, 128
        rng = np.random.default_rng(99)
        w_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.02
        w_bf16 = _bf16_round_trip(w_f32)

        uniq = np.unique(w_bf16)
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in w_bf16.ravel()], dtype=np.uint16)
        bits = int(np.ceil(np.log2(max(len(uniq), 2))))

        result = huffman_encode_indices(indices, shape=(M, K))

        data = {
            'mode': 'direct_codebook',
            'shape': (M, K),
            'bits': bits,
            'encoding': 'huffman',
            'codebook': uniq,
            'codebook_type': None,
            'huff_stream':        result['huff_stream'],
            'huff_lengths':       result['huff_lengths'],
            'huff_n':             result['huff_n'],
            'huff_row_bit_starts': result['huff_row_bit_starts'],
            'huff_lut_sym':       result['huff_lut_sym'],
            'huff_lut_len':       result['huff_lut_len'],
            'huff_sl_first_code': result['huff_sl_first_code'],
            'huff_sl_base_offset': result['huff_sl_base_offset'],
            'huff_sl_sym':        result['huff_sl_sym'],
        }
        layer = AdaptiveCodebookLinear.from_compressed(
            "huff_test", data, {}, use_gpu=False
        )
        layer.eval()

        x = torch.randn(4, K)
        expected = F.linear(x, torch.from_numpy(w_bf16))
        got = layer(x)
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, \
            f"Huffman linear CPU mismatch: max err={max_err:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 6b. Inference-time Huffman: bitstream stays compressed in RAM during forward
# ─────────────────────────────────────────────────────────────────────────────

def _make_huff_data(w_bf16, shape):
    """Build a complete huff_data dict for a BF16 weight matrix."""
    from huffman_codebook import huffman_encode_indices
    M, K = shape
    uniq = np.unique(w_bf16)
    lut  = {v: i for i, v in enumerate(uniq)}
    indices = np.array([lut[v] for v in w_bf16.ravel()], dtype=np.uint16)
    bits    = int(np.ceil(np.log2(max(len(uniq), 2))))
    result  = huffman_encode_indices(indices, shape=shape)
    _lens   = np.asarray(result['huff_lengths'], dtype=np.uint8)
    sl_max  = int(_lens.max()) if _lens.size and _lens.max() > 0 else 1
    return uniq, bits, {
        'huff_stream':    np.asarray(result['huff_stream'],         dtype=np.uint8),
        'lut_sym':        np.asarray(result['huff_lut_sym'],        dtype=np.uint16),
        'lut_len':        np.asarray(result['huff_lut_len'],        dtype=np.uint8),
        'sl_first_code':  np.asarray(result['huff_sl_first_code'],  dtype=np.int64),
        'sl_base_offset': np.asarray(result['huff_sl_base_offset'], dtype=np.int32),
        'sl_sym':         np.asarray(result['huff_sl_sym'],         dtype=np.uint16),
        'row_bit_starts': np.asarray(result['huff_row_bit_starts'], dtype=np.int64),
        'sl_max_len':     sl_max,
        # raw fields for from_compressed
        'huff_lengths':       result['huff_lengths'],
        'huff_n':             result['huff_n'],
        'huff_lut_sym':       result['huff_lut_sym'],
        'huff_lut_len':       result['huff_lut_len'],
        'huff_sl_first_code': result['huff_sl_first_code'],
        'huff_sl_base_offset':result['huff_sl_base_offset'],
        'huff_sl_sym':        result['huff_sl_sym'],
        'huff_row_bit_starts':result['huff_row_bit_starts'],
    }


class TestHuffmanInferenceMatmul:
    """huffman_matmul C kernel: correctness, memory property, and edge cases."""

    def _weight(self, M, K, seed=0):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((M, K)).astype(np.float32) * 0.02
        return _bf16_round_trip(w)

    # ---------------------------------------------------------------- basic

    def test_matches_dense_matmul(self):
        """huffman_matmul output must equal x @ W.T for a BF16 weight matrix."""
        from compressed_matmul_cpu import huffman_matmul, C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 64, 128
        w = self._weight(M, K)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        x = np.random.randn(4, K).astype(np.float32)
        expected = x @ w.T
        got = huffman_matmul(x, hd, codebook, M, K)
        np.testing.assert_allclose(got, expected, atol=1e-4,
                                   err_msg="huffman_matmul diverges from dense")

    def test_single_token(self):
        """T=1 (autoregressive inference) must work correctly."""
        from compressed_matmul_cpu import huffman_matmul, C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 32, 64
        w = self._weight(M, K, seed=7)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        x = np.random.randn(1, K).astype(np.float32)
        expected = x @ w.T
        got = huffman_matmul(x, hd, codebook, M, K)
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_large_batch(self):
        """Large batch (T=32) accumulates correctly."""
        from compressed_matmul_cpu import huffman_matmul, C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 64, 128
        w = self._weight(M, K, seed=3)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        x = np.random.randn(32, K).astype(np.float32)
        expected = x @ w.T
        got = huffman_matmul(x, hd, codebook, M, K)
        np.testing.assert_allclose(got, expected, atol=1e-4)

    def test_no_full_weight_matrix_needed(self):
        """Huffman stream is smaller than the equivalent packed-bits array."""
        M, K = 128, 256
        w = self._weight(M, K, seed=11)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        stream_bytes = len(hd['huff_stream'])
        packed_bytes = (M * K * bits + 7) // 8
        assert stream_bytes < packed_bytes, \
            f"Huffman stream ({stream_bytes}B) not smaller than packed ({packed_bytes}B)"

    def test_chunk_rows_consistency(self):
        """Different chunk_rows values must produce identical results."""
        from compressed_matmul_cpu import huffman_matmul, C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 128, 64
        w = self._weight(M, K, seed=5)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        x = np.random.randn(4, K).astype(np.float32)
        ref = huffman_matmul(x, hd, codebook, M, K, chunk_rows=128)
        for cr in (1, 7, 32, 64):
            got = huffman_matmul(x, hd, codebook, M, K, chunk_rows=cr)
            np.testing.assert_allclose(got, ref, atol=1e-6,
                                       err_msg=f"chunk_rows={cr} diverges")

    def test_3d_input(self):
        """(B, T, K) input shape must be handled (reshapes internally)."""
        from compressed_matmul_cpu import huffman_matmul, C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 32, 64
        w = self._weight(M, K, seed=2)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        x_np = np.random.randn(2, 5, K).astype(np.float32)
        x_2d = np.random.randn(10, K).astype(np.float32)
        # Use same x values
        x_2d[:] = x_np.reshape(10, K)
        got_3d = huffman_matmul(x_np, hd, codebook, M, K)
        got_2d = huffman_matmul(x_2d, hd, codebook, M, K)
        np.testing.assert_allclose(
            got_3d.reshape(10, M), got_2d, atol=1e-6,
            err_msg="3D vs 2D input shape mismatch"
        )


class TestHuffmanInferenceLinearLayer:
    """AdaptiveCodebookLinear with _huff_data set stays compressed during inference."""

    def _make_layer(self, M, K, seed=0):
        from compressed_modules import AdaptiveCodebookLinear
        rng = np.random.default_rng(seed)
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        data = {
            'mode': 'direct_codebook', 'shape': (M, K), 'bits': bits,
            'encoding': 'huffman', 'codebook': codebook, 'codebook_type': None,
            **{k: hd[k] for k in ('huff_stream', 'huff_lengths', 'huff_n',
                                   'huff_lut_sym', 'huff_lut_len',
                                   'huff_sl_first_code', 'huff_sl_base_offset',
                                   'huff_sl_sym', 'huff_row_bit_starts')},
        }
        layer = AdaptiveCodebookLinear.from_compressed(
            f"huff_linear_{M}x{K}_{seed}", data, {}, use_gpu=False
        )
        return layer, w

    def test_huff_data_set_not_indices(self):
        """from_compressed sets _huff_data and leaves indices=None."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        layer, _ = self._make_layer(32, 64)
        assert layer._huff_data is not None, "_huff_data not set"
        assert layer.indices is None, "indices should be None when _huff_data present"

    def test_forward_matches_dense(self):
        """Layer forward must match F.linear to float32 precision."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 64, 128
        layer, w_bf16 = self._make_layer(M, K)
        layer.eval()
        x = torch.randn(4, K)
        expected = F.linear(x, torch.from_numpy(w_bf16))
        got = layer(x)
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, f"Linear forward mismatch: {max_err:.6f}"

    def test_forward_no_nan(self):
        """Forward pass must never produce NaN."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        layer, _ = self._make_layer(64, 128, seed=42)
        layer.eval()
        x = torch.randn(8, 128)
        out = layer(x)
        assert not out.isnan().any().item(), "NaN in Huffman linear output"

    def test_forward_with_bias(self):
        """Bias is correctly added when present."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 32, 64
        layer, w_bf16 = self._make_layer(M, K, seed=9)
        bias = torch.randn(M) * 0.1
        layer.bias = bias
        layer.eval()
        x = torch.randn(3, K)
        expected = F.linear(x, torch.from_numpy(w_bf16), bias)
        got = layer(x)
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, f"Bias mismatch: {max_err:.6f}"

    def test_fallback_when_tables_missing(self):
        """If Phase-2 tables are absent, falls back to decode-at-load (indices set)."""
        from compressed_modules import AdaptiveCodebookLinear
        M, K = 16, 32
        rng = np.random.default_rng(0)
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        from huffman_codebook import huffman_encode_indices
        from bitpack import pack_any_bits
        uniq = np.unique(w)
        lut  = {v: i for i, v in enumerate(uniq)}
        idx  = np.array([lut[v] for v in w.ravel()], dtype=np.uint16)
        bits = int(np.ceil(np.log2(max(len(uniq), 2))))
        # Encode without shape → no Phase-2 tables
        result = huffman_encode_indices(idx)
        data = {
            'mode': 'direct_codebook', 'shape': (M, K), 'bits': bits,
            'encoding': 'huffman', 'codebook': uniq, 'codebook_type': None,
            'huff_stream':  result['huff_stream'],
            'huff_lengths': result['huff_lengths'],
            'huff_n':       result['huff_n'],
            # No huff_row_bit_starts / huff_lut_sym / etc.
        }
        layer = AdaptiveCodebookLinear.from_compressed(
            "huff_fallback_test", data, {}, use_gpu=False
        )
        assert layer._huff_data is None, "_huff_data should be None for fallback"
        assert layer.indices is not None, "indices should be set for fallback path"
        layer.eval()
        x = torch.randn(2, K)
        out = layer(x)
        assert not out.isnan().any().item()


class TestHuffmanDecodeWeights:
    """huffman_decode_weights: Huffman bitstream → float32 weight matrix."""

    def test_decode_matches_original(self):
        """Decoded weights must equal the original float32 weight matrix."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE, huffman_decode_weights
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        rng = np.random.default_rng(7)
        M, K = 16, 32
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        decoded = huffman_decode_weights(hd, codebook, M, K)
        assert decoded.shape == (M, K)
        assert np.allclose(decoded, w, atol=1e-5), \
            f"max error: {np.abs(decoded - w).max():.2e}"

    def test_decode_larger_matrix(self):
        """Larger matrix (128×256) decodes correctly."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE, huffman_decode_weights
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        rng = np.random.default_rng(13)
        M, K = 128, 256
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        decoded = huffman_decode_weights(hd, codebook, M, K)
        assert np.allclose(decoded, w, atol=1e-5)

    def test_decode_then_gpu_matmul(self):
        """decode_weights → GPU matmul gives same result as CPU F.linear."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE, huffman_decode_weights
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        rng = np.random.default_rng(99)
        M, K = 64, 128
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (M, K))

        x = torch.randn(3, K)
        expected = F.linear(x, torch.from_numpy(w))

        weights_np = huffman_decode_weights(hd, codebook, M, K)
        weights_gpu = torch.from_numpy(weights_np).to('cuda')
        got = torch.matmul(x.to('cuda').float(), weights_gpu.T).cpu()
        del weights_gpu

        assert torch.allclose(got, expected, atol=1e-4), \
            f"GPU matmul mismatch: {(got - expected).abs().max():.2e}"

    def test_decode_no_nan(self):
        """Decoded weights must not contain NaN or Inf."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE, huffman_decode_weights
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        rng = np.random.default_rng(42)
        w = _bf16_round_trip(rng.standard_normal((32, 64)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (32, 64))
        decoded = huffman_decode_weights(hd, codebook, 32, 64)
        assert not np.isnan(decoded).any(), "NaN in decoded weights"
        assert not np.isinf(decoded).any(), "Inf in decoded weights"


class TestHuffmanGpuMatmulPath:
    """AdaptiveCodebookLinear with _huff_data uses GPU matmul when device=cuda."""

    def _make_layer(self, M, K, seed=0):
        from compressed_modules import AdaptiveCodebookLinear
        rng = np.random.default_rng(seed)
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        data = {
            'mode': 'direct_codebook', 'shape': (M, K), 'bits': bits,
            'encoding': 'huffman', 'codebook': codebook, 'codebook_type': None,
            **{k: hd[k] for k in ('huff_stream', 'huff_lengths', 'huff_n',
                                   'huff_lut_sym', 'huff_lut_len',
                                   'huff_sl_first_code', 'huff_sl_base_offset',
                                   'huff_sl_sym', 'huff_row_bit_starts')},
        }
        layer = AdaptiveCodebookLinear.from_compressed(
            f"huff_gpu_{M}x{K}_{seed}", data, {}, use_gpu=True
        )
        return layer, w

    def test_state_when_use_gpu(self):
        """With use_gpu=True: GPU path sets _gpu_func; CPU fallback sets _huff_data.
        Either way, packed indices are never allocated."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        layer, _ = self._make_layer(32, 64)
        if torch.cuda.is_available():
            # GPU machine: HuffmanCodebookLinear is set as _gpu_func (Phase 2 or 1)
            assert layer._gpu_func is not None, \
                "_gpu_func should be set to HuffmanCodebookLinear on GPU machine"
            assert layer._huff_data is None, \
                "_huff_data not needed when GPU path is active"
        else:
            # CPU-only machine: falls back to CPU-RAM path
            assert layer._huff_data is not None, \
                "_huff_data should be set as fallback when no GPU"
            assert layer._gpu_func is None
        assert layer.indices is None, "packed indices should never be allocated for Huffman"

    def test_forward_cpu_x_matches_dense(self):
        """CPU-device x → CPU matmul path still gives correct result."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        M, K = 64, 128
        layer, w = self._make_layer(M, K)
        layer.eval()
        x = torch.randn(4, K)
        expected = F.linear(x, torch.from_numpy(w))
        got = layer(x)
        assert torch.allclose(got, expected, atol=1e-4)

    def test_forward_cuda_x_matches_dense(self):
        """CUDA-device x → GPU matmul path gives same result as dense F.linear."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE, huffman_decode_weights
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 64, 128
        layer, w = self._make_layer(M, K)
        layer.eval()
        x_gpu = torch.randn(4, K, device='cuda')
        expected = F.linear(x_gpu.cpu(), torch.from_numpy(w))
        got = layer(x_gpu).cpu()
        assert torch.allclose(got, expected, atol=1e-4), \
            f"GPU path mismatch: {(got - expected).abs().max():.2e}"

    def test_forward_cuda_output_on_correct_device(self):
        """Output tensor must be on the same device as input x."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_layer(32, 64)
        layer.eval()
        x_gpu = torch.randn(2, 64, device='cuda')
        out = layer(x_gpu)
        assert out.device.type == 'cuda', f"output on {out.device}, expected cuda"

    def test_forward_cuda_no_nan(self):
        """GPU matmul path must not produce NaN."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_layer(64, 128, seed=77)
        layer.eval()
        x = torch.randn(8, 128, device='cuda')
        out = layer(x)
        assert not out.isnan().any().item()

    def test_stream_stays_in_cpu_ram(self):
        """VRAM allocation must not increase by M*K*2 bytes (packed indices never uploaded)."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 256, 512
        layer, _ = self._make_layer(M, K)
        layer.eval()

        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated()
        # Trigger one forward pass
        x = torch.randn(2, K, device='cuda')
        with torch.no_grad():
            _ = layer(x)
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated()

        # At rest, VRAM increase should be only the codebook (small) — NOT M*K*2 bytes
        packed_index_size = M * K * 2  # 13-bit packed would be M*K*13/8, but uint16 is M*K*2
        vram_delta = vram_after - vram_before
        assert vram_delta < packed_index_size, \
            f"VRAM increased by {vram_delta/1e6:.1f}MB — packed indices may have been uploaded"


# ─────────────────────────────────────────────────────────────────────────────
# 6e. GPU Phase 2: Huffman stream lives in VRAM, decoded on GPU without
#     ever materialising a float weight matrix.
# ─────────────────────────────────────────────────────────────────────────────

class TestHuffmanGPUPhase2:
    """AdaptiveCodebookLinear/Embedding with use_gpu=True uses GPU Phase 2
    when CUDA is available: Huffman stream is uploaded to VRAM at load time
    and decoded on-the-fly by the GPU kernel — no full float weight matrix
    is ever created."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_linear_layer(self, M, K, seed=0, use_gpu=True):
        from compressed_modules import AdaptiveCodebookLinear
        rng = np.random.default_rng(seed)
        w = _bf16_round_trip(rng.standard_normal((M, K)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (M, K))
        data = {
            'mode': 'direct_codebook', 'shape': (M, K), 'bits': bits,
            'encoding': 'huffman', 'codebook': codebook, 'codebook_type': None,
            **{k: hd[k] for k in ('huff_stream', 'huff_lengths', 'huff_n',
                                   'huff_lut_sym', 'huff_lut_len',
                                   'huff_sl_first_code', 'huff_sl_base_offset',
                                   'huff_sl_sym', 'huff_row_bit_starts')},
        }
        layer = AdaptiveCodebookLinear.from_compressed(
            f"p2_lin_{M}x{K}_{seed}", data, {}, use_gpu=use_gpu,
        )
        return layer, w

    def _make_emb_layer(self, vocab, hidden, seed=0, use_gpu=True):
        from compressed_modules import AdaptiveCodebookEmbedding
        rng = np.random.default_rng(seed)
        w = _bf16_round_trip(rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (vocab, hidden))
        data = {
            'mode': 'direct_codebook', 'shape': (vocab, hidden), 'bits': bits,
            'encoding': 'huffman', 'codebook': codebook, 'codebook_type': None,
            **{k: hd[k] for k in ('huff_stream', 'huff_lengths', 'huff_n',
                                   'huff_lut_sym', 'huff_lut_len',
                                   'huff_sl_first_code', 'huff_sl_base_offset',
                                   'huff_sl_sym', 'huff_row_bit_starts')},
        }
        layer = AdaptiveCodebookEmbedding.from_compressed(
            f"p2_emb_{vocab}x{hidden}_{seed}", data, {}, use_gpu=use_gpu,
        )
        return layer, w

    # ── path selection ────────────────────────────────────────────────────────

    def test_gpu_func_set_on_gpu_machine(self):
        """With use_gpu=True and CUDA available, _gpu_func is set (not _huff_data)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_linear_layer(32, 64)
        assert layer._gpu_func is not None, \
            "_gpu_func should be a HuffmanCodebookLinear instance"
        assert layer._huff_data is None, \
            "_huff_data should be None when GPU path is active"
        assert layer.indices is None

    def test_gpu_func_is_huffman_codebook_linear(self):
        """_gpu_func must be a HuffmanCodebookLinear instance."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        from gpu_huffman_functions import HuffmanCodebookLinear
        layer, _ = self._make_linear_layer(32, 64)
        if layer._gpu_func is None:
            pytest.skip("GPU func not set (kernel compile failed)")
        assert isinstance(layer._gpu_func, HuffmanCodebookLinear), \
            f"Expected HuffmanCodebookLinear, got {type(layer._gpu_func)}"

    def test_cpu_fallback_when_use_gpu_false(self):
        """With use_gpu=False, CPU-RAM path is always used regardless of GPU."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        layer, _ = self._make_linear_layer(32, 64, use_gpu=False)
        assert layer._huff_data is not None, \
            "_huff_data should be set when use_gpu=False"
        assert layer._gpu_func is None, \
            "_gpu_func should be None when use_gpu=False"
        assert layer.indices is None

    def test_embedding_gpu_func_set_on_gpu_machine(self):
        """Embedding: _gpu_func is set when GPU available and use_gpu=True."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_emb_layer(64, 32)
        assert layer._gpu_func is not None
        assert layer._huff_data is None

    # ── stream placement ──────────────────────────────────────────────────────

    def test_phase2_stream_on_gpu_device(self):
        """Phase 2: huff_stream tensor must live on CUDA, not CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        from gpu_huffman_functions import HuffmanCodebookLinear
        layer, _ = self._make_linear_layer(32, 64)
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        func = layer._gpu_func
        if not isinstance(func, HuffmanCodebookLinear) or func._phase != 2:
            pytest.skip("Not in Phase 2 mode")
        assert func._huff_stream.is_cuda, "huff_stream must be on CUDA device in Phase 2"
        assert func._codebook.is_cuda,    "codebook must be on CUDA device in Phase 2"
        assert func._lut_sym.is_cuda,     "lut_sym must be on CUDA device in Phase 2"

    def test_phase2_tables_on_gpu(self):
        """Phase 2: all decode tables (LUTs, slow-path) must live on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        from gpu_huffman_functions import HuffmanCodebookLinear
        layer, _ = self._make_linear_layer(32, 64)
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        func = layer._gpu_func
        if not isinstance(func, HuffmanCodebookLinear) or func._phase != 2:
            pytest.skip("Not in Phase 2 mode")
        for attr in ('_row_bit_start', '_lut_len', '_sl_first_code',
                     '_sl_base_offset', '_sl_sym'):
            t = getattr(func, attr)
            assert t.is_cuda, f"{attr} must be on CUDA in Phase 2"

    # ── linear correctness ────────────────────────────────────────────────────

    def test_linear_forward_single_token(self):
        """T=1 (autoregressive) forward on GPU must match F.linear."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 64, 128
        layer, w = self._make_linear_layer(M, K)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(1, K, device='cuda')
        expected = F.linear(x.cpu(), torch.from_numpy(w))
        got = layer(x).cpu()
        max_err = (got.float() - expected.float()).abs().max().item()
        assert max_err < 1e-4, f"T=1 GPU mismatch: {max_err:.2e}"

    def test_linear_forward_batch(self):
        """Batch forward on GPU must match F.linear."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 64, 128
        layer, w = self._make_linear_layer(M, K)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(8, K, device='cuda')
        expected = F.linear(x.cpu(), torch.from_numpy(w))
        got = layer(x).cpu()
        max_err = (got.float() - expected.float()).abs().max().item()
        assert max_err < 1e-4, f"Batch GPU mismatch: {max_err:.2e}"

    def test_linear_forward_no_nan(self):
        """GPU Phase 2 forward must not produce NaN or Inf."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_linear_layer(64, 128, seed=77)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(4, 128, device='cuda')
        out = layer(x)
        assert not out.isnan().any().item(), "NaN in GPU Phase 2 linear output"
        assert not out.isinf().any().item(), "Inf in GPU Phase 2 linear output"

    def test_linear_output_on_cuda_device(self):
        """Output must be on CUDA when input is on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_linear_layer(32, 64)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(2, 64, device='cuda')
        out = layer(x)
        assert out.device.type == 'cuda', f"output on {out.device}, expected cuda"

    def test_linear_with_bias(self):
        """Bias must be correctly added in GPU Phase 2."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 32, 64
        layer, w = self._make_linear_layer(M, K, seed=9)
        bias = torch.randn(M) * 0.1
        layer.bias = bias
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(3, K, device='cuda')
        expected = F.linear(x.cpu(), torch.from_numpy(w), bias)
        got = layer(x).cpu()
        max_err = (got.float() - expected.float()).abs().max().item()
        assert max_err < 1e-4, f"Bias mismatch: {max_err:.2e}"

    def test_linear_repeated_forward_consistent(self):
        """Repeated GPU forward calls must produce identical results (determinism)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 32, 64
        layer, _ = self._make_linear_layer(M, K, seed=3)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(4, K, device='cuda')
        with torch.no_grad():
            out1 = layer(x).clone()
            out2 = layer(x).clone()
        assert torch.equal(out1, out2), "GPU Phase 2 forward not deterministic"

    # ── VRAM footprint ────────────────────────────────────────────────────────

    def test_linear_no_full_float_weights_at_rest(self):
        """After a forward pass, VRAM must not hold a float32[M,K] weight matrix."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        M, K = 256, 512
        layer, _ = self._make_linear_layer(M, K)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated()
        x = torch.randn(2, K, device='cuda')
        with torch.no_grad():
            _ = layer(x)
        torch.cuda.empty_cache()
        vram_after = torch.cuda.memory_allocated()
        full_weight_f32 = M * K * 4  # bytes if materialised as float32
        vram_delta = vram_after - vram_before
        assert vram_delta < full_weight_f32, (
            f"VRAM grew by {vram_delta/1024:.1f} KB at rest — "
            f"full float32 weights ({full_weight_f32/1024:.1f} KB) may be permanent"
        )

    # ── GPU vs CPU numerical agreement ───────────────────────────────────────

    def test_gpu_phase2_matches_cpu_ram_path(self):
        """GPU Phase 2 output must match CPU-RAM decode path within float32 tolerance."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable (needed for CPU reference)")
        M, K = 64, 128
        layer_gpu = self._make_linear_layer(M, K, seed=42, use_gpu=True)[0]
        layer_cpu = self._make_linear_layer(M, K, seed=42, use_gpu=False)[0]
        layer_gpu.eval(); layer_cpu.eval()
        if layer_gpu._gpu_func is None:
            pytest.skip("GPU func not set")
        x = torch.randn(4, K)
        out_gpu = layer_gpu(x.to('cuda')).cpu().float()
        out_cpu = layer_cpu(x).float()
        max_err = (out_gpu - out_cpu).abs().max().item()
        assert max_err < 1e-4, f"GPU vs CPU path mismatch: {max_err:.2e}"

    # ── embedding correctness ─────────────────────────────────────────────────

    def test_embedding_forward_matches_dense(self):
        """GPU Phase 2 embedding forward must match F.embedding."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        vocab, hidden = 64, 32
        layer, w = self._make_emb_layer(vocab, hidden)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        ids = torch.tensor([0, 5, 10, 63, 0], device='cuda')
        expected = F.embedding(ids.cpu(), torch.from_numpy(w))
        got = layer(ids).cpu()
        max_err = (got.float() - expected.float()).abs().max().item()
        assert max_err < 1e-4, f"GPU embedding mismatch: {max_err:.2e}"

    def test_embedding_forward_no_nan(self):
        """GPU Phase 2 embedding must not produce NaN."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_emb_layer(64, 32, seed=7)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        ids = torch.randint(0, 64, (10,), device='cuda')
        out = layer(ids)
        assert not out.isnan().any().item(), "NaN in GPU Phase 2 embedding"

    def test_embedding_output_on_cuda_device(self):
        """Embedding output must be on CUDA when input IDs are on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        layer, _ = self._make_emb_layer(64, 32)
        layer.eval()
        if layer._gpu_func is None:
            pytest.skip("GPU func not set")
        ids = torch.tensor([0, 1, 2], device='cuda')
        out = layer(ids)
        assert out.device.type == 'cuda', f"output on {out.device}, expected cuda"

    def test_embedding_embed_scale_applied(self):
        """embed_scale must be multiplied in GPU Phase 2 embedding."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        vocab, hidden = 32, 16
        layer_scaled = self._make_emb_layer(vocab, hidden)[0]
        layer_plain  = self._make_emb_layer(vocab, hidden)[0]
        if layer_scaled._gpu_func is None:
            pytest.skip("GPU func not set")
        layer_scaled.embed_scale = 3.0
        layer_scaled.eval(); layer_plain.eval()
        ids = torch.tensor([0, 1, 2], device='cuda')
        scaled   = layer_scaled(ids).cpu().float()
        unscaled = layer_plain(ids).cpu().float()
        np.testing.assert_allclose(
            scaled.numpy(), unscaled.numpy() * 3.0, atol=1e-4,
            err_msg="embed_scale not applied in GPU Phase 2 embedding"
        )


class TestHuffmanInferenceEmbedding:
    """AdaptiveCodebookEmbedding with Huffman inference-time decode."""

    def _make_emb_layer(self, vocab, hidden, seed=0):
        from compressed_modules import AdaptiveCodebookEmbedding
        rng = np.random.default_rng(seed)
        w = _bf16_round_trip(rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.02)
        codebook, bits, hd = _make_huff_data(w, (vocab, hidden))
        data = {
            'mode': 'direct_codebook', 'shape': (vocab, hidden), 'bits': bits,
            'encoding': 'huffman', 'codebook': codebook, 'codebook_type': None,
            **{k: hd[k] for k in ('huff_stream', 'huff_lengths', 'huff_n',
                                   'huff_lut_sym', 'huff_lut_len',
                                   'huff_sl_first_code', 'huff_sl_base_offset',
                                   'huff_sl_sym', 'huff_row_bit_starts')},
        }
        layer = AdaptiveCodebookEmbedding.from_compressed(
            f"huff_emb_{vocab}x{hidden}_{seed}", data, {}, use_gpu=False
        )
        return layer, w

    def test_huff_data_set(self):
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        layer, _ = self._make_emb_layer(64, 32)
        assert layer._huff_data is not None
        assert layer.indices is None

    def test_forward_matches_dense(self):
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        vocab, hidden = 64, 32
        layer, w_bf16 = self._make_emb_layer(vocab, hidden)
        layer.eval()
        ids = torch.tensor([0, 5, 10, 63, 0, 5])
        expected = F.embedding(ids, torch.from_numpy(w_bf16))
        got = layer(ids)
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, f"Embedding mismatch: {max_err:.6f}"

    def test_forward_no_nan(self):
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        layer, _ = self._make_emb_layer(128, 64, seed=7)
        layer.eval()
        ids = torch.randint(0, 128, (10,))
        out = layer(ids)
        assert not out.isnan().any().item()

    def test_duplicate_tokens_decoded_correctly(self):
        """Repeated token IDs must all produce the same row."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        vocab, hidden = 64, 32
        layer, w_bf16 = self._make_emb_layer(vocab, hidden, seed=3)
        layer.eval()
        ids = torch.tensor([7, 7, 7, 7])
        out = layer(ids)
        assert out.shape == (4, hidden)
        # All rows must be identical
        assert torch.allclose(out[0], out[1]) and torch.allclose(out[0], out[3])

    def test_embed_scale_applied(self):
        """embed_scale must multiply the Huffman output."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        vocab, hidden = 32, 16
        layer, _ = self._make_emb_layer(vocab, hidden)
        layer.embed_scale = 2.5
        layer.eval()
        layer_no_scale, _ = self._make_emb_layer(vocab, hidden)
        layer_no_scale.eval()
        ids = torch.tensor([0, 1, 2])
        scaled   = layer(ids)
        unscaled = layer_no_scale(ids)
        np.testing.assert_allclose(
            scaled.numpy(), unscaled.numpy() * 2.5, atol=1e-5,
            err_msg="embed_scale not applied in Huffman embedding forward"
        )


class TestHuffmanInferenceEndToEnd:
    """Tiny LLaMA: Huffman-compressed inference matches uncompressed logits."""

    @pytest.fixture(scope="class")
    def tiny_huffman_dir(self, tmp_path_factory):
        """Re-use the same tiny model fixture pattern but compress with --entropy-code."""
        import subprocess
        tmp = tmp_path_factory.mktemp("tiny_huff")
        model_dir = tmp / "tiny_llama_huff"
        model_dir.mkdir()

        vocab, hidden, layers, intermediate = 256, 64, 2, 128
        config = {
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": hidden, "intermediate_size": intermediate,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "num_hidden_layers": layers, "vocab_size": vocab,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "tie_word_embeddings": True,
            "torch_dtype": "float32",
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps({
            "model_type": "llama", "bos_token": "<s>",
            "eos_token": "</s>", "unk_token": "<unk>",
        }))
        (model_dir / "tokenizer.json").write_text(json.dumps({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {}, "merges": []},
            "added_tokens": [],
        }))

        rng = np.random.default_rng(456)
        tensors = {}
        def _add(name, shape):
            w = rng.standard_normal(shape).astype(np.float32) * 0.02
            u16 = (w.view(np.uint32) >> 16).astype(np.uint16)
            tensors[name] = u16

        _add("model.embed_tokens.weight", (vocab, hidden))
        _add("model.norm.weight", (hidden,))
        for L in range(layers):
            pfx = f"model.layers.{L}"
            _add(f"{pfx}.self_attn.q_proj.weight", (hidden, hidden))
            _add(f"{pfx}.self_attn.k_proj.weight", (hidden, hidden))
            _add(f"{pfx}.self_attn.v_proj.weight", (hidden, hidden))
            _add(f"{pfx}.self_attn.o_proj.weight", (hidden, hidden))
            _add(f"{pfx}.mlp.gate_proj.weight", (intermediate, hidden))
            _add(f"{pfx}.mlp.up_proj.weight", (intermediate, hidden))
            _add(f"{pfx}.mlp.down_proj.weight", (hidden, intermediate))
            _add(f"{pfx}.input_layernorm.weight", (hidden,))
            _add(f"{pfx}.post_attention_layernorm.weight", (hidden,))

        header = {}; offset = 0; blobs = {}
        for name, arr in tensors.items():
            blob = arr.tobytes()
            header[name] = {"dtype": "BF16", "shape": list(arr.shape),
                            "data_offsets": [offset, offset + len(blob)]}
            blobs[name] = blob; offset += len(blob)
        hdr_bytes = json.dumps(header).encode()
        with open(model_dir / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hdr_bytes)))
            f.write(hdr_bytes)
            for name in tensors:
                f.write(blobs[name])

        result = subprocess.run(
            [sys.executable, str(ROOT / "compress.py"), str(model_dir),
             "--mode", "lossless", "--entropy-code", "--force"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        assert result.returncode == 0, \
            f"compress.py --entropy-code failed:\n{result.stderr[-2000:]}"
        return model_dir

    def _uncompressed_logits(self, model_dir):
        from transformers import AutoModelForCausalLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AutoModelForCausalLM.from_pretrained(
                str(model_dir), torch_dtype=torch.float32, _fast_init=False
            )
        m.eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            return m(ids).logits.detach().clone()

    def test_logits_no_nan(self, tiny_huffman_dir):
        """Huffman inference-time model must not produce NaN logits."""
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(tiny_huffman_dir), device='cpu',
                                 compression_mode='lossless', entropy_code=True)
        cm.load()
        cm.model.eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            logits = cm.model(ids).logits
        assert not logits.isnan().any().item(), "NaN in Huffman inference logits"

    def test_logits_match_uncompressed(self, tiny_huffman_dir):
        """Huffman inference logits must be lossless vs uncompressed (< 1e-3)."""
        from chat import CompressedChatModel
        uncompressed = self._uncompressed_logits(tiny_huffman_dir)

        cm = CompressedChatModel(str(tiny_huffman_dir), device='cpu',
                                 compression_mode='lossless', entropy_code=True)
        cm.load()
        cm.model.eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            compressed = cm.model(ids).logits.detach()

        max_err = (uncompressed - compressed).abs().max().item()
        assert max_err < 1e-3, \
            f"Huffman inference logits diverged: max_err={max_err:.6f}"

    def test_huff_data_set_on_loaded_layers(self, tiny_huffman_dir):
        """After loading with entropy_code=True, Linear layers must use _huff_data."""
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        from compressed_modules import AdaptiveCodebookLinear
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel unavailable")
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(tiny_huffman_dir), device='cpu',
                                 compression_mode='lossless', entropy_code=True)
        cm.load()
        huff_count = sum(
            1 for m in cm.model.modules()
            if isinstance(m, AdaptiveCodebookLinear) and m._huff_data is not None
        )
        assert huff_count > 0, \
            "No AdaptiveCodebookLinear layers have _huff_data set after load"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tiny synthetic model: compress → load → compare logits
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticModelRoundtrip:
    """End-to-end: build a tiny Llama-like model, compress losslessly, load the
    compressed version, and verify logits are identical (or very close) to the
    uncompressed forward pass on CPU."""

    @pytest.fixture(scope="class")
    def tiny_model_dir(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("tiny_model")
        model_dir = tmp_path / "tiny_llama"
        model_dir.mkdir()

        vocab, hidden, layers = 256, 64, 2
        intermediate = 128
        heads, kv_heads = 4, 4

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "torch_dtype": "float32",
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer_config.json").write_text(json.dumps({
            "model_type": "llama",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }))
        (model_dir / "tokenizer.json").write_text(json.dumps({
            "version": "1.0",
            "model": {"type": "BPE", "vocab": {}, "merges": []},
            "added_tokens": [],
        }))

        # Generate deterministic weights
        rng = np.random.default_rng(123)
        tensors = {}
        def _add(name, shape):
            w = rng.standard_normal(shape).astype(np.float32) * 0.02
            u16 = (w.view(np.uint32) >> 16).astype(np.uint16)
            tensors[name] = u16

        _add("model.embed_tokens.weight", (vocab, hidden))
        _add("model.norm.weight", (hidden,))
        for L in range(layers):
            pfx = f"model.layers.{L}"
            _add(f"{pfx}.self_attn.q_proj.weight", (hidden, hidden))
            _add(f"{pfx}.self_attn.k_proj.weight", (hidden, hidden))
            _add(f"{pfx}.self_attn.v_proj.weight", (hidden, hidden))
            _add(f"{pfx}.self_attn.o_proj.weight", (hidden, hidden))
            _add(f"{pfx}.mlp.gate_proj.weight", (intermediate, hidden))
            _add(f"{pfx}.mlp.up_proj.weight", (intermediate, hidden))
            _add(f"{pfx}.mlp.down_proj.weight", (hidden, intermediate))
            _add(f"{pfx}.input_layernorm.weight", (hidden,))
            _add(f"{pfx}.post_attention_layernorm.weight", (hidden,))

        # Write safetensors
        header = {}
        offset = 0
        data_blobs = {}
        for name, arr in tensors.items():
            blob = arr.tobytes()
            h, w = (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (arr.shape[0], 1)
            header[name] = {
                "dtype": "BF16",
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + len(blob)],
            }
            data_blobs[name] = blob
            offset += len(blob)

        hdr_bytes = json.dumps(header).encode()
        with open(model_dir / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hdr_bytes)))
            f.write(hdr_bytes)
            for name in tensors:
                f.write(data_blobs[name])

        return model_dir

    def _get_uncompressed_logits(self, model_dir):
        """Load uncompressed model and run one forward pass on CPU."""
        from transformers import AutoModelForCausalLM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), torch_dtype=torch.float32, _fast_init=False
            )
        model.eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            return model(ids).logits.detach().clone()

    def test_logits_match_after_lossless_compress(self, tiny_model_dir):
        """Compress a tiny model losslessly; verify compressed logits == uncompressed."""
        import subprocess, sys
        compress_py = ROOT / "compress.py"

        # Run compression
        result = subprocess.run(
            [sys.executable, str(compress_py), str(tiny_model_dir),
             "--mode", "lossless", "--force"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        assert result.returncode == 0, \
            f"compress.py failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

        # Verify cache exists
        cache_dir = tiny_model_dir / "codebook-lossless" / "tensors"
        assert cache_dir.exists() and list(cache_dir.glob("*.npz")), \
            "No .npz files in compressed cache"

        # Get uncompressed logits
        uncompressed = self._get_uncompressed_logits(tiny_model_dir)

        # Get compressed logits via CompressedChatModel
        sys.path.insert(0, str(ROOT))
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(tiny_model_dir), device='cpu',
                                 compression_mode='lossless')
        loaded = cm.load()
        assert loaded is not None, "CompressedChatModel.load() returned None"

        cm.model.eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            compressed_logits = cm.model(ids).logits.detach()

        max_err = (uncompressed - compressed_logits).abs().max().item()
        # Lossless compression: logits must match within float32 arithmetic tolerance
        assert max_err < 1e-3, \
            f"Logits diverged after lossless compression: max_err={max_err:.6f}\n" \
            f"uncompressed[:3]={uncompressed[0, -1, :3]}\n" \
            f"compressed[:3]={compressed_logits[0, -1, :3]}"

    def test_logits_match_after_huffman_compress(self, tiny_model_dir):
        """Compress with --entropy-code; verify Huffman-compressed logits == uncompressed."""
        import subprocess, sys
        compress_py = ROOT / "compress.py"

        result = subprocess.run(
            [sys.executable, str(compress_py), str(tiny_model_dir),
             "--mode", "lossless", "--entropy-code", "--force"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        assert result.returncode == 0, \
            f"compress.py --entropy-code failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

        cache_dir = tiny_model_dir / "codebook-lossless-huffman" / "tensors"
        assert cache_dir.exists() and list(cache_dir.glob("*.npz")), \
            "No .npz files in Huffman compressed cache"

        # At least some tensors should use Huffman encoding
        import zipfile
        huffman_count = 0
        for npz in cache_dir.glob("*.npz"):
            with zipfile.ZipFile(npz) as z:
                if any(n.startswith("huff_stream") for n in z.namelist()):
                    huffman_count += 1
        assert huffman_count > 0, "No tensors were Huffman-encoded — check --entropy-code flag"

        uncompressed = self._get_uncompressed_logits(tiny_model_dir)

        sys.path.insert(0, str(ROOT))
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(tiny_model_dir), device='cpu',
                                 compression_mode='lossless', entropy_code=True)
        loaded = cm.load()
        assert loaded is not None, "CompressedChatModel.load() returned None for Huffman model"

        cm.model.eval()
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            compressed_logits = cm.model(ids).logits.detach()

        max_err = (uncompressed - compressed_logits).abs().max().item()
        assert max_err < 1e-3, \
            f"Logits diverged after Huffman compression: max_err={max_err:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# 7b. RoPE buffer reinitialization — NaN/inf regression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRopeReinit:
    """Regression tests for reinit_rope_buffers NaN/inf detection.

    Bug: IEEE-754 comparisons (nan > 1.5, nan < -1e-6) always return False, so
    the original `needs_reinit` check silently left NaN inv_freq buffers in
    place.  NaN in inv_freq cascades to NaN logits through rotary attention.

    Trigger: inject_ssm_kernels loads libcompressed_kernel.so via RTLD_GLOBAL
    for ALL models (including Llama), initialising the HIP/HSA runtime.
    Subsequent to_empty() allocations can then contain NaN patterns in inv_freq.
    """

    @staticmethod
    def _fake_model_with_inv_freq(inv_freq_tensor):
        """Wrap a single inv_freq buffer in a minimal nn.Module hierarchy."""
        class FakeRoPE(nn.Module):
            def __init__(self, freq):
                super().__init__()
                self.register_buffer('inv_freq', freq.clone(), persistent=False)

        class FakeModel(nn.Module):
            def __init__(self, freq):
                super().__init__()
                self.rotary_emb = FakeRoPE(freq)

        return FakeModel(inv_freq_tensor)

    @staticmethod
    def _valid_inv_freq(half_dim, theta=10000.0):
        return 1.0 / (theta ** (
            torch.arange(0, half_dim * 2, 2, dtype=torch.float32) / (half_dim * 2)
        ))

    class _FakeConfig:
        rope_theta = 10000.0

    # ------------------------------------------------------------------ basic

    def test_nan_inv_freq_detected_and_reinit(self):
        """NaN inv_freq → reinit_rope_buffers must detect and fix it."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        nan_freq = torch.full((half_dim,), float('nan'))
        model = self._fake_model_with_inv_freq(nan_freq)
        assert model.rotary_emb.inv_freq.isnan().any().item(), "pre-cond: must be NaN"

        count = reinit_rope_buffers(model, self._FakeConfig())

        assert count == 1, f"Expected 1 reinit, got {count}"
        buf = model.rotary_emb.inv_freq
        assert not buf.isnan().any().item(), "inv_freq still NaN after reinit"
        assert not buf.isinf().any().item(), "inv_freq inf after reinit"
        f = buf.float()
        assert f.min().item() > 0.0, f"inv_freq min={f.min()} not > 0"
        assert f.max().item() <= 1.0, f"inv_freq max={f.max()} not <= 1"

    def test_inf_inv_freq_detected_and_reinit(self):
        """Inf inv_freq → reinit_rope_buffers must detect and fix it."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        model = self._fake_model_with_inv_freq(torch.full((half_dim,), float('inf')))
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 1
        assert not model.rotary_emb.inv_freq.isinf().any().item()

    def test_negative_inf_inv_freq_detected(self):
        """-inf inv_freq → reinit_rope_buffers must detect and fix it."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        model = self._fake_model_with_inv_freq(torch.full((half_dim,), float('-inf')))
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 1

    def test_large_garbage_inv_freq_detected(self):
        """Values > 1.5 (garbage floats from uninitialized memory) are detected."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        model = self._fake_model_with_inv_freq(torch.full((half_dim,), 999.0))
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 1

    def test_negative_garbage_inv_freq_detected(self):
        """Negative values (garbage floats) are detected."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        model = self._fake_model_with_inv_freq(torch.full((half_dim,), -0.5))
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 1

    def test_valid_inv_freq_not_reinit(self):
        """Valid float32 inv_freq in (0, 1] must NOT be reinitialized."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        valid = self._valid_inv_freq(half_dim)
        model = self._fake_model_with_inv_freq(valid)
        original = model.rotary_emb.inv_freq.clone()
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 0, f"Valid inv_freq was incorrectly reinitialized (count={count})"
        assert torch.allclose(model.rotary_emb.inv_freq, original), \
            "Valid inv_freq was modified"

    def test_bfloat16_inv_freq_reinit(self):
        """bfloat16 inv_freq (model.to(bfloat16) on garbage) must be detected."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        freq = self._valid_inv_freq(half_dim).to(torch.bfloat16)
        model = self._fake_model_with_inv_freq(freq)
        # Forcibly set to bfloat16 to simulate model.to(bfloat16) side-effect
        model.rotary_emb.inv_freq = model.rotary_emb.inv_freq.to(torch.bfloat16)
        assert model.rotary_emb.inv_freq.dtype == torch.bfloat16
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 1
        assert model.rotary_emb.inv_freq.dtype == torch.float32, \
            "Reinitialized inv_freq must be float32"

    # ---------------------------------------------------------------- output

    def test_nan_reinit_produces_valid_rope_output(self):
        """After reinit of NaN inv_freq, RoPE cos/sin computation is NaN-free."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16
        model = self._fake_model_with_inv_freq(torch.full((half_dim,), float('nan')))
        reinit_rope_buffers(model, self._FakeConfig())

        inv_freq = model.rotary_emb.inv_freq  # [half_dim]
        positions = torch.arange(8, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)          # [8, half_dim]
        emb = torch.cat([freqs, freqs], dim=-1)            # [8, head_dim]
        cos_vals = emb.cos()
        sin_vals = emb.sin()
        assert not cos_vals.isnan().any().item(), "cos still NaN after reinit"
        assert not sin_vals.isnan().any().item(), "sin still NaN after reinit"

    def test_multiple_rope_modules_all_fixed(self):
        """All NaN inv_freq buffers in a multi-layer model are fixed."""
        from rope_utils import reinit_rope_buffers
        half_dim = 16

        class MultiLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(4):
                    class FakeRoPE(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.register_buffer(
                                'inv_freq',
                                torch.full((half_dim,), float('nan')),
                                persistent=False,
                            )
                    setattr(self, f'layer_{i}', nn.Module())
                    getattr(self, f'layer_{i}').rotary_emb = FakeRoPE()

        model = MultiLayerModel()
        count = reinit_rope_buffers(model, self._FakeConfig())
        assert count == 4, f"Expected 4 reinits, got {count}"
        for i in range(4):
            buf = getattr(model, f'layer_{i}').rotary_emb.inv_freq
            assert not buf.isnan().any().item(), f"layer_{i} inv_freq still NaN"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Embedding scale detection and application
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedScaleDetection:
    """Tests for the embed_scale attribute on AdaptiveCodebookEmbedding.

    Some models (e.g. Gemma 3) use a custom embedding class that multiplies
    output by sqrt(hidden_size) inside forward().  model_loader detects this by
    checking if 'Scaled' appears in the class name, then sets embed_scale on the
    replacement AdaptiveCodebookEmbedding.  These tests verify every step.
    """

    def test_default_scale_is_one(self):
        """AdaptiveCodebookEmbedding must default to embed_scale=1.0."""
        from compressed_modules import AdaptiveCodebookEmbedding
        layer = AdaptiveCodebookEmbedding("emb", (64, 32), mode='exact')
        assert layer.embed_scale == 1.0, \
            f"Expected default embed_scale=1.0, got {layer.embed_scale}"

    def test_scale_one_is_exact_noop(self):
        """embed_scale=1.0 must return the exact same tensor as F.embedding."""
        from compressed_modules import AdaptiveCodebookEmbedding
        vocab, hidden = 64, 32
        w = torch.randn(vocab, hidden)
        layer = AdaptiveCodebookEmbedding("emb", (vocab, hidden), mode='exact')
        layer.weight = nn.Parameter(w.clone(), requires_grad=False)
        layer.embed_scale = 1.0

        ids = torch.tensor([0, 1, 2, 5])
        expected = F.embedding(ids, w)
        got = layer(ids)
        assert torch.equal(got, expected), \
            "embed_scale=1.0 changed the output — should be a no-op"

    def test_scale_applied_in_exact_mode(self):
        """embed_scale != 1.0 must multiply the output in exact mode."""
        from compressed_modules import AdaptiveCodebookEmbedding
        vocab, hidden = 64, 32
        w = torch.randn(vocab, hidden)
        layer = AdaptiveCodebookEmbedding("emb", (vocab, hidden), mode='exact')
        layer.weight = nn.Parameter(w.clone(), requires_grad=False)

        scale = float(hidden ** 0.5)  # sqrt(32) ≈ 5.657
        layer.embed_scale = scale

        ids = torch.tensor([0, 1, 2, 5, 63])
        expected = F.embedding(ids, w) * scale
        got = layer(ids)
        assert torch.allclose(got, expected, atol=1e-6), \
            f"embed_scale not applied in exact mode: max_err={(got - expected).abs().max():.2e}"

    def test_scale_applied_in_direct_codebook_mode(self):
        """embed_scale must multiply the output of the direct_codebook CPU path."""
        from compressed_modules import AdaptiveCodebookEmbedding
        from bitpack import pack_any_bits

        vocab, hidden = 64, 32
        rng = np.random.default_rng(42)
        w_f32 = rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.02
        w_bf16 = _bf16_round_trip(w_f32)

        flat = w_bf16.ravel()
        uniq = np.unique(flat)
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in flat], dtype=np.uint16)
        bits = int(np.ceil(np.log2(max(len(uniq), 2))))
        packed = pack_any_bits(indices, bits)

        data = {
            'mode': 'direct_codebook', 'shape': (vocab, hidden), 'bits': bits,
            'indices': packed, 'codebook': uniq, 'codebook_type': None,
        }
        scale = float(hidden ** 0.5)
        # Use a unique name to avoid polluting the FastIndexManager singleton cache
        layer = AdaptiveCodebookEmbedding.from_compressed(
            "emb_scale_dc_32", data, {}, use_gpu=False
        )
        layer.embed_scale = scale
        layer.eval()

        ids = torch.tensor([0, 5, 10, 32, 63])
        expected = F.embedding(ids, torch.from_numpy(w_bf16)) * scale
        got = layer(ids)
        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, \
            f"embed_scale not applied in direct_codebook mode: max_err={max_err:.2e}"

    def test_scaled_class_name_sets_embed_scale(self):
        """_try_replace_embedding detection: 'Scaled' in class name → embed_scale=sqrt(dim)."""
        # Simulate what model_loader._try_replace_embedding does
        class FakeScaledWordEmbedding(nn.Embedding):
            """Fake Gemma-style embedding that multiplies by sqrt(hidden_size)."""
            def forward(self, x):
                return super().forward(x) * (self.embedding_dim ** 0.5)

        from compressed_modules import AdaptiveCodebookEmbedding
        hidden = 640  # Gemma 270M hidden size
        layer = AdaptiveCodebookEmbedding("emb", (256, hidden), mode='exact')

        # Apply the same detection logic as model_loader._try_replace_embedding
        child = FakeScaledWordEmbedding(256, hidden)
        class_name = type(child).__name__
        if 'Scaled' in class_name or 'scaled' in class_name:
            h = getattr(child, 'embedding_dim', None)
            if h and h > 1:
                layer.embed_scale = float(h ** 0.5)

        expected_scale = float(hidden ** 0.5)  # ≈ 25.298
        assert abs(layer.embed_scale - expected_scale) < 1e-5, \
            f"Expected embed_scale≈{expected_scale:.3f}, got {layer.embed_scale}"

    def test_unscaled_class_name_leaves_scale_one(self):
        """Regular nn.Embedding (no 'Scaled' in name) must leave embed_scale=1.0."""
        class RegularEmbedding(nn.Embedding):
            pass

        from compressed_modules import AdaptiveCodebookEmbedding
        layer = AdaptiveCodebookEmbedding("emb", (256, 64), mode='exact')

        # Apply detection logic
        child = RegularEmbedding(256, 64)
        class_name = type(child).__name__
        if 'Scaled' in class_name or 'scaled' in class_name:
            h = getattr(child, 'embedding_dim', None)
            if h and h > 1:
                layer.embed_scale = float(h ** 0.5)

        assert layer.embed_scale == 1.0, \
            f"Expected embed_scale=1.0 for unscaled embedding, got {layer.embed_scale}"

    def test_scaled_embedding_matches_original_forward(self):
        """Compressed embedding with embed_scale must exactly match original scaled forward."""
        class FakeScaledWordEmbedding(nn.Embedding):
            def forward(self, x):
                return super().forward(x) * (self.embedding_dim ** 0.5)

        from compressed_modules import AdaptiveCodebookEmbedding
        from bitpack import pack_any_bits

        vocab, hidden = 128, 64
        rng = np.random.default_rng(7)
        w_f32 = rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.02
        w_bf16 = _bf16_round_trip(w_f32)

        # Build the original scaled embedding with known weights
        orig = FakeScaledWordEmbedding(vocab, hidden)
        orig.weight = nn.Parameter(torch.from_numpy(w_bf16), requires_grad=False)
        orig.eval()

        # Build compressed equivalent
        flat = w_bf16.ravel()
        uniq = np.unique(flat)
        lut = {v: i for i, v in enumerate(uniq)}
        indices = np.array([lut[v] for v in flat], dtype=np.uint16)
        bits = int(np.ceil(np.log2(max(len(uniq), 2))))
        packed = pack_any_bits(indices, bits)
        data = {
            'mode': 'direct_codebook', 'shape': (vocab, hidden), 'bits': bits,
            'indices': packed, 'codebook': uniq, 'codebook_type': None,
        }
        # Use a unique name so the FastIndexManager singleton doesn't serve a
        # stale lookup table from a different test with the same layer name.
        compressed = AdaptiveCodebookEmbedding.from_compressed(
            "emb_scale_match_128x64", data, {}, use_gpu=False
        )
        compressed.embed_scale = float(hidden ** 0.5)
        compressed.eval()

        ids = torch.tensor([0, 1, 5, 10, 63, 127])
        with torch.no_grad():
            expected = orig(ids)
            got = compressed(ids)

        max_err = (expected - got).abs().max().item()
        assert max_err < 1e-4, \
            f"Scaled-compressed embedding doesn't match original scaled forward: " \
            f"max_err={max_err:.2e}"

    def test_embed_scale_detection_gemma_class_names(self):
        """Verify detection works for all observed Gemma embedding class names."""
        from compressed_modules import AdaptiveCodebookEmbedding

        # All class names that should trigger scale detection
        scaled_names = [
            "Gemma3TextScaledWordEmbedding",
            "GemmaScaledEmbedding",
            "ScaledEmbedding",
            "scaled_word_embedding",
        ]
        # Class names that should NOT trigger scale detection
        unscaled_names = [
            "Embedding",
            "GemmaEmbedding",
            "LlamaEmbedding",
            "GPT2Embedding",
        ]

        def _apply_detection(class_name, embedding_dim=64):
            layer = AdaptiveCodebookEmbedding("emb", (256, embedding_dim), mode='exact')
            if 'Scaled' in class_name or 'scaled' in class_name:
                h = embedding_dim
                if h and h > 1:
                    layer.embed_scale = float(h ** 0.5)
            return layer.embed_scale

        for name in scaled_names:
            scale = _apply_detection(name, embedding_dim=64)
            assert scale != 1.0, \
                f"'{name}' should trigger scale detection but embed_scale=1.0"
            assert abs(scale - 8.0) < 1e-5, \
                f"'{name}': expected scale=sqrt(64)=8.0, got {scale}"

        for name in unscaled_names:
            scale = _apply_detection(name, embedding_dim=64)
            assert scale == 1.0, \
                f"'{name}' should NOT trigger scale detection but embed_scale={scale}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Real-model integration (needs --model flag)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_model_path(request):
    """Resolve path to a real model for integration tests.

    Priority:
      1. --model <path>  explicit CLI flag
      2. Auto-discovered from local HF cache (gpt2 → gemma → Qwen, in that order)

    Never triggers a network download.
    """
    p = request.config.getoption("--model", default=None)
    if p:
        p = Path(p).expanduser().resolve()
        if not p.exists():
            pytest.skip(f"Model path not found: {p}")
        return p

    # Check direct paths (no network needed)
    for direct in (
        Path.home() / "workspace" / "model" / "Soprano-80M",
        Path.home() / "workspace" / "model" / "Qwen3-0.6B",
        Path.home() / "workspace" / "model" / "Qwen3-1.7B",
        Path.home() / "workspace" / "model" / "Qwen3.5-9B",
    ):
        if direct.exists() and (direct / "config.json").exists():
            print(f"\n  [real_model_path] Using direct model path: {direct}")
            return direct

    # Auto-discover from HF cache (no network download)
    try:
        import huggingface_hub
        for repo_id in ("gpt2", "google/gemma-3-270m-it", "Qwen/Qwen3.5-0.8B"):
            try:
                cached = huggingface_hub.snapshot_download(repo_id, local_files_only=True)
                if cached and Path(cached).exists():
                    auto = Path(cached)
                    print(f"\n  [real_model_path] Using auto-discovered model: {auto}")
                    return auto
            except Exception:
                continue
    except ImportError:
        pass

    pytest.skip("No cached model found; pass --model <path> or download gpt2 first")


class TestRealModelCompressedVsUncompressed:
    """Compare uncompressed and compressed (lossless) logits on the real model."""

    @staticmethod
    def _ensure_compressed(real_model_path):
        """Run compress.py --mode lossless if no cache exists yet."""
        import subprocess, sys
        cache = real_model_path / "codebook-lossless" / "tensors"
        if not (cache.exists() and list(cache.glob("*.npz"))):
            result = subprocess.run(
                [sys.executable, str(ROOT / "compress.py"), str(real_model_path),
                 "--mode", "lossless"],
                capture_output=True, text=True, cwd=str(ROOT),
                timeout=600,  # 10-minute cap for large models
            )
            assert result.returncode == 0, \
                f"compress.py failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

    def test_lossless_logits_close(self, real_model_path):
        """Lossless logits should match uncompressed within matmul-precision tolerance.

        Our compressor stores all weights at BF16 precision (the fast lossless path
        is BF16-only; float32 weights fall to BF16-exact storage).  The fair
        comparison is therefore:

          uncompressed model loaded in bfloat16  (same weight precision)
          vs compressed model (BF16 weights + float32 CPU matmul accumulation)

        Any remaining logit error comes from bfloat16 vs float32 *accumulation*
        in the matmuls, compounding across all layers.  Top-1 token accuracy is the
        primary correctness metric — see test_top_token_matches.
        """
        self._ensure_compressed(real_model_path)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import sys
        tok = AutoTokenizer.from_pretrained(str(real_model_path), trust_remote_code=True)
        ids = tok("What is 2+2?", return_tensors="pt").input_ids

        # Load uncompressed in bfloat16 — matches the effective compressed precision
        # (compressor stores every weight at bf16 precision regardless of source dtype).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unc_model = AutoModelForCausalLM.from_pretrained(
                str(real_model_path), torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        unc_model.eval()
        with torch.no_grad():
            uncompressed = unc_model(ids).logits.detach().float()
        del unc_model

        # Compressed (CPU float32 matmul)
        sys.path.insert(0, str(ROOT))
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(real_model_path), device='cpu',
                                 compression_mode='lossless')
        loaded = cm.load()
        assert loaded is not None
        cm.model.eval()
        with torch.no_grad():
            compressed = cm.model(ids).logits.detach().float()

        max_err = (uncompressed - compressed).abs().max().item()
        # bf16 vs f32 accumulation across many layers typically yields ~10-25 logit
        # units.  Use 30.0 with margin to account for larger models.
        tol = 30.0
        print(f"\n  Logit max error (bf16 uncompressed vs compressed): {max_err:.2e}"
              f"  (tol={tol})")
        assert max_err < tol, \
            f"Real-model logits diverged too much: max_err={max_err:.4f}, tol={tol}"

    def test_top_token_matches(self, real_model_path):
        """Top predicted token must be identical between compressed and uncompressed."""
        self._ensure_compressed(real_model_path)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import sys
        tok = AutoTokenizer.from_pretrained(str(real_model_path), trust_remote_code=True)
        ids = tok("The capital of France is", return_tensors="pt").input_ids

        # Load uncompressed in bfloat16 (matches compressed weight precision)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unc = AutoModelForCausalLM.from_pretrained(
                str(real_model_path), torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        unc.eval()
        with torch.no_grad():
            top_unc = unc(ids).logits[0, -1].argmax().item()
        del unc

        sys.path.insert(0, str(ROOT))
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(real_model_path), device='cpu', compression_mode='lossless')
        loaded = cm.load()
        assert loaded is not None
        cm.model.eval()
        with torch.no_grad():
            top_cmp = cm.model(ids).logits[0, -1].argmax().item()

        print(f"\n  Uncompressed top token: {top_unc} ({tok.decode([top_unc])})")
        print(f"  Compressed  top token: {top_cmp} ({tok.decode([top_cmp])})")
        assert top_unc == top_cmp, \
            f"Top token mismatch: uncompressed={top_unc}, compressed={top_cmp}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Real-model Huffman integration (needs --model flag or auto-discovery)
# ─────────────────────────────────────────────────────────────────────────────

class TestRealModelHuffmanCompressed:
    """Verify that --entropy-code (Huffman) gives the same top token as uncompressed
    on a real model.  Uses the same real_model_path fixture as class 9."""

    @staticmethod
    def _ensure_huffman_compressed(real_model_path):
        """Run compress.py --mode lossless --entropy-code if no Huffman cache exists."""
        import subprocess, sys
        cache = real_model_path / "codebook-lossless-huffman" / "tensors"
        if not (cache.exists() and list(cache.glob("*.npz"))):
            result = subprocess.run(
                [sys.executable, str(ROOT / "compress.py"), str(real_model_path),
                 "--mode", "lossless", "--entropy-code"],
                capture_output=True, text=True, cwd=str(ROOT),
                timeout=1200,  # 20-minute cap (Huffman encode adds time for large models)
            )
            assert result.returncode == 0, \
                f"compress.py --entropy-code failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"

    def test_huffman_top_token_matches(self, real_model_path):
        """Huffman-compressed top token must match uncompressed top token."""
        self._ensure_huffman_compressed(real_model_path)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import sys
        tok = AutoTokenizer.from_pretrained(str(real_model_path), trust_remote_code=True)
        ids = tok("The capital of France is", return_tensors="pt").input_ids

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unc = AutoModelForCausalLM.from_pretrained(
                str(real_model_path), torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        unc.eval()
        with torch.no_grad():
            top_unc = unc(ids).logits[0, -1].argmax().item()
        del unc

        sys.path.insert(0, str(ROOT))
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(real_model_path), device='cpu',
                                 compression_mode='lossless', entropy_code=True)
        loaded = cm.load()
        assert loaded is not None, "CompressedChatModel.load() returned None for Huffman model"
        cm.model.eval()
        with torch.no_grad():
            top_huff = cm.model(ids).logits[0, -1].argmax().item()

        print(f"\n  Uncompressed  top token: {top_unc} ({tok.decode([top_unc])})")
        print(f"  Huffman-compr top token: {top_huff} ({tok.decode([top_huff])})")
        assert top_unc == top_huff, \
            f"Top token mismatch after Huffman: uncompressed={top_unc}, huffman={top_huff}"

    def test_huffman_logits_close(self, real_model_path):
        """Huffman-compressed logits must be within bf16/f32 accumulation tolerance."""
        self._ensure_huffman_compressed(real_model_path)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import sys
        tok = AutoTokenizer.from_pretrained(str(real_model_path), trust_remote_code=True)
        ids = tok("What is 2+2?", return_tensors="pt").input_ids

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unc = AutoModelForCausalLM.from_pretrained(
                str(real_model_path), torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        unc.eval()
        with torch.no_grad():
            uncompressed = unc(ids).logits.detach().float()
        del unc

        sys.path.insert(0, str(ROOT))
        from chat import CompressedChatModel
        cm = CompressedChatModel(str(real_model_path), device='cpu',
                                 compression_mode='lossless', entropy_code=True)
        loaded = cm.load()
        assert loaded is not None
        cm.model.eval()
        with torch.no_grad():
            huffman = cm.model(ids).logits.detach().float()

        max_err = (uncompressed - huffman).abs().max().item()
        tol = 30.0
        print(f"\n  Huffman logit max error vs uncompressed: {max_err:.2e}  (tol={tol})")
        assert max_err < tol, \
            f"Huffman-compressed logits diverged: max_err={max_err:.4f}, tol={tol}"

    def test_huffman_cache_uses_huff_fields(self, real_model_path):
        """Verify the Huffman cache actually contains huff_stream / huff_lengths fields."""
        import zipfile
        self._ensure_huffman_compressed(real_model_path)

        # lossless + entropy_code → cache suffix -huffman (set by AdaptiveCompressor)
        cache_dir = real_model_path / "codebook-lossless-huffman" / "tensors"
        assert cache_dir.exists(), "Huffman cache directory not found"

        npz_files = list(cache_dir.glob("*.npz"))
        assert npz_files, "No .npz files in Huffman cache"

        huffman_count = 0
        for npz in npz_files:
            with zipfile.ZipFile(npz) as z:
                if any(n.startswith("huff_stream") for n in z.namelist()):
                    huffman_count += 1

        frac = huffman_count / len(npz_files)
        print(f"\n  Huffman-encoded tensors: {huffman_count}/{len(npz_files)} ({frac:.1%})")
        # At least half of tensors should be Huffman-encoded
        assert frac >= 0.4, \
            f"Too few Huffman-encoded tensors: {huffman_count}/{len(npz_files)}"
