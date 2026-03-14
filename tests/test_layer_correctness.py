"""
test_layer_correctness.py — Per-layer cosine similarity for a compressed model.

Tests that each AdaptiveCodebookLinear/Embedding module produces output within
tolerance of the fully decompressed reference on the same input.

Run with any model that has a complete codebook cache:
    pytest proofofconcept/tests/test_layer_correctness.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct

Or (default falls back to Qwen3.5-0.8B if no --model given):
    pytest proofofconcept/tests/test_layer_correctness.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── path setup ──────────────────────────────────────────────────────────────
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from adaptive_compressor import AdaptiveCompressor
from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
from compressed_matmul_cpu import compressed_matmul as c_matmul, C_KERNEL_AVAILABLE


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_path(request):
    explicit = request.config.getoption("--model")
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = Path("~/workspace/model/Qwen3.5-0.8B").expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    cache_dir = p / "codebook" / "tensors"
    if not cache_dir.exists() or not list(cache_dir.glob("*.npz")):
        pytest.skip(f"No codebook cache at {cache_dir}")
    return p


@pytest.fixture(scope="session")
def compressor(model_path):
    comp = AdaptiveCompressor(model_path, compression_mode="lossless", store_in_model=True)
    comp.load_compressed(load_tensors=False)
    return comp


@pytest.fixture(scope="session")
def global_codebooks(compressor):
    """Load global codebooks exactly as chat.py does at inference time."""
    return compressor._load_global_codebooks()  # dict of {ttype: bfloat16 tensor}


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(np.dot(a, b) / denom)


def get_weight_tensors(compressor, global_codebooks_np=None, limit=None):
    """Yield (name, data, codebook_np) for direct_codebook weight tensors, up to `limit`.

    codebook_np is always a resolved float32 numpy array (local or global).
    Tensors where neither local nor global codebook can be resolved are skipped.
    """
    # Load global codebooks as float32 numpy once
    raw_global = compressor._load_global_codebooks()
    global_np = {k: v.float().numpy() for k, v in raw_global.items()}

    cache_dir = compressor.cache_dir / "tensors"
    count = 0
    for f in sorted(cache_dir.glob("*.npz")):
        d = np.load(f)
        if str(d["mode"]) != "direct_codebook":
            continue
        name = str(d["name"])
        data = compressor._get_compressed_tensor_data(name)
        if data is None:
            continue
        # Only yield 2D weight tensors (skip 1D biases, etc.)
        shape = [int(s) for s in data["shape"]]
        if len(shape) != 2:
            continue
        # Resolve codebook
        cb = data.get("codebook")
        if cb is not None and (hasattr(cb, 'size') and cb.size > 0):
            cb_np = cb.astype(np.float32) if isinstance(cb, np.ndarray) else cb.float().numpy()
        else:
            ttype = data.get("codebook_type") or data.get("type")
            cb_np = global_np.get(ttype) if ttype else None
        if cb_np is None:
            continue  # can't resolve codebook — skip
        yield name, data, cb_np
        count += 1
        if limit and count >= limit:
            break


# ── tests ─────────────────────────────────────────────────────────────────────

class TestBiasLoading:
    """Verify that direct_codebook biases decompress to valid (non-garbage) values."""

    def test_no_bias_file_is_all_zeros(self, compressor):
        """Any tensor in cache should decompress to non-zero values."""
        cache_dir = compressor.cache_dir / "tensors"
        bias_files = list(cache_dir.glob("*bias*.npz"))
        if not bias_files:
            pytest.skip("No bias files in this model's cache")
        for f in bias_files[:5]:
            d = np.load(f)
            name = str(d["name"])
            tensor = compressor.get_tensor(name)
            if tensor is None:
                continue
            assert not np.all(tensor == 0), f"{name}: bias decompressed to all zeros (likely garbage)"
            assert np.isfinite(tensor).all(), f"{name}: bias contains NaN/Inf"

    def test_direct_codebook_bias_decompresses(self, compressor):
        """direct_codebook biases should produce valid float arrays."""
        cache_dir = compressor.cache_dir / "tensors"
        dc_biases = [f for f in cache_dir.glob("*bias*.npz")
                     if str(np.load(f)["mode"]) == "direct_codebook"]
        if not dc_biases:
            pytest.skip("No direct_codebook bias files")
        for f in dc_biases[:3]:
            d = np.load(f)
            name = str(d["name"])
            tensor = compressor.get_tensor(name)
            assert tensor is not None, f"get_tensor returned None for {name}"
            assert np.isfinite(tensor).all(), f"{name}: decompressed bias has NaN/Inf"
            assert tensor.std() > 0, f"{name}: decompressed bias has zero std (suspicious)"


class TestCKernelAccuracy:
    """C kernel must match reference decompression to float tolerance."""

    @pytest.mark.skipif(not C_KERNEL_AVAILABLE, reason="C kernel not compiled")
    def test_c_kernel_matches_reference_small(self, compressor):
        """First few weight tensors: C kernel output must match x @ W.T within 1e-3."""
        batch = 4
        tested = 0
        for name, data, cb in get_weight_tensors(compressor, limit=10):
            M, K = [int(s) for s in data["shape"]]
            if M * K > 10_000_000:
                continue  # skip huge tensors in unit tests
            packed = data["indices"]
            bits = int(data.get("bits", 8))
            w_ref = compressor.get_tensor(name).reshape(M, K)
            x = np.random.default_rng(42).standard_normal((batch, K)).astype(np.float32)
            out_c = c_matmul(x, packed, cb, M, K, bits, C=len(cb))
            out_ref = x @ w_ref.T
            cs = cosine_sim(out_c, out_ref)
            assert cs > 0.999, f"{name}: C kernel cosine sim {cs:.6f} < 0.999"
            tested += 1
            if tested >= 5:
                break
        if tested == 0:
            pytest.skip("No eligible tensors found")

    @pytest.mark.skipif(not C_KERNEL_AVAILABLE, reason="C kernel not compiled")
    def test_c_kernel_large_shapes(self, compressor):
        """Large tensors (M, K ≥ 2048) must still produce correct output."""
        batch = 2
        found = False
        for name, data, cb in get_weight_tensors(compressor):
            M, K = [int(s) for s in data["shape"]]
            if M < 2048 or K < 2048:
                continue
            found = True
            packed = data["indices"]
            bits = int(data.get("bits", 8))
            w_ref = compressor.get_tensor(name).reshape(M, K)
            x = np.random.default_rng(0).standard_normal((batch, K)).astype(np.float32)
            out_c = c_matmul(x, packed, cb, M, K, bits, C=len(cb))
            out_ref = x @ w_ref.T
            cs = cosine_sim(out_c, out_ref)
            assert cs > 0.999, f"{name}: C kernel cosine sim {cs:.6f} < 0.999 for shape [{M},{K}]"
            break  # one large tensor is enough
        if not found:
            pytest.skip("No tensors with M,K >= 2048 in this model")


class TestModuleForward:
    """AdaptiveCodebookLinear / Embedding forward must match reference."""

    def test_linear_forward_matches_reference(self, compressor, global_codebooks):
        """First 8 Linear layers: forward output cosine sim > 0.999."""
        count = 0
        for name, data, _cb in get_weight_tensors(compressor, limit=20):
            M, K = [int(s) for s in data["shape"]]
            if M * K > 10_000_000:
                continue
            layer = AdaptiveCodebookLinear.from_compressed(name, data, global_codebooks, use_gpu=False)
            w_ref = torch.from_numpy(compressor.get_tensor(name).reshape(M, K))
            x = torch.randn(3, K)
            out_layer = layer(x).float()
            out_ref = torch.nn.functional.linear(x, w_ref)
            cs = torch.nn.functional.cosine_similarity(
                out_layer.reshape(1, -1), out_ref.reshape(1, -1)
            ).item()
            assert cs > 0.999, f"{name}: forward cosine sim {cs:.6f} < 0.999"
            count += 1
            if count >= 8:
                break
        if count == 0:
            pytest.skip("No eligible tensors found")

    def test_linear_forward_with_bias(self, compressor, global_codebooks):
        """Layers with direct_codebook bias: bias must be applied correctly."""
        cache_dir = compressor.cache_dir / "tensors"
        dc_bias_names = []
        for f in cache_dir.glob("*bias*.npz"):
            d = np.load(f)
            if str(d["mode"]) == "direct_codebook":
                dc_bias_names.append(str(d["name"]))

        if not dc_bias_names:
            pytest.skip("No direct_codebook biases in this model")

        for bias_name in dc_bias_names[:2]:
            # Derive weight name from bias name (replace trailing .bias with .weight)
            weight_name = bias_name[:-5] + ".weight" if bias_name.endswith(".bias") else None
            if weight_name is None:
                continue
            weight_data = compressor._get_compressed_tensor_data(weight_name)
            if weight_data is None:
                continue
            M, K = [int(s) for s in weight_data["shape"]]
            if M * K > 4_000_000:
                continue

            bias_tensor = compressor.get_tensor(bias_name)
            if bias_tensor is None:
                continue

            layer = AdaptiveCodebookLinear.from_compressed(weight_name, weight_data, global_codebooks, use_gpu=False)
            # Manually attach decompressed bias (simulating what chat.py does)
            layer.bias = torch.from_numpy(bias_tensor)

            w_ref = torch.from_numpy(compressor.get_tensor(weight_name).reshape(M, K))
            b_ref = torch.from_numpy(bias_tensor)
            x = torch.randn(3, K)

            out_layer = layer(x).float()
            out_ref = torch.nn.functional.linear(x, w_ref, b_ref)

            cs = torch.nn.functional.cosine_similarity(
                out_layer.reshape(1, -1), out_ref.reshape(1, -1)
            ).item()
            assert cs > 0.999, f"{weight_name} (with bias): forward cosine sim {cs:.6f} < 0.999"

    def test_embedding_forward_matches_reference(self, compressor, global_codebooks):
        """Embedding layer forward: per-token output must match reference."""
        cache_dir = compressor.cache_dir / "tensors"
        emb_files = [f for f in cache_dir.glob("*embed_tokens*weight*.npz")]
        if not emb_files:
            pytest.skip("No embed_tokens weight in cache")

        f = emb_files[0]
        d = np.load(f)
        if str(d["mode"]) != "direct_codebook":
            pytest.skip("embed_tokens not direct_codebook in this model")

        name = str(d["name"])
        data = compressor._get_compressed_tensor_data(name)
        emb = AdaptiveCodebookEmbedding.from_compressed(name, data, global_codebooks, use_gpu=False)

        vocab_size, hidden = [int(s) for s in data["shape"]]
        w_ref = torch.from_numpy(compressor.get_tensor(name).reshape(vocab_size, hidden))

        tok_ids = torch.tensor([0, 1, 42, 100, vocab_size - 1])
        out_emb = emb(tok_ids).float()
        out_ref = w_ref[tok_ids].float()

        cs = torch.nn.functional.cosine_similarity(
            out_emb.reshape(1, -1), out_ref.reshape(1, -1)
        ).item()
        assert cs > 0.999, f"Embedding cosine sim {cs:.6f} < 0.999"
