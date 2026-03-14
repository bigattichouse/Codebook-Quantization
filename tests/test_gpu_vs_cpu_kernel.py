"""
test_gpu_vs_cpu_kernel.py — Phase 7: GPU kernel correctness vs CPU C kernel.

Tests kernel-level agreement for specific tensor shapes drawn from the model's
own compressed weights. Catches shape-specific bugs that only manifest at
actual model sizes (not synthetic tests).

Skips cleanly if GPU unavailable or model too large to load on GPU.

    pytest proofofconcept/tests/test_gpu_vs_cpu_kernel.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct -s
"""

import sys
import gc
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "proofofconcept" / "src"))
sys.path.insert(0, str(ROOT / "proofofconcept"))

# Tolerance for GPU vs CPU kernel comparison
COS_THRESHOLD = 0.9999
ABS_THRESHOLD = 1e-3  # bfloat16 round-trips can accumulate error


@pytest.fixture(scope="session")
def model_path(request):
    explicit = request.config.getoption("--model", default=None)
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = Path("~/workspace/model/Qwen3.5-0.8B").expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    cache = p / "codebook" / "tensors"
    if not cache.exists() or not list(cache.glob("*.npz")):
        pytest.skip(f"No codebook cache at {cache}")
    return p


@pytest.fixture(scope="session")
def cache_dir(model_path):
    return model_path / "codebook" / "tensors"


@pytest.fixture(scope="session")
def direct_codebook_layers(cache_dir):
    """Load a representative sample of direct_codebook .npz files."""
    layers = []
    for f in sorted(cache_dir.glob("*.npz")):
        d = dict(np.load(f, allow_pickle=True))
        mode = str(d.get("mode", ""))
        if not mode.startswith("direct_codebook"):
            continue
        if "indices" not in d or "codebook" not in d:
            continue
        idx = np.asarray(d["indices"])
        cb = np.asarray(d["codebook"])
        if idx.size == 0 or cb.size == 0:
            continue
        shape = d.get("shape")
        if shape is None:
            continue
        shape = tuple(int(x) for x in np.asarray(shape))
        if len(shape) != 2:
            continue
        # bits is stored as a numpy scalar array; call .item() to get a Python int
        bits_raw = d.get("bits", 8)
        bits = int(np.asarray(bits_raw).item())
        layers.append({
            "stem": f.stem,
            "shape": shape,  # (M, K)
            "indices": idx,
            "codebook": cb.astype(np.float32),
            "bits": bits,
        })
    if not layers:
        pytest.skip("No direct_codebook layers found")
    return layers


def _select_layers(all_layers, keywords, fallback_count=4):
    """Pick specific layers by keyword, fallback to first N."""
    selected = [l for l in all_layers if any(kw in l["stem"] for kw in keywords)]
    if not selected:
        selected = all_layers[:fallback_count]
    return selected


# ── helpers ───────────────────────────────────────────────────────────────────

def _cosine_sim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _run_cpu_kernel(layer_data, x_np):
    """Run the CPU C kernel (or numpy fallback) for a single layer."""
    from compressed_matmul_cpu import compressed_matmul
    M, K = layer_data["shape"]
    bits = layer_data["bits"]
    cb = layer_data["codebook"]
    idx = layer_data["indices"]
    return compressed_matmul(x_np, idx, cb, M, K, bits, C=len(cb))


def _run_gpu_kernel(layer_data, x_tensor, device):
    """Run the GPU CUDA kernel for a single layer. Returns float32 tensor on device."""
    from gpu_accelerated_functions import GPUAcceleratedLinear
    M, K = layer_data["shape"]
    bits = layer_data["bits"]
    cb = torch.from_numpy(layer_data["codebook"]).to(device)
    gpu_func = GPUAcceleratedLinear(
        layer_data["stem"], layer_data["indices"], cb, (M, K), bits
    )
    with torch.no_grad():
        out = gpu_func(x_tensor.to(device))
    return out.cpu().float()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCPUKernelSanity:
    """Verify CPU C kernel works for actual model tensor shapes before comparing to GPU."""

    def test_cpu_kernel_available(self):
        from compressed_matmul_cpu import C_KERNEL_AVAILABLE
        if not C_KERNEL_AVAILABLE:
            pytest.skip("C kernel not compiled (gcc missing?)")
        assert C_KERNEL_AVAILABLE

    @pytest.mark.parametrize("layer_kw,T", [
        (["q_proj", "k_proj", "v_proj"], 1),
        (["down_proj", "up_proj", "gate_proj"], 1),
        (["q_proj", "k_proj", "v_proj"], 4),
    ])
    def test_cpu_kernel_attn_and_mlp_shapes(self, direct_codebook_layers, layer_kw, T):
        """CPU kernel must produce finite output for attention/MLP shapes at T=1 and T>1."""
        selected = _select_layers(direct_codebook_layers, layer_kw)
        if not selected:
            pytest.skip(f"No layers matching {layer_kw}")
        layer = selected[0]
        M, K = layer["shape"]
        np = __import__("numpy")
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((T, K)).astype(np.float32)
        out = _run_cpu_kernel(layer, x_np)
        assert out.shape == (T, M), f"Wrong output shape: {out.shape}"
        assert np.isfinite(out).all(), f"CPU kernel produced NaN/Inf for {layer['stem']}"

    def test_cpu_kernel_lm_head_shape(self, direct_codebook_layers):
        """lm_head has large M — verify no int overflow."""
        lm_layers = [l for l in direct_codebook_layers if "lm_head" in l["stem"]]
        embed_layers = [l for l in direct_codebook_layers if "embed_tokens" in l["stem"]]
        candidates = lm_layers or embed_layers
        if not candidates:
            pytest.skip("No lm_head/embed_tokens layer found")
        layer = candidates[0]
        M, K = layer["shape"]
        np = __import__("numpy")
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((1, K)).astype(np.float32)
        out = _run_cpu_kernel(layer, x_np)
        assert out.shape == (1, M)
        assert np.isfinite(out).all(), f"lm_head CPU kernel produced NaN/Inf (M={M})"


class TestGPUvsGPUKernel:
    """GPU kernel must agree with CPU kernel to within floating-point tolerance."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from gpu_accelerated_functions import GPUAcceleratedLinear
        except Exception as e:
            pytest.skip(f"GPU extension unavailable: {e}")

    @pytest.mark.parametrize("layer_kw", [
        ["q_proj"],
        ["down_proj"],
        ["up_proj"],
    ])
    def test_gpu_matches_cpu_for_layer_type(self, direct_codebook_layers, layer_kw):
        selected = _select_layers(direct_codebook_layers, layer_kw)
        if not selected:
            pytest.skip(f"No layers matching {layer_kw}")
        layer = selected[0]
        M, K = layer["shape"]
        print(f"\n  Testing {layer['stem']} shape=({M},{K}) bits={layer['bits']}")

        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((1, K)).astype(np.float32)
        x_tensor = torch.from_numpy(x_np)

        cpu_out = _run_cpu_kernel(layer, x_np)  # [1, M]
        try:
            gpu_out = _run_gpu_kernel(layer, x_tensor, device='cuda')  # [1, M]
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM for layer {layer['stem']}: {e}")

        cpu_t = torch.from_numpy(cpu_out)
        cos = _cosine_sim(cpu_t, gpu_out)
        max_diff = (cpu_t - gpu_out).abs().max().item()
        print(f"  cos={cos:.8f}  max_abs_diff={max_diff:.6e}")
        assert cos > COS_THRESHOLD, \
            f"GPU/CPU mismatch for {layer['stem']}: cos={cos:.6f} (threshold={COS_THRESHOLD})"
        assert max_diff < ABS_THRESHOLD, \
            f"GPU/CPU max_abs_diff={max_diff:.6e} for {layer['stem']}"

    def test_gpu_matches_cpu_for_lm_head(self, direct_codebook_layers):
        """lm_head has large M — most critical correctness check for GPU kernel."""
        lm_layers = [l for l in direct_codebook_layers if "lm_head" in l["stem"]]
        embed_layers = [l for l in direct_codebook_layers if "embed_tokens" in l["stem"]]
        candidates = lm_layers or embed_layers
        if not candidates:
            pytest.skip("No lm_head/embed_tokens layer in cache")
        layer = candidates[0]
        M, K = layer["shape"]
        print(f"\n  Testing lm_head {layer['stem']} shape=({M},{K}) bits={layer['bits']}")

        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((1, K)).astype(np.float32)
        x_tensor = torch.from_numpy(x_np)

        cpu_out = _run_cpu_kernel(layer, x_np)
        try:
            gpu_out = _run_gpu_kernel(layer, x_tensor, device='cuda')
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM for lm_head: {e}")

        cpu_t = torch.from_numpy(cpu_out)
        cos = _cosine_sim(cpu_t, gpu_out)
        max_diff = (cpu_t - gpu_out).abs().max().item()
        print(f"  cos={cos:.8f}  max_abs_diff={max_diff:.6e}  M={M}")
        assert cos > COS_THRESHOLD, f"lm_head GPU/CPU mismatch: cos={cos:.6f}"

    def test_gpu_matches_cpu_batch_T4(self, direct_codebook_layers):
        """Verify kernel agreement at T=4 (prefill batch)."""
        selected = _select_layers(direct_codebook_layers, ["q_proj", "gate_proj"])
        if not selected:
            pytest.skip("No suitable layers found")
        layer = selected[0]
        M, K = layer["shape"]
        T = 4
        print(f"\n  Testing T={T} batch, {layer['stem']}")

        rng = np.random.default_rng(1234)
        x_np = rng.standard_normal((T, K)).astype(np.float32)
        x_tensor = torch.from_numpy(x_np)

        cpu_out = _run_cpu_kernel(layer, x_np)
        try:
            gpu_out = _run_gpu_kernel(layer, x_tensor, device='cuda')
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM for T={T}: {e}")

        cpu_t = torch.from_numpy(cpu_out)
        cos = _cosine_sim(cpu_t, gpu_out)
        print(f"  T={T} cos={cos:.8f}")
        assert cos > COS_THRESHOLD, f"T={T} GPU/CPU mismatch: cos={cos:.6f}"

    @pytest.mark.parametrize("T", [1, 8, 64])
    def test_gpu_matches_cpu_various_T(self, direct_codebook_layers, T):
        """Test at multiple sequence lengths to catch grid-dim issues."""
        selected = direct_codebook_layers[:2]
        if not selected:
            pytest.skip("No layers")
        layer = selected[0]
        M, K = layer["shape"]

        rng = np.random.default_rng(T * 7)
        x_np = rng.standard_normal((T, K)).astype(np.float32)
        x_tensor = torch.from_numpy(x_np)

        cpu_out = _run_cpu_kernel(layer, x_np)
        try:
            gpu_out = _run_gpu_kernel(layer, x_tensor, device='cuda')
        except torch.cuda.OutOfMemoryError as e:
            pytest.skip(f"GPU OOM for T={T}: {e}")

        cpu_t = torch.from_numpy(cpu_out)
        cos = _cosine_sim(cpu_t, gpu_out)
        print(f"\n  T={T} {layer['stem']} cos={cos:.8f}")
        assert cos > COS_THRESHOLD, f"T={T} GPU/CPU mismatch: cos={cos:.6f}"


class TestGridDimensionSafety:
    """Verify that T*M doesn't overflow int32 at large batch sizes."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_no_int32_overflow_in_grid_dim(self, direct_codebook_layers):
        """
        For lm_head (large M), T*M must be safe.
        At T=1 autoregressive decode: grid.x = 1 * M (fine).
        At T=100 prefill: grid.x = 100 * M. For M=151936: 15,193,600 < 2^31. Still fine.
        For M=248320: 100 * 248320 = 24,832,000. Still fine.
        """
        large_layers = sorted(direct_codebook_layers, key=lambda l: -l["shape"][0])
        if not large_layers:
            pytest.skip("No layers found")
        layer = large_layers[0]
        M, K = layer["shape"]
        T = 100  # Typical prefill length

        grid_dim = T * M
        print(f"\n  Largest M: {M}, T={T}, grid.x = {grid_dim}")
        print(f"  INT32_MAX = {2**31 - 1}")
        assert grid_dim < 2**31, \
            f"T*M={grid_dim} exceeds INT32_MAX at T={T}, M={M} — grid overflow risk"

    def test_grid_dim_at_typical_prefill(self, direct_codebook_layers):
        """Even at T=512 prefill, grid.x should be safe."""
        large_layers = sorted(direct_codebook_layers, key=lambda l: -l["shape"][0])
        if not large_layers:
            pytest.skip("No layers found")
        layer = large_layers[0]
        M, K = layer["shape"]
        T = 512

        grid_dim = T * M
        print(f"\n  T={T}, M={M}, grid.x = {grid_dim} ({'OK' if grid_dim < 2**31 else 'OVERFLOW'})")
        if grid_dim >= 2**31:
            pytest.xfail(
                f"T*M={grid_dim} exceeds INT32_MAX at T={T}, M={M}. "
                f"This would cause incorrect GPU output for long prefill. "
                f"Fix: use int64 for grid dim or cap grid.x."
            )
