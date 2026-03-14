"""
test_memory_budget.py — Phase 3: Compute VRAM/RAM requirements before load.

Answers: will this model fit on the P2200 (5 GB VRAM)?
         will it fit in CPU RAM?
         what is the minimum memory needed?

No model load. Fast.

    pytest proofofconcept/tests/test_memory_budget.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "proofofconcept" / "src"))
sys.path.insert(0, str(ROOT / "proofofconcept"))

# P2200 VRAM in bytes (5 GB)
P2200_VRAM_BYTES = 5 * 1024 ** 3
# Headroom: leave 1 GB for activations, KV cache, PyTorch internals
VRAM_HEADROOM_BYTES = 1 * 1024 ** 3
VRAM_USABLE = P2200_VRAM_BYTES - VRAM_HEADROOM_BYTES

# Maximum RAM we expect on the workstation (leave 8 GB for OS + Python)
MAX_RAM_BYTES = 24 * 1024 ** 3


@pytest.fixture(scope="session")
def model_path(request):
    explicit = request.config.getoption("--model", default=None)
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = Path("~/workspace/model/Qwen3.5-0.8B").expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    return p


@pytest.fixture(scope="session")
def cache_dir(model_path):
    d = model_path / "codebook" / "tensors"
    if not d.exists() or not list(d.glob("*.npz")):
        pytest.skip(f"No cache at {d}")
    return d


@pytest.fixture(scope="session")
def budget(cache_dir):
    """Compute memory budget from cache files without loading the model."""
    index_bytes = 0
    exact_bytes = 0
    codebook_bytes = 0
    n_direct = 0
    n_exact = 0
    n_linear_quant = 0
    layer_sizes = {}  # stem → index_bytes for per-layer analysis

    for f in sorted(cache_dir.glob("*.npz")):
        try:
            d = dict(np.load(f, allow_pickle=True))
        except Exception:
            continue

        mode = str(d.get("mode", "unknown"))

        if mode == "direct_codebook":
            idx = d.get("indices")
            if idx is not None:
                nb = np.asarray(idx).nbytes
                index_bytes += nb
                layer_sizes[f.stem] = nb
            cb = d.get("codebook")
            if cb is not None and hasattr(cb, "size") and cb.size > 0:
                codebook_bytes += np.asarray(cb).astype(np.float32).nbytes
            n_direct += 1

        elif mode == "exact":
            data = d.get("data")
            if data is not None:
                exact_bytes += np.asarray(data).nbytes
            n_exact += 1

        elif mode == "linear_quant":
            idx = d.get("indices")
            if idx is not None:
                index_bytes += np.asarray(idx).nbytes
            n_linear_quant += 1

    return {
        "index_bytes": index_bytes,
        "exact_bytes": exact_bytes,
        "codebook_bytes": codebook_bytes,
        "n_direct": n_direct,
        "n_exact": n_exact,
        "n_linear_quant": n_linear_quant,
        "layer_sizes": layer_sizes,
        "total_compressed_bytes": index_bytes + exact_bytes + codebook_bytes,
    }


# ── tests ─────────────────────────────────────────────────────────────────────

class TestBudgetReporting:
    def test_print_full_budget(self, budget):
        """Always-pass: just prints the budget for human inspection."""
        gb = 1024 ** 3
        print(f"\n  === Memory Budget ===")
        print(f"  Packed index bytes:  {budget['index_bytes']/gb:.3f} GB")
        print(f"  Exact weight bytes:  {budget['exact_bytes']/gb:.3f} GB")
        print(f"  Codebook bytes:      {budget['codebook_bytes']/1e6:.2f} MB")
        print(f"  TOTAL compressed:    {budget['total_compressed_bytes']/gb:.3f} GB")
        print(f"  Tensor counts: direct={budget['n_direct']}, "
              f"exact={budget['n_exact']}, linear_quant={budget['n_linear_quant']}")
        fits_vram = budget['index_bytes'] + budget['exact_bytes'] <= VRAM_USABLE
        print(f"  Fits in P2200 VRAM (4 GB usable): {'✅ YES' if fits_vram else '❌ NO'}")

    def test_total_bytes_nonzero(self, budget):
        assert budget["total_compressed_bytes"] > 0, "No data found in cache"

    def test_sanity_cap_on_index_bytes(self, budget):
        """Indices > 100 GB would indicate a bug in .npz files."""
        assert budget["index_bytes"] < 100 * 1024 ** 3, \
            f"index_bytes implausibly large: {budget['index_bytes']/1e9:.1f} GB"


class TestVRAMFit:
    def test_fits_in_cpu_ram(self, budget):
        total = budget["total_compressed_bytes"]
        if total > MAX_RAM_BYTES:
            pytest.skip(
                f"Model needs {total/1e9:.1f} GB RAM — exceeds assumed {MAX_RAM_BYTES/1e9:.0f} GB limit. "
                f"Use mmap path."
            )
        assert total <= MAX_RAM_BYTES

    def test_vram_fit_or_skip(self, budget):
        """
        Does not fail if model doesn't fit — it SKIPs with a helpful message.
        Failing means 'something is wrong'; skipping means 'VRAM too small, use CPU path'.
        """
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        total = budget["index_bytes"] + budget["exact_bytes"]
        if total > VRAM_USABLE:
            pytest.skip(
                f"Model needs ~{total/1e9:.2f} GB VRAM; P2200 has ~{VRAM_USABLE/1e9:.1f} GB usable. "
                f"Use device='cpu' or mmap path."
            )
        assert total <= VRAM_USABLE

    def test_print_vram_recommendation(self, budget):
        """Always passes. Prints whether GPU or CPU path is recommended."""
        total = budget["index_bytes"] + budget["exact_bytes"]
        gb = total / 1e9
        if total <= VRAM_USABLE:
            print(f"\n  ✅ GPU path recommended ({gb:.2f} GB fits in {VRAM_USABLE/1e9:.1f} GB usable VRAM)")
        elif total <= MAX_RAM_BYTES:
            print(f"\n  ⚠️  CPU-only path recommended ({gb:.2f} GB exceeds VRAM, fits in RAM)")
        else:
            print(f"\n  ❌ Model too large for RAM ({gb:.2f} GB). Consider mmap path.")


class TestLargestLayers:
    def test_print_largest_layers(self, budget):
        """Always passes. Prints the 10 largest layers by index size."""
        sizes = budget["layer_sizes"]
        top10 = sorted(sizes.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 largest layers by packed index size:")
        for stem, nb in top10:
            print(f"    {nb/1e6:7.2f} MB  {stem}")

    def test_no_single_layer_dominates(self, budget):
        """No single tensor should be > 80% of total index bytes (except lm_head)."""
        sizes = budget["layer_sizes"]
        total = budget["index_bytes"]
        if total == 0:
            pytest.skip("No index bytes")
        for stem, nb in sizes.items():
            frac = nb / total
            if frac > 0.8 and "lm_head" not in stem and "embed_tokens" not in stem:
                pytest.fail(
                    f"Layer '{stem}' is {frac*100:.1f}% of total index bytes "
                    f"({nb/1e6:.1f} MB / {total/1e6:.1f} MB) — suspicious"
                )
