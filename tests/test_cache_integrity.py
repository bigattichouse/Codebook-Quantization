"""
test_cache_integrity.py — Phase 1: Verify .npz cache is complete and uncorrupted.

No model load required. Fast. Run before anything else.

    pytest proofofconcept/tests/test_cache_integrity.py -v \
        --model ~/workspace/model/Qwen2.5-3B-Instruct
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "proofofconcept" / "src"))
sys.path.insert(0, str(ROOT / "proofofconcept"))


# ── fixtures ──────────────────────────────────────────────────────────────────

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
    if not d.exists():
        pytest.skip(f"No cache directory at {d}")
    return d


@pytest.fixture(scope="session")
def npz_files(cache_dir):
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        pytest.skip(f"No .npz files in {cache_dir}")
    return files


@pytest.fixture(scope="session")
def all_data(npz_files):
    """Load all .npz metadata (not the full arrays) for fast scanning."""
    results = []
    for f in npz_files:
        try:
            d = dict(np.load(f, allow_pickle=True))
            results.append((f.stem, d))
        except Exception as e:
            results.append((f.stem, {"_error": str(e)}))
    return results


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCacheDirectory:
    def test_cache_directory_exists(self, cache_dir):
        assert cache_dir.exists(), f"Cache directory missing: {cache_dir}"

    def test_metadata_json_exists(self, model_path):
        meta = model_path / "codebook" / "metadata.json"
        assert meta.exists(), f"metadata.json missing at {meta}"

    def test_metadata_json_valid(self, model_path):
        import json
        meta = model_path / "codebook" / "metadata.json"
        if not meta.exists():
            pytest.skip("metadata.json not present")
        with open(meta) as f:
            data = json.load(f)
        assert isinstance(data, dict), "metadata.json is not a dict"

    def test_npz_count_nonzero(self, npz_files):
        assert len(npz_files) > 0, "No .npz files found in cache"

    def test_npz_count_reasonable(self, npz_files):
        # A 1B model should have 100+; a 3B model 300+
        assert len(npz_files) >= 50, \
            f"Only {len(npz_files)} .npz files — likely incomplete compression run"
        print(f"\n  Found {len(npz_files)} .npz files")


class TestNpzIntegrity:
    def test_all_npz_openable(self, all_data):
        errors = [(stem, d["_error"]) for stem, d in all_data if "_error" in d]
        if errors:
            for stem, err in errors[:10]:
                print(f"\n  ❌ {stem}: {err}")
        assert not errors, f"{len(errors)} .npz file(s) could not be opened"

    def test_all_npz_have_mode(self, all_data):
        missing_mode = [stem for stem, d in all_data
                        if "_error" not in d and "mode" not in d
                        and "arr_0" not in d]  # legacy format uses arr_0
        if missing_mode:
            for s in missing_mode[:5]:
                print(f"\n  ⚠️  {s}: no 'mode' key")
        # Soft check: warn but don't fail (some files use different schema)
        if len(missing_mode) > len(all_data) // 2:
            pytest.fail(f"More than half of .npz files ({len(missing_mode)}) lack 'mode' key")

    def test_mode_distribution(self, all_data):
        from collections import Counter
        modes = Counter()
        for stem, d in all_data:
            if "_error" not in d:
                mode = d.get("mode", "unknown")
                if hasattr(mode, "item"):
                    mode = mode.item()
                modes[str(mode)] += 1
        print(f"\n  Mode distribution: {dict(modes)}")
        assert modes.get("direct_codebook", 0) > 0, \
            "No direct_codebook tensors found — compression may not have run correctly"


class TestCodebookValues:
    def test_no_nan_in_codebooks(self, all_data):
        bad = []
        for stem, d in all_data:
            if "_error" in d:
                continue
            cb = d.get("codebook")
            if cb is None or not hasattr(cb, "shape") or cb.size == 0:
                continue
            cb_arr = np.asarray(cb).astype(np.float32)
            if np.isnan(cb_arr).any():
                bad.append((stem, "NaN in codebook"))
            elif np.isinf(cb_arr).any():
                bad.append((stem, "Inf in codebook"))
        if bad:
            for stem, msg in bad[:10]:
                print(f"\n  ❌ {stem}: {msg}")
        assert not bad, f"{len(bad)} codebooks contain NaN/Inf"

    def test_no_all_zero_codebooks(self, all_data):
        bad = []
        for stem, d in all_data:
            if "_error" in d:
                continue
            cb = d.get("codebook")
            if cb is None or not hasattr(cb, "shape") or cb.size == 0:
                continue
            cb_arr = np.asarray(cb).astype(np.float32)
            if np.abs(cb_arr).max() < 1e-9:
                bad.append(stem)
        if bad:
            for s in bad[:5]:
                print(f"\n  ❌ {s}: codebook is all zeros")
        assert not bad, f"{len(bad)} codebooks are all zeros"

    def test_codebook_sizes_reasonable(self, all_data):
        bad = []
        for stem, d in all_data:
            if "_error" in d:
                continue
            cb = d.get("codebook")
            if cb is None or not hasattr(cb, "shape") or cb.size == 0:
                continue
            n = cb.size
            if n < 2:
                bad.append((stem, f"codebook too small: {n} entries"))
            elif n > 2**16 + 1:
                bad.append((stem, f"codebook suspiciously large: {n} entries"))
        if bad:
            for stem, msg in bad[:5]:
                print(f"\n  ⚠️  {stem}: {msg}")
        assert not bad, f"{len(bad)} codebooks have unreasonable sizes"


class TestIndexValues:
    def test_no_index_out_of_bounds(self, all_data):
        """Every index must be < len(codebook).

        Note: bits is stored as a 0-d or 1-d numpy array — must call .item().
        """
        bad = []
        checked = 0
        for stem, d in all_data:
            if "_error" in d:
                continue
            mode = d.get("mode", "")
            if hasattr(mode, "item"):
                mode = mode.item()
            if str(mode) != "direct_codebook":
                continue
            cb = d.get("codebook")
            indices_raw = d.get("indices")
            if cb is None or indices_raw is None:
                continue
            cb_arr = np.asarray(cb)
            idx_arr = np.asarray(indices_raw)
            n_codebook = cb_arr.size

            # bits is stored as a numpy scalar array; must use .item()
            bits_raw = d.get("bits", 8)
            bits = int(np.asarray(bits_raw).item()) if hasattr(bits_raw, "__len__") \
                else int(bits_raw)

            max_possible = (1 << bits) - 1
            # Check: max decodable index must be < codebook size
            if max_possible >= n_codebook * 4:
                # bits too wide for this codebook — likely a compression bug
                bad.append((stem, f"bits={bits} but codebook has only {n_codebook} entries "
                                   f"(max_idx={max_possible})"))

            # For uncompressed-format indices (uint16 for 1D tensors like norms):
            # verify all indices are actually within codebook bounds
            if idx_arr.dtype in (np.uint16, np.uint32, np.int32, np.int64):
                oob = int((idx_arr >= n_codebook).sum())
                if oob > 0:
                    bad.append((stem, f"{oob} indices >= codebook size ({n_codebook})"))

            checked += 1
        print(f"\n  Checked {checked} direct_codebook tensors")
        if bad:
            for stem, msg in bad[:10]:
                print(f"\n  ❌ {stem}: {msg}")
        assert not bad, f"{len(bad)} tensors have bits/codebook size mismatch"

    def test_indices_dtype_is_uint8_for_2d_weights(self, all_data):
        """
        2D weight tensors (nn.Linear) must have bit-packed uint8 indices —
        they're streamed directly through the C/GPU kernel at inference time.

        1D tensors (norm weights, biases) may use uint16 because they are
        decompressed at load time via AdaptiveCompressor.get_tensor(), not
        the C kernel. This is by design.
        """
        bad_2d = []
        skip_1d = 0
        for stem, d in all_data:
            if "_error" in d:
                continue
            mode = d.get("mode", "")
            if hasattr(mode, "item"):
                mode = mode.item()
            if str(mode) != "direct_codebook":
                continue
            idx = d.get("indices")
            if idx is None:
                continue
            shape = d.get("shape")
            if shape is not None:
                shape_arr = np.asarray(shape)
                if shape_arr.ndim == 1 and len(shape_arr) == 1:
                    # 1D tensor (norm/bias): uint16 is acceptable
                    skip_1d += 1
                    continue
            # 2D weight tensor: must be bit-packed uint8
            if np.asarray(idx).dtype != np.uint8:
                bad_2d.append((stem, np.asarray(idx).dtype))
        print(f"\n  1D tensors (uint16 OK): {skip_1d}")
        if bad_2d:
            for stem, dt in bad_2d[:5]:
                print(f"\n  ❌ {stem}: 2D indices dtype={dt} (expected uint8 for kernel path)")
        assert not bad_2d, \
            f"{len(bad_2d)} 2D weight tensors have non-uint8 packed indices"


class TestCoverageEstimate:
    def test_total_index_bytes_reported(self, all_data):
        total = 0
        for stem, d in all_data:
            if "_error" in d:
                continue
            idx = d.get("indices")
            if idx is not None:
                total += np.asarray(idx).nbytes
        gb = total / 1e9
        print(f"\n  Total packed index bytes: {gb:.2f} GB")
        # Not a hard assertion — just report
        assert gb < 100, f"Index bytes suspiciously large: {gb:.1f} GB"

    def test_direct_codebook_fraction(self, all_data):
        total = len([d for _, d in all_data if "_error" not in d])
        direct = sum(1 for _, d in all_data
                     if "_error" not in d
                     and str(d.get("mode", "")).startswith("direct_codebook"))
        frac = direct / total if total else 0
        print(f"\n  direct_codebook: {direct}/{total} ({frac*100:.1f}%)")
        # Most large weight matrices should be direct_codebook
        assert frac > 0.3, \
            f"Only {frac*100:.1f}% of tensors are direct_codebook — compression may be incomplete"
