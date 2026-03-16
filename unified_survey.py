#!/usr/bin/env python3
"""
unified_survey.py — GGUF + codebook combined SNR survey.

Merges both quantization families into one table sorted by SNR so you can
directly compare them at every quality level:

  GGUF std   — every tensor at the same GGUF level (uniform quant)
  CB std     — every tensor at the same codebook level (uniform k-means VQ)
  GGUF mix   — optimal per-tensor GGUF assignment at a given SNR floor
  CB mix     — optimal per-tensor codebook assignment at a given SNR floor

The "vs nearest std" column compares mixed rows against the cheapest standard
row (GGUF or CB) that achieves the same quality, so savings/costs are visible
across both families.

Usage:
    python proofofconcept/unified_survey.py /path/to/model
    python proofofconcept/unified_survey.py /path/to/model --force

Requires snr_profile.json and codebook_profile.json in the model directory.
Run these first if not present:
    python quantization/snr_quant.py       <model> --survey
    python quantization/compare_codebook.py <model>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Locate quantization/ package relative to this script
_QUANT_DIR = Path(__file__).resolve().parent.parent / "quantization"
sys.path.insert(0, str(_QUANT_DIR))

from src.quant_sim import QUANT_LEVELS
from src.codebook_sim import CODEBOOK_LEVELS
from src.snr_profiler import Profiler, LevelResult, TensorProfile
from src.optimizer import optimize_for_snr, compute_model_snr, compute_total_bytes
from src.reporter import MIXED_SNR_FLOORS, row_tag, print_snr_winners, print_size_winners


# ---------------------------------------------------------------------------
# Codebook profile I/O
# ---------------------------------------------------------------------------

_CB_CACHE_NAME = "codebook_profile.json"


def _deserialise_cb(raw: list[dict]) -> list[TensorProfile]:
    profiles = []
    for r in raw:
        levels = {k: LevelResult(**v) for k, v in r["levels"].items()}
        profiles.append(TensorProfile(
            name=r["name"], shape=r["shape"], num_params=r["num_params"],
            tensor_class=r["tensor_class"], policy=r["policy"], levels=levels,
        ))
    return profiles


def _run_cb_profiler(
    model_path: Path,
    gguf_profiles: list[TensorProfile],
    sample_size: int,
    verbose: bool,
) -> list[TensorProfile]:
    """Profile all tensors at CB3–CB12.  Mirrors compare_codebook.py logic."""
    from src.tensor_loader import ModelLoader

    gguf_by_name = {p.name: p for p in gguf_profiles}
    loader = ModelLoader(model_path)
    metas  = list(loader.tensors())
    n      = len(metas)

    if verbose:
        print(f"  Profiling {n} tensors at {len(CODEBOOK_LEVELS)} codebook levels ...")
        print(f"  (sampling up to {sample_size:,} elements per tensor)")

    profiles: list[TensorProfile] = []
    for i, meta in enumerate(metas):
        if verbose:
            print(
                f"  [{i+1:4d}/{n}  {(i+1)/n*100:5.1f}%]  {meta.name:<60}",
                end="\r", flush=True,
            )

        gp     = gguf_by_name.get(meta.name)
        policy = gp.policy if gp else "optimize"

        levels: dict[str, LevelResult] = {
            "F16": LevelResult(snr_db=96.0, estimated_bytes=meta.num_params * 2)
        }

        if policy != "always_f16":
            raw_data = loader.load_tensor(meta).astype(np.float32)
            flat = raw_data.flatten()
            if flat.size > sample_size:
                idx  = np.random.default_rng(42).choice(flat.size, sample_size, replace=False)
                flat = flat[idx]

            # Lossless estimate: count unique float32 values in the sample.
            # BF16 has at most 65536 distinct bit patterns, so a 100K sample
            # captures nearly all unique values for typical trained weights.
            unique_count = max(int(len(np.unique(flat))), 2)
            lossless_bits = int(np.ceil(np.log2(unique_count)))
            lossless_bytes = (meta.num_params * lossless_bits + 7) // 8 + unique_count * 4
            levels["CB_lossless"] = LevelResult(snr_db=96.0, estimated_bytes=lossless_bytes)

            for cb in CODEBOOK_LEVELS:
                if policy == "always_q8_min" and cb.bits < 8:
                    continue
                dq      = cb.simulate(flat)
                signal  = float(np.mean(flat ** 2))
                noise   = float(np.mean((flat - dq) ** 2))
                snr     = 96.0 if noise < 1e-12 else float(
                    np.clip(10.0 * np.log10(signal / noise), 0.0, 96.0)
                )
                levels[cb.name] = LevelResult(
                    snr_db=snr,
                    estimated_bytes=cb.estimated_bytes(meta.num_params),
                )

        profiles.append(TensorProfile(
            name=meta.name, shape=list(meta.shape),
            num_params=meta.num_params,
            tensor_class=meta.tensor_class,
            policy=policy, levels=levels,
        ))

    if verbose:
        print(" " * 80, end="\r")

    return profiles


def _add_lossless_estimates(
    profiles: list[TensorProfile],
    model_path: Path,
    verbose: bool,
) -> None:
    """Add CB_lossless level to any profile missing it.

    Counts unique float32 values in the FULL tensor (no sampling) so the
    estimate is accurate — BF16 has at most 65536 patterns, so loading the
    full tensor is cheap and avoids undercounting rare values.
    Modifies profiles in-place; does not rewrite the cache file.
    """
    missing = [p for p in profiles if "CB_lossless" not in p.levels and p.policy != "always_f16"]
    if not missing:
        return

    from src.tensor_loader import ModelLoader
    loader   = ModelLoader(model_path)
    by_name  = {p.name: p for p in missing}
    n        = len(missing)
    done     = 0

    for meta in loader.tensors():
        p = by_name.get(meta.name)
        if p is None:
            continue
        if verbose:
            done += 1
            print(f"  lossless est [{done:4d}/{n}]  {meta.name:<60}", end="\r", flush=True)

        # Load full tensor to count all unique BF16 values
        flat         = loader.load_tensor(meta).astype(np.float32).flatten()
        unique_count = max(int(len(np.unique(flat))), 2)
        bits         = int(np.ceil(np.log2(unique_count)))
        est_bytes    = (meta.num_params * bits + 7) // 8 + unique_count * 4
        p.levels["CB_lossless"] = LevelResult(snr_db=96.0, estimated_bytes=est_bytes)

    if verbose:
        print(" " * 80, end="\r")


def load_cb_profiles(
    model_path: Path,
    gguf_profiles: list[TensorProfile],
    sample_size: int,
    force: bool,
    verbose: bool,
) -> list[TensorProfile]:
    cache = model_path / _CB_CACHE_NAME
    if cache.exists() and not force:
        if verbose:
            print(f"  Loading codebook profile from {cache.name}")
        with open(cache) as f:
            profiles = _deserialise_cb(json.load(f))
        _add_lossless_estimates(profiles, model_path, verbose)
        return profiles

    profiles = _run_cb_profiler(model_path, gguf_profiles, sample_size, verbose)

    raw = [
        {
            "name": p.name, "shape": p.shape, "num_params": p.num_params,
            "tensor_class": p.tensor_class, "policy": p.policy,
            "levels": {k: {"snr_db": v.snr_db, "estimated_bytes": v.estimated_bytes}
                       for k, v in p.levels.items()},
        }
        for p in profiles
    ]
    with open(cache, "w") as f:
        json.dump(raw, f, indent=2)
    if verbose:
        print(f"  Codebook profile saved to {cache.name}")

    return profiles


# ---------------------------------------------------------------------------
# Level-mix helper (works for both GGUF and CB level names)
# ---------------------------------------------------------------------------

_GGUF_ORDER: list[str] = [q.name for q in QUANT_LEVELS]
_CB_ORDER:   list[str] = [cb.name for cb in CODEBOOK_LEVELS] + ["CB_lossless", "F16"]


def _level_mix_pct(
    assignment: dict[str, str], profiles: list[TensorProfile]
) -> dict[str, float]:
    total = sum(p.num_params for p in profiles)
    if total == 0:
        return {}
    by_level: dict[str, int] = {}
    for p in profiles:
        lvl = assignment.get(p.name, "F16")
        by_level[lvl] = by_level.get(lvl, 0) + p.num_params
    return {lvl: 100.0 * cnt / total for lvl, cnt in by_level.items()}


_DISPLAY_NAMES = {"CB_lossless": "lossless"}


def _format_mix(pct: dict[str, float], source: str, threshold: float = 1.0) -> str:
    order = _GGUF_ORDER if source == "GGUF" else _CB_ORDER
    known = set(order)
    parts = []
    for lvl in order:
        p = pct.get(lvl, 0.0)
        if p >= threshold:
            label = _DISPLAY_NAMES.get(lvl, lvl)
            parts.append(f"{p:.0f}%{label}")
    for lvl, p in sorted(pct.items()):
        if lvl not in known and p >= threshold:
            label = _DISPLAY_NAMES.get(lvl, lvl)
            parts.append(f"{p:.0f}%{label}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Uniform-row builders
# ---------------------------------------------------------------------------

def _gguf_uniform_rows(profiles: list[TensorProfile]) -> list[dict]:
    gguf_order = [q.name for q in QUANT_LEVELS]
    q8_idx     = gguf_order.index("Q8_0")
    rows = []
    for level in QUANT_LEVELS:
        target_idx = gguf_order.index(level.name)
        assignment: dict[str, str] = {}
        for p in profiles:
            if p.policy == "always_f16":
                assignment[p.name] = "F16"
            elif p.policy == "always_q8_min":
                assignment[p.name] = level.name if target_idx >= q8_idx else "Q8_0"
            else:
                assignment[p.name] = level.name if level.name in p.levels else "F16"
        snr   = compute_model_snr(assignment, profiles)
        total = compute_total_bytes(assignment, profiles)
        mix   = _level_mix_pct(assignment, profiles)
        rows.append({
            "kind": "standard", "source": "GGUF",
            "label": level.name,
            "total_bytes": total, "snr_db": snr, "mix_pct": mix,
        })
    return rows


def _cb_uniform_rows(cb_profiles: list[TensorProfile]) -> list[dict]:
    rows = []
    for cb in CODEBOOK_LEVELS:
        assignment: dict[str, str] = {}
        for p in cb_profiles:
            if p.policy == "always_f16":
                assignment[p.name] = "F16"
            elif p.policy == "always_q8_min":
                lvl = cb.name if cb.bits >= 8 else "CB8"
                assignment[p.name] = lvl if lvl in p.levels else "F16"
            else:
                assignment[p.name] = cb.name if cb.name in p.levels else "F16"
        snr   = compute_model_snr(assignment, cb_profiles)
        total = compute_total_bytes(assignment, cb_profiles)
        mix   = _level_mix_pct(assignment, cb_profiles)
        rows.append({
            "kind": "standard", "source": "CB",
            "label": cb.name,
            "total_bytes": total, "snr_db": snr, "mix_pct": mix,
        })

    # CB lossless: bit-exact reconstruction.  BF16 tensors typically use only
    # a few thousand of the 65536 possible BF16 patterns, so lossless storage
    # needs ~12-15 bits/weight — smaller than F16 while being bit-perfect.
    lossless_assign: dict[str, str] = {}
    for p in cb_profiles:
        if p.policy == "always_f16":
            lossless_assign[p.name] = "F16"
        else:
            lossless_assign[p.name] = "CB_lossless" if "CB_lossless" in p.levels else "F16"
    total = compute_total_bytes(lossless_assign, cb_profiles)
    snr   = compute_model_snr(lossless_assign, cb_profiles)
    mix   = _level_mix_pct(lossless_assign, cb_profiles)
    rows.append({
        "kind": "standard", "source": "CB",
        "label": "lossless",
        "total_bytes": total, "snr_db": snr, "mix_pct": mix,
    })

    return rows


# ---------------------------------------------------------------------------
# Mixed-row builder (deduplicated)
# ---------------------------------------------------------------------------

def _build_mixed_rows(profiles: list[TensorProfile], source: str) -> list[dict]:
    rows = []
    for floor in MIXED_SNR_FLOORS:
        assignment = optimize_for_snr(profiles, float(floor))
        mix = _level_mix_pct(assignment.levels, profiles)
        rows.append({
            "kind": "mixed", "source": source,
            "label": f"≥{floor}dB",
            "floor": floor,
            "total_bytes": assignment.total_bytes,
            "snr_db": assignment.model_snr_db,
            "mix_pct": mix,
            "achievable": assignment.model_snr_db >= floor - 0.1,
        })

    # Deduplicate: multiple floors can collapse to the same result at the model ceiling
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (row["total_bytes"], round(row["snr_db"], 1))
        if key not in seen or row["floor"] > seen[key]["floor"]:
            seen[key] = row
    return sorted(seen.values(), key=lambda r: -r["floor"])


# ---------------------------------------------------------------------------
# "vs nearest standard" comparison
# ---------------------------------------------------------------------------

def _vs_nearest_std(row: dict, all_std: list[dict]) -> str | None:
    """Compare a mixed row against the cheapest standard achieving similar SNR."""
    target = row["snr_db"]
    candidates = [r for r in all_std if r["snr_db"] >= target - 0.5]
    if not candidates:
        return None
    ref = min(candidates, key=lambda r: r["total_bytes"])
    if ref["total_bytes"] == 0:
        return None
    delta = (row["total_bytes"] - ref["total_bytes"]) / ref["total_bytes"] * 100
    sign  = "+" if delta > 0 else ""
    return f"{sign}{delta:.0f}% vs {ref['label']}"


# ---------------------------------------------------------------------------
# Main printer
# ---------------------------------------------------------------------------

def print_unified_survey(
    gguf_profiles: list[TensorProfile],
    cb_profiles:   list[TensorProfile],
    model_path:    Path,
) -> None:
    base         = model_path.name
    n_total      = len(gguf_profiles)
    n_f16        = sum(1 for p in gguf_profiles if p.policy == "always_f16")
    n_q8         = sum(1 for p in gguf_profiles if p.policy == "always_q8_min")
    n_opt        = sum(1 for p in gguf_profiles if p.policy == "optimize")
    total_params = sum(p.num_params for p in gguf_profiles)
    f16_bytes    = sum(p.num_params * 2 for p in gguf_profiles)

    print()
    print("=" * 92)
    print(f"  UNIFIED SNR SURVEY: {base}")
    print("=" * 92)
    print(
        f"  {n_total} tensors  |  {total_params/1e9:.2f}B params  |  "
        f"{f16_bytes/1e9:.1f} GB F16 baseline"
    )
    print(
        f"  {n_f16} always-F16 (layernorms/biases)  |  "
        f"{n_q8} always-Q8_0-min (embeddings)  |  "
        f"{n_opt} optimizable"
    )
    print()

    print("  Building GGUF standard rows ...", end="\r", flush=True)
    gguf_std   = _gguf_uniform_rows(gguf_profiles)
    print("  Building codebook standard rows ...", end="\r", flush=True)
    cb_std     = _cb_uniform_rows(cb_profiles)
    print("  Running GGUF mixed optimizer ...", end="\r", flush=True)
    gguf_mixed = _build_mixed_rows(gguf_profiles, "GGUF")
    print("  Running codebook mixed optimizer ...", end="\r", flush=True)
    cb_mixed   = _build_mixed_rows(cb_profiles, "CB")
    print(" " * 55, end="\r")

    all_std  = gguf_std + cb_std
    all_rows = gguf_std + cb_std + gguf_mixed + cb_mixed
    # Sort descending by SNR; ties: standard before mixed
    all_rows.sort(key=lambda r: (-r["snr_db"], 0 if r["kind"] == "standard" else 1))

    print(
        f"  {'Type':<10}  {'Label':<12}  {'Size':>8}  {'SNR':>8}  "
        f"{'vs nearest std':>20}  {'Level mix (% of params)'}"
    )
    print(
        f"  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}  "
        f"{'-'*20}  {'-'*40}"
    )

    for row in all_rows:
        gb   = row["total_bytes"] / 1e9
        snr  = row["snr_db"]
        src  = row["source"]
        mix  = _format_mix(row["mix_pct"], src)

        if row["kind"] == "standard":
            tag    = f"{src} std"
            vs_str = ""
            flag   = ""
        else:
            tag    = f"{src} mix"
            vs_str = _vs_nearest_std(row, all_std) or ""
            flag   = ""
            if vs_str.startswith("-"):
                flag = " ◀"
            if not row.get("achievable", True):
                flag = " †"

        print(
            f"  {tag:<10}  {row['label']:<12}  {gb:>7.1f}GB  {snr:>6.1f}dB  "
            f"{vs_str:>20}{flag:<2}  {mix}"
        )

    print()
    print("  ◀ = mixed is smaller than nearest same-quality standard quant")
    print("  † = SNR floor exceeds model ceiling; best achievable shown")
    print()
    print("  Crossover: codebook wins on size below ~20 dB; GGUF wins above ~20 dB.")
    print()

    print_snr_winners(all_rows, base_name=base)
    print_size_winners(all_rows, f16_bytes, base_name=base)

    print("=" * 92)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified GGUF + codebook SNR survey.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_path", type=Path, help="Path to model directory")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-profile even if caches exist (snr_profile.json / codebook_profile.json)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=100_000,
        help="Max elements to sample per tensor (default: 100000)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        print(f"Error: {model_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    verbose = not args.quiet

    # GGUF profiles (cached in snr_profile.json)
    profiler      = Profiler(model_path, sample_size=args.sample_size, verbose=verbose)
    gguf_profiles = profiler.run(force=args.force)

    # Codebook profiles (cached in codebook_profile.json)
    cb_profiles = load_cb_profiles(
        model_path, gguf_profiles,
        sample_size=args.sample_size,
        force=args.force,
        verbose=verbose,
    )

    print_unified_survey(gguf_profiles, cb_profiles, model_path)


if __name__ == "__main__":
    main()
