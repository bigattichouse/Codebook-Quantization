#!/usr/bin/env python3
"""
survey.py — Codebook compression survey for a model.

Shows two complementary views:

  1. EXISTING CACHES — what you've already compressed at each --db target.
     Reads each codebook-*dB/ directory, measures disk size, reports the
     SNR target and any stored accuracy stats.

  2. LEVEL REFERENCE — per-bit-depth uniform codebook SNR/size estimates.
     If quantization/codebook_profile.json exists next to the model, reads
     the measured per-tensor SNR values (computed by compare_codebook.py)
     to show what each CB level actually achieves on this model.
     Falls back to a quick header-based size estimate if not available.

Usage:
    ./venv/bin/python survey.py ~/workspace/model/Qwen3.5-9B
    ./venv/bin/python survey.py ~/workspace/model/Qwen3.5-9B --profile
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))


# ---------------------------------------------------------------------------
# Existing cache scanner
# ---------------------------------------------------------------------------

def _scan_existing_caches(model_path: Path) -> list[dict]:
    """Find all codebook-*dB/ directories and measure their sizes."""
    caches = []

    for entry in sorted(model_path.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name

        if name.startswith("codebook-") and name.endswith("dB"):
            db_str = name[len("codebook-"):-len("dB")]
            try:
                db_val = float(db_str)
            except ValueError:
                db_val = None
            label = f"{db_str} dB"
        elif name == "codebook-lossless":
            db_val = None
            label = "lossless"
        else:
            continue

        # Disk size (compressed npz on disk)
        disk_bytes = sum(
            f.stat().st_size
            for f in entry.rglob("*")
            if f.is_file()
        )

        # Try to get accuracy stats from metadata
        meta_path = entry / "metadata.json"
        snr_info = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                acc = meta.get("accuracy_stats", {})
                if acc:
                    snr_info = acc
            except Exception:
                pass

        caches.append({
            "label": label,
            "db": db_val,
            "disk_gb": disk_bytes / 1e9,
            "snr_info": snr_info,
            "path": str(entry),
        })

    caches.sort(key=lambda c: (c["db"] is None, -(c["db"] or 0)))
    return caches


# ---------------------------------------------------------------------------
# CB level reference (from codebook_profile.json)
# ---------------------------------------------------------------------------

def _load_cb_profile(model_path: Path) -> list[dict] | None:
    """Load per-level SNR/size data from quantization/codebook_profile.json."""
    profile_path = model_path / "codebook_profile.json"
    if not profile_path.exists():
        return None

    with open(profile_path) as f:
        raw = json.load(f)

    # Collect all CB level names present
    cb_levels: dict[str, dict] = {}   # level_name -> {total_bytes, snr_pairs}

    # Collect per-level contributions from all tensor policies
    f16_fixed_bytes = 0    # always_f16 tensors: fixed at F16 regardless of CB level

    # always_q8_min tensors: use their CB level entry when available (CB8+),
    # otherwise fall back to CB8 (the policy minimum for codebook)
    q8_min_by_level: dict[str, dict] = {}  # level -> {bytes, snr_pairs}

    for tensor in raw:
        policy = tensor.get("policy", "optimize")
        num_params = tensor.get("num_params", 0)
        levels = tensor.get("levels", {})

        if policy == "always_f16":
            f16_fixed_bytes += levels.get("F16", {}).get("estimated_bytes", num_params * 2)
            continue

        if policy == "always_q8_min":
            # For each CB level, use that level's data if available (CB8+),
            # else use CB8 minimum
            cb8_lr = levels.get("CB8") or levels.get("F16")
            for lvl_name, lr in levels.items():
                if not lvl_name.startswith("CB"):
                    continue
                if lvl_name not in q8_min_by_level:
                    q8_min_by_level[lvl_name] = {"total_bytes": 0, "snr_pairs": []}
                q8_min_by_level[lvl_name]["total_bytes"] += lr["estimated_bytes"]
                q8_min_by_level[lvl_name]["snr_pairs"].append((lr["snr_db"], num_params))
            # For levels below CB8 (not profiled), record using CB8 as minimum
            if cb8_lr:
                for bits in [3, 4, 5, 6, 7]:
                    key = f"CB{bits}"
                    if key not in q8_min_by_level:
                        q8_min_by_level[key] = {"total_bytes": 0, "snr_pairs": []}
                    q8_min_by_level[key]["total_bytes"] += cb8_lr["estimated_bytes"]
                    q8_min_by_level[key]["snr_pairs"].append((cb8_lr["snr_db"], num_params))
            continue

        for lvl_name, lr in levels.items():
            if not lvl_name.startswith("CB"):
                continue
            if lvl_name not in cb_levels:
                cb_levels[lvl_name] = {"total_bytes": 0, "snr_pairs": []}
            cb_levels[lvl_name]["total_bytes"] += lr["estimated_bytes"]
            cb_levels[lvl_name]["snr_pairs"].append((lr["snr_db"], num_params))

    # Compute P5 model SNR for each level (include always_q8_min pairs too)
    results = []
    for lvl_name, data in sorted(cb_levels.items(), key=lambda x: int(x[0][2:])):
        q8_entry = q8_min_by_level.get(lvl_name, {"total_bytes": 0, "snr_pairs": []})
        all_pairs = sorted(data["snr_pairs"] + q8_entry["snr_pairs"], key=lambda x: x[0])
        total_w = sum(w for _, w in all_pairs)
        target = total_w * 5.0 / 100.0
        cumulative = 0.0
        model_snr = all_pairs[-1][0] if all_pairs else 0.0
        for snr, w in all_pairs:
            cumulative += w
            if cumulative >= target:
                model_snr = snr
                break
        bits = int(lvl_name[2:])
        total_gb = (f16_fixed_bytes + q8_entry["total_bytes"] + data["total_bytes"]) / 1e9
        results.append({
            "name": lvl_name,
            "bits": bits,
            "size_gb": total_gb,
            "model_snr": model_snr,
        })

    return results if results else None


# ---------------------------------------------------------------------------
# Quick header-based size estimate (fallback when no profile)
# ---------------------------------------------------------------------------

def _quick_size_estimate(model_path: Path, bits: int) -> float:
    """Rough size estimate: all params at `bits` bpw + centroid table per tensor."""
    st_files = sorted(model_path.glob("*.safetensors"))
    total_bytes = 0
    for st in st_files:
        with open(st, "rb") as f:
            hdr_size = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(hdr_size))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            n = math.prod(info.get("shape", [1]))
            # Critical tensors stay at F16
            name_l = name.lower()
            is_critical = n < 10_000 or any(
                p in name_l for p in ("norm", "layernorm", "ln_", "bias")
            )
            if is_critical:
                total_bytes += n * 2
            else:
                index_bytes = (n * bits + 7) // 8
                cb_bytes = (1 << bits) * 4
                total_bytes += index_bytes + cb_bytes
    return total_bytes / 1e9


# ---------------------------------------------------------------------------
# GGUF Survey
# ---------------------------------------------------------------------------

def _survey_gguf_model(model_path: Path) -> bool:
    """Inspect a GGUF model or directory of GGUF shards and display survey stats."""
    if model_path.is_file() and (".gguf" in model_path.name):
        gguf_files = [model_path]
    elif model_path.is_dir():
        gguf_files = sorted(model_path.glob("*.gguf*"))
    else:
        return False

    if not gguf_files:
        return False

    try:
        import gguf
    except ImportError:
        print("  Notice: 'gguf' package not installed; run: pip install gguf")
        return False

    total_disk_bytes = sum(f.stat().st_size for f in gguf_files)
    total_disk_gb = total_disk_bytes / 1e9

    # Inspect shard 1 or single file for metadata
    try:
        reader = gguf.GGUFReader(str(gguf_files[0]))
    except Exception as e:
        print(f"  Warning: Failed to read GGUF metadata from {gguf_files[0].name}: {e}")
        return False

    arch_field = reader.get_field("general.architecture")
    arch_str = (
        bytes(arch_field.parts[-1]).decode("utf-8", errors="ignore")
        if arch_field
        else "unknown"
    )
    name_field = reader.get_field("general.name")
    name_str = (
        bytes(name_field.parts[-1]).decode("utf-8", errors="ignore")
        if name_field
        else model_path.stem
    )

    def _get_meta_int(key: str) -> int | None:
        f = reader.get_field(f"{arch_str}.{key}")
        if f and len(f.parts) > 0:
            try:
                return int(f.parts[-1][0])
            except Exception:
                return None
        return None

    block_count = _get_meta_int("block_count") or 48
    ctx_len = _get_meta_int("context_length")
    embed_dim = _get_meta_int("embedding_length") or 2560
    expert_count = _get_meta_int("expert_count")
    expert_used = _get_meta_int("expert_used_count")
    expert_ffn = _get_meta_int("expert_feed_forward_length")

    # Count parameters across all readable shards
    total_params = 0
    tensors_counted = 0
    for gf in gguf_files:
        try:
            r = gguf.GGUFReader(str(gf))
            for t in r.tensors:
                total_params += math.prod(t.shape)
                tensors_counted += 1
        except Exception:
            pass

    # If some shards are still downloading or unreadable, calculate from architecture
    if tensors_counted < 50:
        if expert_count and expert_used and expert_ffn:
            expert_params = expert_count * 3 * embed_dim * expert_ffn * block_count
            dense_params = block_count * (4 * embed_dim * embed_dim) + (200_000 * embed_dim)
            total_params = expert_params + dense_params
            active_params = (expert_used * 3 * embed_dim * expert_ffn * block_count) + dense_params
        else:
            total_params = int((total_disk_bytes * 8) / 4.5)
            active_params = total_params
    else:
        if expert_count and expert_used and expert_ffn:
            expert_params = expert_count * 3 * embed_dim * expert_ffn * block_count
            active_expert_params = expert_used * 3 * embed_dim * expert_ffn * block_count
            dense_params = max(0, total_params - expert_params)
            active_params = dense_params + active_expert_params
        else:
            active_params = total_params

    total_params_b = total_params / 1e9
    active_params_b = active_params / 1e9

    print()
    print("=" * 76)
    print(f"  GGUF MODEL SURVEY: {name_str}")
    print("=" * 76)
    print(f"  Architecture   : {arch_str}")
    print(f"  Layers (blocks): {block_count}")
    print(f"  Context window : {ctx_len:,} tokens" if ctx_len else f"  Embedding dim  : {embed_dim}")
    if expert_count:
        print(f"  MoE Routing    : {expert_count} experts total, {expert_used} active per token")
        print(f"  Total Params   : {total_params_b:.1f}B total ({active_params_b:.1f}B active per token)")
    else:
        print(f"  Total Params   : {total_params_b:.1f}B dense")
    print(f"  GGUF on disk   : {total_disk_gb:.1f} GB ({len(gguf_files)} shard file{'s' if len(gguf_files) > 1 else ''})")
    print()

    # Measured lossless headroom: Huffman on the stored integer quant codes
    from gguf_codes import print_headroom, scan_gguf

    print("  Lossless Headroom (measured, CPU-only block sampling):")
    print("  Huffman-codes the integer quant codes per tensor; block scales kept raw.")
    print()
    res = scan_gguf(model_path, max_blocks=2048, progress=False)
    print_headroom(res, context=False)
    saved_gb = (res["raw_bytes"] - res["huff_bytes"]) / 1e9
    lossless_gb = total_disk_gb - saved_gb
    print()

    # Hardware & MI50 Budget Analysis
    mi50_vram = 32.0
    usable_vram = 28.0  # leave 4GB for KV cache / context / workspace
    host_ram = 128.0

    print("  MI50 (32 GB VRAM) & System (128 GB RAM) Budget Analysis:")
    print(f"  - Target Hardware : AMD Instinct MI50 (32 GB HBM2) | Host RAM: {host_ram:.0f} GB")
    print(f"  - Model Total Size: {total_disk_gb:.1f} GB")

    if total_disk_gb <= usable_vram:
        print("  - VRAM Fit        : Full model fits entirely in 32 GB VRAM!")
        offload_layers = block_count
    else:
        vram_layer_ratio = usable_vram / total_disk_gb
        offload_layers = min(block_count, max(1, int(block_count * vram_layer_ratio)))
        vram_used = (total_disk_gb / block_count) * offload_layers
        cpu_ram_needed = total_disk_gb - vram_used
        print(f"  - VRAM Fit        : Model exceeds 32 GB VRAM ({total_disk_gb:.1f} GB > {mi50_vram:.0f} GB).")
        print(f"  - System RAM Fit  : Fits in host CPU RAM ({total_disk_gb:.1f} GB < {host_ram:.0f} GB).")
        print(f"  - Recommended Offload (llama.cpp):")
        print(f"      Offload {offload_layers} of {block_count} layers to MI50 VRAM (~{vram_used:.1f} GB VRAM).")
        print(f"      Remaining {block_count - offload_layers} layers execute in system RAM (~{cpu_ram_needed:.1f} GB).")
        if expert_count:
            print(f"      Because this is MoE (only {expert_used} active experts/token), memory bandwidth")
            print(f"      demand is only ~{active_params_b:.1f}B params/token during generation!")

    print()
    saved_pct = 100 * saved_gb / total_disk_gb
    print("  Can This Quant Be Shrunk Losslessly?")
    print(f"  - Measured: {total_disk_gb:.1f} GB -> {lossless_gb:.1f} GB ({saved_pct:.1f}% saved).")
    print("    Code histograms are shared model-wide but nearly flat: k-quant scale fitting")
    print("    spreads each block across the full code range, so little entropy is left.")
    print("  - Re-codebooking the dequantized floats does not help: every block has its own")
    print("    fp16 scale, so a tensor holds millions of distinct float values.")
    print("  - Larger levers are structural: mmap lookup-only tables (token / per-layer")
    print("    embeddings) and keep inactive MoE experts in host RAM.")
    print()
    print("  Recommended Command to Run:")
    print(f"    ~/workspace/llama.cpp/build/bin/llama-cli \\")
    print(f"      -m {gguf_files[0]} \\")
    print(f"      -ngl {offload_layers if total_disk_gb > usable_vram else 99} \\")
    print(f"      -p \"Hello\" -n 128")
    print("=" * 76)
    print()
    return True


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def print_survey(model_path: Path) -> None:
    if _survey_gguf_model(model_path):
        return

    base_name = model_path.name

    print()
    print("=" * 72)
    print(f"  CODEBOOK SURVEY: {base_name}")
    print("=" * 72)

    # ── 1. Existing caches ────────────────────────────────────────────────
    caches = _scan_existing_caches(model_path)

    if caches:
        print()
        print("  Existing compression caches:")
        print()
        print(f"  {'Target':<12}  {'Disk size':>10}  Notes")
        print(f"  {'-'*12}  {'-'*10}  {'-'*40}")
        for c in caches:
            snr_note = ""
            if c["snr_info"]:
                mn = c["snr_info"].get("min_snr_db")
                me = c["snr_info"].get("mean_snr_db")
                if mn is not None:
                    snr_note = f"SNR min={mn:.1f} mean={me:.1f} dB"
            print(f"  {c['label']:<12}  {c['disk_gb']:>8.1f} GB  {snr_note}")
        print()
        print("  Note: disk size is gzip-compressed .npz; decompressed runtime")
        print("  footprint is larger (indices stored as uint16 per tensor).")
    else:
        print()
        print("  No existing codebook caches found.")
        print("  Run:  ./venv/bin/python compress.py <model> --db 30")

    # ── 2. CB level reference ─────────────────────────────────────────────
    print()
    print("  Codebook level reference (uniform: all compressible tensors at same level):")
    print()

    cb_data = _load_cb_profile(model_path)

    if cb_data:
        print(f"  {'Level':<8}  {'bpw':>5}  {'Size':>9}  {'Model SNR (P5)':>16}  {'Source'}")
        print(f"  {'-'*8}  {'-'*5}  {'-'*9}  {'-'*16}  {'-'*20}")
        for row in cb_data:
            print(
                f"  {row['name']:<8}  {row['bits']:>5.1f}  "
                f"{row['size_gb']:>8.1f}GB  {row['model_snr']:>14.1f}dB  "
                f"measured (codebook_profile.json)"
            )
        print()
        print("  Run  quantization/compare_codebook.py <model>  to regenerate profile.")
    else:
        print(f"  {'Level':<8}  {'bpw':>5}  {'Size (est.)':>12}  {'Source'}")
        print(f"  {'-'*8}  {'-'*5}  {'-'*12}  {'-'*30}")
        for bits in [3, 4, 5, 6, 7, 8, 10, 12]:
            sz = _quick_size_estimate(model_path, bits)
            print(
                f"  CB{bits:<6}  {bits:>5.1f}  {sz:>10.1f}GB  "
                f"estimated (header scan, no SNR)"
            )
        print()
        print("  For measured SNR values, run:")
        print("    python quantization/compare_codebook.py <model>")
        print("  Then re-run this survey.")

    # ── 3. Mixed-precision reminder ────────────────────────────────────────
    print()
    print("  For mixed-precision codebook assignment at a specific SNR floor:")
    print("    python quantization/compare_codebook.py <model>")
    print()
    print("  For GGUF mixed-precision survey:")
    print("    python quantization/snr_quant.py <model> --survey")
    print()
    print("  Reference document: spec/QUANT_LEVELS_REFERENCE.md")
    print()
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Survey codebook compression options for a model."
    )
    parser.add_argument("model_path", type=Path, help="Path to the model directory")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"Error: {model_path} does not exist", file=sys.stderr)
        sys.exit(1)

    print_survey(model_path)


if __name__ == "__main__":
    main()
