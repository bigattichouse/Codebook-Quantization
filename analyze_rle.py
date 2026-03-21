#!/usr/bin/env python3
"""
Analyze compression headroom in packed codebook index streams.

Tests zstd on raw index arrays to estimate the theoretical ceiling
for entropy-based improvements (Huffman, ANS, etc.).

Usage:
    python analyze_rle.py [codebook_dir]
    python analyze_rle.py /path/to/model/codebook
"""

import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from bitpack import unpack_any_bits

try:
    import zstandard as _zstd_mod
    class zstd:
        @staticmethod
        def compress(data, level=3):
            return _zstd_mod.ZstdCompressor(level=level).compress(data)
    HAS_ZSTD = True
except ImportError:
    try:
        import zstd
        HAS_ZSTD = True
    except ImportError:
        HAS_ZSTD = False
        print("WARNING: no zstd library found. Install with: pip install zstandard")


def entropy_bits(arr):
    """Shannon entropy in bits/symbol of a 1D integer array."""
    counts = np.bincount(arr.astype(np.int64)).astype(np.float64)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def analyze_model(codebook_dir: str):
    tensor_dir = Path(codebook_dir) / "tensors"
    if not tensor_dir.exists():
        print(f"ERROR: tensor dir not found: {tensor_dir}")
        sys.exit(1)

    npz_files = sorted(tensor_dir.glob("*.npz"))
    print(f"Analyzing {len(npz_files)} tensors in {codebook_dir}\n")

    hdr = f"{'Tensor':<55} {'mode':<14} {'bits':>4} {'params':>10} {'cur_B':>10} {'zstd_B':>10} {'save%':>6} {'entropy':>8} {'ceiling_b':>10}"
    print(hdr)
    print("-" * len(hdr))

    # Accumulators per mode
    totals = {}  # mode -> {packed, zstd, params, entropy_sum, entropy_n}

    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        name = str(d.get("name", f.stem))
        mode = str(d.get("mode", "unknown"))
        short = name[-53:] if len(name) > 53 else name

        def acc(m, packed_b, zstd_b, params, ent=None, ceiling_b=None):
            t = totals.setdefault(m, dict(packed=0, zstd=0, params=0, entropy_sum=0.0, entropy_n=0, ceiling=0))
            t["packed"] += packed_b
            t["zstd"] += zstd_b
            t["params"] += params
            if ent is not None:
                t["entropy_sum"] += ent
                t["entropy_n"] += 1
            if ceiling_b is not None:
                t["ceiling"] += ceiling_b

        if mode == "direct_codebook":
            indices_packed = d["indices"]
            bits = int(d["bits"])
            shape = list(d["shape"])
            n = int(np.prod(shape))

            packed_b = len(indices_packed)
            indices_raw = unpack_any_bits(indices_packed, bits, n)

            ent = entropy_bits(indices_raw)
            # theoretical minimum bytes if we could encode at entropy rate
            ceiling_b = int(np.ceil(ent * n / 8))

            if HAS_ZSTD:
                zstd_b = len(zstd.compress(indices_packed.tobytes(), 3))
            else:
                zstd_b = ceiling_b  # fall back to theoretical

            save_pct = (1 - zstd_b / packed_b) * 100 if packed_b else 0
            print(f"{short:<55} {mode:<14} {bits:>4} {n:>10,} {packed_b:>10,} {zstd_b:>10,} {save_pct:>5.1f}% {ent:>8.3f} {ceiling_b:>10,}")
            acc(mode, packed_b, zstd_b, n, ent, ceiling_b)

        elif mode == "exact":
            data = d["data"]
            raw_b = len(data.tobytes())
            zstd_b = len(zstd.compress(data.tobytes(), 3)) if HAS_ZSTD else raw_b
            save_pct = (1 - zstd_b / raw_b) * 100 if raw_b else 0
            print(f"{short:<55} {'exact':<14} {'--':>4} {data.size:>10,} {raw_b:>10,} {zstd_b:>10,} {save_pct:>5.1f}% {'--':>8} {'--':>10}")
            acc("exact", raw_b, zstd_b, data.size)

        elif mode == "linear_quant":
            indices = d["indices"]
            raw_b = len(indices.tobytes())
            zstd_b = len(zstd.compress(indices.tobytes(), 3)) if HAS_ZSTD else raw_b
            save_pct = (1 - zstd_b / raw_b) * 100 if raw_b else 0
            ent = entropy_bits(indices.flatten())
            print(f"{short:<55} {'linear_quant':<14} {'8':>4} {indices.size:>10,} {raw_b:>10,} {zstd_b:>10,} {save_pct:>5.1f}% {ent:>8.3f} {'--':>10}")
            acc("linear_quant", raw_b, zstd_b, indices.size, ent)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("\nSUMMARY BY MODE:")
    grand_packed = grand_zstd = grand_params = grand_ceiling = 0
    for mode, t in totals.items():
        p, z, n, c = t["packed"], t["zstd"], t["params"], t["ceiling"]
        save_zstd = (1 - z / p) * 100 if p else 0
        save_ceil = (1 - c / p) * 100 if p and c else float("nan")
        avg_ent = t["entropy_sum"] / t["entropy_n"] if t["entropy_n"] else float("nan")
        cur_bpp  = p * 8 / n if n else 0
        ceil_bpp = c * 8 / n if n and c else float("nan")
        print(f"  {mode:<16} {p:>12,} B  cur={cur_bpp:.3f} b/param  "
              f"avg_entropy={avg_ent:.3f} b/sym  entropy_ceil={ceil_bpp:.3f} b/param  ({save_ceil:.1f}% headroom)")
        grand_packed += p; grand_zstd += z; grand_params += n; grand_ceiling += c

    if grand_packed:
        cur_bpp   = grand_packed  * 8 / grand_params
        ceil_bpp  = grand_ceiling * 8 / grand_params if grand_ceiling else float("nan")
        headroom  = (1 - grand_ceiling / grand_packed) * 100 if grand_ceiling else float("nan")
        zstd_save = (1 - grand_zstd / grand_packed) * 100

        print(f"\n  {'TOTAL':<16} {grand_packed:>12,} B → entropy ceiling {grand_ceiling:>12,} B")
        print(f"\n  Current packing : {cur_bpp:.3f} bits/param")
        print(f"  Entropy ceiling : {ceil_bpp:.3f} bits/param  ({headroom:.1f}% potential reduction via Huffman/ANS on raw indices)")
        print(f"  zstd on packed  : {grand_zstd * 8 / grand_params:.3f} bits/param  ({zstd_save:.1f}% — low because LCM packing destroys index structure)")
        print(f"\n  KEY INSIGHT: Fixed-width bit-packing destroys index distribution. Entropy coding")
        print(f"  the raw index stream (before packing) could recover the full {headroom:.1f}% headroom.")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/bigattichouse/workspace/model/Qwen3.5-0.8B/codebook"
    analyze_model(model_path)
