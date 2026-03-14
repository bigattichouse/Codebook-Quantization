#!/usr/bin/env python3
"""
analyze.py — Estimate compressed model size without compressing.

Runs the histogram analysis pass (reads all weights, builds codebooks) and
reports per-mode size estimates, coverage, and expected MSE.

Usage:
    ./venv/bin/python proofofconcept/analyze.py ~/workspace/model/Qwen3.5-9B
    ./venv/bin/python proofofconcept/analyze.py ~/workspace/model/Qwen3.5-9B --quick
"""

import sys
import json
import math
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from compressor import classify_tensor

# ─── helpers ──────────────────────────────────────────────────────────────────

CRITICAL_PATTERNS = ('norm', 'layernorm', 'ln_', 'lm_head', 'embed_tokens',
                     'gate.weight', 'router', '.bias')

def is_critical(name: str, num_params: int) -> bool:
    name_l = name.lower()
    return num_params < 10_000 or any(p in name_l for p in CRITICAL_PATTERNS)

def bits_for_k(k: int) -> int:
    """Bits needed to index k codebook entries."""
    if k <= 0:
        return 16   # exact
    return max(1, math.ceil(math.log2(k)))

def compressed_bytes(num_params: int, bits: int) -> int:
    return math.ceil(num_params * bits / 8)

# Codebook sizes per tensor type per mode
MODE_CB_SIZES = {
    #              embedding  attention  mlp_ffn  moe_expert  (others = 0 = exact)
    'lossless':  {'embedding': 4096, 'attention': 8192, 'mlp_ffn': 8192, 'moe_expert': 8192},
    'balanced':  {'embedding': 4096, 'attention': 256,  'mlp_ffn': 256,  'moe_expert': 256},
    'aggressive':{'embedding': 4096, 'attention': 128,  'mlp_ffn': 128,  'moe_expert': 128},
}

# ─── header-only (quick) ──────────────────────────────────────────────────────

def quick_analyze(model_path: Path):
    st_files = sorted(model_path.glob('*.safetensors'))
    tensors = {}
    for st in st_files:
        with open(st, 'rb') as f:
            hdr_size = int.from_bytes(f.read(8), 'little')
            hdr = json.loads(f.read(hdr_size))
        for name, info in hdr.items():
            if name == '__metadata__':
                continue
            shape = tuple(info['shape'])
            n = 1
            for d in shape:
                n *= d
            dtype_bytes = 2 if info.get('dtype', 'BF16') in ('BF16', 'F16') else 1
            tensors[name] = {
                'shape': shape,
                'n': n,
                'bytes': n * dtype_bytes,
                'type': classify_tensor(name),
                'critical': is_critical(name, n),
            }

    total_params = sum(t['n'] for t in tensors.values())
    total_bytes = sum(t['bytes'] for t in tensors.values())
    critical_bytes = sum(t['bytes'] for t in tensors.values() if t['critical'])
    compressible_params = sum(t['n'] for t in tensors.values() if not t['critical'])

    print(f"\n{'='*70}")
    print(f"QUICK ANALYSIS (header scan only — run without --quick for MSE estimates)")
    print(f"{'='*70}")
    print(f"Model  : {model_path.name}")
    print(f"Shards : {len(st_files)}")
    print(f"Tensors: {len(tensors)}")
    print(f"Params : {total_params/1e9:.3f}B")
    print(f"Size   : {total_bytes/1e9:.2f} GB  (bfloat16)")

    critical_count = sum(1 for t in tensors.values() if t['critical'])
    print(f"Critical (kept exact): {critical_count} tensors, {critical_bytes/1e9:.2f} GB")
    print()

    print(f"{'Mode':<12} {'CB bits':>7} {'Compressed':>12} {'Savings':>10} {'Bits/w':>8}")
    print(f"{'-'*12} {'-'*7} {'-'*12} {'-'*10} {'-'*8}")

    for mode, cb_sizes in MODE_CB_SIZES.items():
        comp_bytes = critical_bytes   # critical tensors already counted at full 16-bit
        total_bits = 0
        for t in tensors.values():
            ttype = t['type']
            k = cb_sizes.get(ttype, 0)
            if t['critical']:
                bits = 16
                # already in comp_bytes — just count bits for avg
            elif k == 0:
                bits = 16
                comp_bytes += compressed_bytes(t['n'], bits)
            else:
                bits = bits_for_k(k)
                comp_bytes += compressed_bytes(t['n'], bits)
            total_bits += t['n'] * bits

        savings = (1 - comp_bytes / total_bytes) * 100
        avg_bits = total_bits / total_params
        print(f"{mode:<12} {'varied':>7} {comp_bytes/1e9:>10.2f}GB {savings:>9.1f}% {avg_bits:>7.2f}")

    print()
    print("Note: 'lossless' uses large codebooks → ~13 bits/weight → modest savings.")
    print("      'balanced' uses 8-bit codebooks → ~8 bits/weight → ~50% savings.")
    print("      MSE estimates require --full (reads all weights).")
    print()


# ─── full (histogram-based) ───────────────────────────────────────────────────

def _simulate_mode(tensor_info: dict, codebooks_by_mode: dict, mode: str) -> dict:
    """Estimate compressed size and MSE for a given mode's codebooks."""
    codebooks = codebooks_by_mode[mode]
    total_orig = 0
    total_comp = 0
    total_sq_err = 0.0
    total_params_c = 0
    category_stats = {}

    for name, info in tensor_info.items():
        n = 1
        for d in info['shape']: n *= d
        orig = n * 2  # bfloat16
        ttype = info['type']
        cb = codebooks.get(ttype)
        total_orig += orig

        if is_critical(name, n) or cb is None:
            total_comp += orig
            continue

        k = len(cb)
        bits = bits_for_k(k)
        comp_sz = compressed_bytes(n, bits) + k * 4  # indices + codebook floats
        total_comp += comp_sz
        total_params_c += n

        # Quantization noise: Δ²/12 where Δ = spacing between codebook centroids
        if k > 1:
            delta = float(cb.max() - cb.min()) / (k - 1)
            mse_est = (delta ** 2) / 12.0
        else:
            mse_est = 0.0
        total_sq_err += mse_est * n

        s = category_stats.setdefault(ttype, {'n': 0, 'k': 0, 'bits': 0})
        s['n'] += n
        s['k'] = k
        s['bits'] = bits

    savings = (1 - total_comp / total_orig) * 100 if total_orig > 0 else 0
    avg_mse = total_sq_err / total_params_c if total_params_c > 0 else 0.0
    snr_db = -10 * math.log10(avg_mse) if avg_mse > 0 else float('inf')
    return {
        'orig_gb': total_orig / 1e9,
        'comp_gb': total_comp / 1e9,
        'savings_pct': savings,
        'avg_mse': avg_mse,
        'snr_db': snr_db,
        'category_stats': category_stats,
    }


def full_analyze(model_path: Path):
    """Run histogram analysis once, then estimate all three modes."""
    from adaptive_compressor import AdaptiveCompressor

    print(f"\n{'='*70}")
    print(f"FULL ANALYSIS — reads all weights, builds codebooks, estimates MSE")
    print(f"{'='*70}")
    print(f"Model: {model_path.name}")
    print()

    # One histogram pass (mode-independent) using 'lossless' to get large codebooks.
    # Smaller-mode codebooks will be subsets of this.
    print("Building histograms (reads all weights once)...")
    comp_base = AdaptiveCompressor(model_path, compression_mode='lossless',
                                   store_in_model=True, force_rebuild=False)
    comp_base.model_hash = comp_base._compute_model_hash()
    comp_base._analyze_and_build_codebooks()
    tensor_info = comp_base.tensor_info

    # Build codebooks for each mode separately (fast — works on histograms already loaded)
    codebooks_by_mode = {'lossless': comp_base.codebooks}
    for mode, internal in (('balanced', 'balanced'), ('aggressive', 'lossy')):
        c = AdaptiveCompressor(model_path, compression_mode=internal,
                               store_in_model=True, force_rebuild=False)
        c.model_hash = comp_base.model_hash
        c._analyze_and_build_codebooks()
        codebooks_by_mode[mode] = c.codebooks

    results = {}
    for mode in ('lossless', 'balanced', 'aggressive'):
        r = _simulate_mode(tensor_info, codebooks_by_mode, mode)
        results[mode] = r
        print(f"\n── {mode} ──")
        print(f"  Size   : {r['comp_gb']:.2f} GB  (vs {r['orig_gb']:.2f} GB original)")
        print(f"  Savings: {r['savings_pct']:.1f}%")
        print(f"  MSE    : {r['avg_mse']:.2e}  (SNR {r['snr_db']:.1f} dB)" if math.isfinite(r['snr_db'])
              else f"  MSE    : {r['avg_mse']:.2e}  (lossless)")
        print(f"  Per category:")
        for ttype, s in sorted(r['category_stats'].items()):
            print(f"    {ttype:<14}: k={s['k']:>5}  →  {s['bits']}-bit  ({s['n']/1e6:.0f}M params)")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mode':<12} {'Size':>10} {'Savings':>10} {'Avg MSE':>12} {'SNR dB':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for mode, r in results.items():
        snr = f"{r['snr_db']:.1f}" if math.isfinite(r['snr_db']) else "∞"
        print(f"{mode:<12} {r['comp_gb']:>8.2f}GB {r['savings_pct']:>9.1f}% "
              f"{r['avg_mse']:>12.2e} {snr:>10}")
    print()

    # VRAM check
    try:
        import subprocess
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total',
                                       '--format=csv,noheader,nounits'], text=True)
        vram_mb = int(out.strip().split('\n')[0])
        vram_gb = vram_mb / 1024
        print(f"GPU VRAM: {vram_gb:.1f} GB")
        for mode, r in results.items():
            fits = "FITS" if r['comp_gb'] < vram_gb * 0.85 else "TOO LARGE"
            print(f"  {mode:<12}: {r['comp_gb']:.2f} GB  →  {fits}")
    except Exception:
        pass
    print()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Estimate compressed model size')
    parser.add_argument('model_path', help='Path to the model directory')
    parser.add_argument('--quick', action='store_true',
                        help='Header scan only — no weight reading, no MSE (very fast)')
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        print(f"Error: {model_path} does not exist")
        sys.exit(1)

    if not list(model_path.glob('*.safetensors')):
        print(f"Error: no .safetensors files in {model_path}")
        sys.exit(1)

    if args.quick:
        quick_analyze(model_path)
    else:
        full_analyze(model_path)


if __name__ == '__main__':
    main()
