#!/usr/bin/env python3
"""
compress.py — Compress a model to codebook format.

CPU-only: reads safetensors, builds codebooks, saves compressed tensors.
Does NOT load model architecture or touch the GPU.
Run this before chat.py.

Quality is specified as an SNR target in dB (signal-to-noise ratio).
Higher dB = better reconstruction, larger cache.  Modes are aliases:
  lossless → bit-perfect (MSE ≤ 1e-9, no SNR limit)
  balanced → 30 dB  (≈ 1000:1 signal/noise — recommended)
  lossy    → 25 dB  (≈ 316:1 signal/noise — smaller, still usable)

You can specify any dB target directly with --db:
  --db 35   →  codebook-35dB/   (high quality, near-lossless size)
  --db 30   →  codebook-30dB/   (same as balanced)
  --db 25   →  codebook-25dB/   (same as lossy)
  --db 20   →  codebook-20dB/   (aggressive, noticeable PPL increase)

Multiple quality tiers can coexist under the same model directory.

Usage:
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --mode lossless
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --db 35
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --force
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adaptive_compressor import AdaptiveCompressor

# Default SNR targets per mode (dB)
MODE_SNR = {'balanced': 30.0, 'lossy': 25.0, 'lossless': None}


def main():
    parser = argparse.ArgumentParser(description='Compress model weights to codebook format')
    parser.add_argument('model_path', help='Path to the model directory')
    parser.add_argument('--mode', default='balanced',
                        choices=['lossless', 'balanced', 'lossy'],
                        help='Compression mode (default: balanced)\n'
                             '  lossless : bit-perfect, MSE ≤ 1e-9\n'
                             '  balanced : SNR ≥ 30 dB (alias for --db 30)\n'
                             '  lossy    : SNR ≥ 25 dB (alias for --db 25)')
    parser.add_argument('--db', type=float, default=None,
                        help='SNR target in dB — overrides mode default.\n'
                             'Codebook saved as codebook-{N}dB/ (e.g. codebook-30dB/).\n'
                             'Typical range: 20 (aggressive) … 40 (near-lossless).')
    parser.add_argument('--bits', type=int, default=None,
                        choices=list(range(4, 14)),
                        help='Hard bit-width cap for lossy mode (4-13, default 8).')
    parser.add_argument('--entropy-code', action='store_true',
                        help='Replace LCM bit-packing with Huffman entropy coding.\n'
                             'Reduces index streams by ~18%% additional compression.\n'
                             'Load time is slower (CPU Huffman decode); inference speed\n'
                             'unchanged (expands to fixed-width before GPU upload).\n'
                             'Codebook stored in a separate subdirectory:\n'
                             '  codebook-lossless-huffman/, codebook-30dB-huffman/, etc.')
    parser.add_argument('--huffman-max-params', type=int, default=10_000_000,
                        help='Tensors larger than this many parameters fall back to\n'
                             'fixed-width LCM packing (default: 10M).  Increase to\n'
                             'Huffman-encode the embedding/lm_head at the cost of\n'
                             'slower model load.  Only relevant with --entropy-code.')
    parser.add_argument('--force', action='store_true',
                        help='Recompress even if cache already exists')
    parser.add_argument('--mse-threshold', type=float, default=None,
                        help='Max allowed MSE per weight (overrides mode default)')
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        print(f"Error: model path does not exist: {model_path}")
        sys.exit(1)

    st_files = list(model_path.glob('*.safetensors'))
    if not st_files:
        print(f"Error: no .safetensors files found in {model_path}")
        sys.exit(1)

    # Resolve effective SNR target
    snr_db = args.db if args.db is not None else MODE_SNR.get(args.mode)
    target_bits = args.bits

    # Determine where the cache will land so we can check for existing data
    _huff_suffix = '-huffman' if args.entropy_code else ''
    if args.mode == 'lossless':
        cache_tensors = model_path / f'codebook-lossless{_huff_suffix}' / 'tensors'
    elif snr_db is not None:
        cache_tensors = model_path / f'codebook-{int(snr_db)}dB{_huff_suffix}' / 'tensors'
    else:
        cache_tensors = model_path / f'codebook{_huff_suffix}' / 'tensors'

    already_done = cache_tensors.exists() and len(list(cache_tensors.glob('*.npz'))) > 0

    # Quality label for display
    if args.mode == 'lossless':
        quality_label = 'lossless (bit-perfect)'
    elif snr_db is not None:
        quality_label = f'{snr_db:.0f} dB SNR'
        if args.db is None:
            quality_label += f'  (--mode {args.mode} alias)'
    else:
        quality_label = args.mode
    if args.entropy_code:
        quality_label += '  + Huffman entropy coding'

    print(f"\n{'='*70}")
    print(f"COMPRESS MODEL")
    print(f"{'='*70}")
    print(f"Model  : {model_path}")
    print(f"Shards : {len(st_files)}")
    print(f"Quality: {quality_label}")
    if target_bits is not None:
        print(f"Bit cap: {target_bits} bits")
    if args.mse_threshold is not None:
        print(f"MSE    : {args.mse_threshold} (override)")
    print(f"Cache  : {cache_tensors}")
    if already_done and not args.force:
        print(f"\nCache already exists ({len(list(cache_tensors.glob('*.npz')))} tensors).")
        print("Use --force to recompress.")
        sys.exit(0)
    print(f"{'='*70}\n")

    # Show RAM / worker info
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
        workers = AdaptiveCompressor._safe_parallel_workers()
        print(f"RAM    : {avail_gb:.1f} GB available / {total_gb:.1f} GB total")
        print(f"Workers: {workers} (RAM-adaptive)\n")
    except ImportError:
        pass

    kwargs = dict(
        compression_mode=args.mode,
        store_in_model=True,
        force_rebuild=args.force,
        snr_db=snr_db,
        entropy_code=args.entropy_code,
        huffman_max_params=args.huffman_max_params,
    )
    if args.mse_threshold is not None:
        kwargs['mse_threshold'] = args.mse_threshold
    if target_bits is not None:
        kwargs['target_bits'] = target_bits

    t0 = time.time()
    compressor = AdaptiveCompressor(model_path, **kwargs)
    _, metadata = compressor.load_compressed(load_tensors=False)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Actual SNR summary (from metadata written during compression)
    if metadata and metadata.get('snr_target_db') is not None:
        t = metadata['snr_target_db']
        mn = metadata.get('snr_actual_min', '?')
        avg = metadata.get('snr_actual_mean', '?')
        mx = metadata.get('snr_actual_max', '?')
        n = metadata.get('snr_n_tensors', '?')
        print(f"SNR target : {t:.0f} dB")
        print(f"SNR actual : min={mn} dB  mean={avg} dB  max={mx} dB  ({n} tensors)")

    if cache_tensors.exists():
        npz_count = len(list(cache_tensors.glob('*.npz')))
        cache_size_gb = sum(f.stat().st_size for f in cache_tensors.glob('*.npz')) / 1e9
        orig_gb = sum(f.stat().st_size for f in model_path.glob('*.safetensors')) / 1e9
        print(f"Tensors : {npz_count}")
        print(f"Original: {orig_gb:.2f} GB")
        print(f"Cache   : {cache_size_gb:.2f} GB  (ratio {orig_gb/cache_size_gb:.2f}x)")
        print(f"Location: {cache_tensors}")
    print(f"{'='*70}\n")
    print("Next step: run inference with")
    print(f"  ./venv/bin/python proofofconcept/chat.py {model_path} --mode {args.mode}")


if __name__ == '__main__':
    main()
