#!/usr/bin/env python3
"""
compress.py — Compress a model to codebook format.

CPU-only: reads safetensors, builds codebooks, saves compressed tensors.
Does NOT load model architecture or touch the GPU.
Run this before chat.py.

Usage:
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --mode lossless
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --mode aggressive
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --force
    ./venv/bin/python proofofconcept/compress.py ~/workspace/model/Qwen3.5-9B --mse-threshold 0.001
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from adaptive_compressor import AdaptiveCompressor


def main():
    parser = argparse.ArgumentParser(description='Compress model weights to codebook format')
    parser.add_argument('model_path', help='Path to the model directory')
    parser.add_argument('--mode', default='balanced',
                        choices=['lossless', 'balanced', 'lossy'],
                        help='Compression mode (default: balanced)\n'
                             '  lossless : exact values, ~13-bit, cos > 0.999\n'
                             '  balanced : adaptive 8-12 bit k-means, steps up if needed\n'
                             '  lossy    : hard bit cap via --bits N (default 8)')
    parser.add_argument('--bits', type=int, default=None,
                        choices=list(range(4, 14)),
                        help='Hard bit-width cap for lossy mode (4-13, default 8).\n'
                             'Balanced will use up to this many bits; lossy uses exactly this.')
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

    internal_mode = args.mode
    target_bits = args.bits

    cache_tensors = model_path / 'codebook' / 'tensors'
    already_done = cache_tensors.exists() and len(list(cache_tensors.glob('*.npz'))) > 0

    print(f"\n{'='*70}")
    print(f"COMPRESS MODEL")
    print(f"{'='*70}")
    print(f"Model : {model_path}")
    print(f"Shards: {len(st_files)}")
    print(f"Mode  : {args.mode}", end="")
    if target_bits is not None:
        print(f"  (--bits {target_bits})")
    else:
        print()
    if args.mse_threshold is not None:
        print(f"MSE   : {args.mse_threshold} (override)")
    print(f"Cache : {cache_tensors}")
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
        compression_mode=internal_mode,
        store_in_model=True,
        force_rebuild=args.force,
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
    print(f"  ./venv/bin/python proofofconcept/chat.py {model_path}")


if __name__ == '__main__':
    main()
