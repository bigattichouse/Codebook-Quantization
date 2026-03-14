#!/usr/bin/env python3
"""
Memory profiling script for compressed model loading.

Measures RAM and VRAM at each phase of model loading, then reports
final vs. theoretical uncompressed usage.

Usage:
    ./venv/bin/python proofofconcept/tests/memory_profile_load.py ~/workspace/model/Qwen3.5-0.8B
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import psutil
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def get_peak_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def estimate_uncompressed_gb(model_path: Path) -> float:
    """Estimate uncompressed model size from safetensors files."""
    total = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
    return total / 1e9


def main():
    parser = argparse.ArgumentParser(description="Memory profile compressed model load")
    parser.add_argument("model_path", type=Path, help="Path to model directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--mode", default="balanced", help="Compression mode")
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    if not model_path.exists():
        print(f"Error: {model_path} does not exist")
        sys.exit(1)

    uncompressed_gb = estimate_uncompressed_gb(model_path)

    print(f"{'=' * 70}")
    print(f"MEMORY PROFILE: {model_path.name}")
    print(f"{'=' * 70}")
    print(f"Theoretical uncompressed size: {uncompressed_gb:.2f} GB")
    print()

    # Phase 0: Baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rss_baseline = get_rss_gb()
    vram_baseline = get_vram_gb()
    print(f"Phase 0 — Baseline:")
    print(f"  RAM (RSS):  {rss_baseline:.3f} GB")
    print(f"  VRAM:       {vram_baseline:.3f} GB")
    print()

    # Phase 1: Import and create compressor
    from adaptive_compressor import AdaptiveCompressor
    mse_target = 0.0025  # balanced default
    compressor = AdaptiveCompressor(
        model_path,
        compression_mode=args.mode,
        store_in_model=True,
        mse_threshold=mse_target,
    )
    rss_import = get_rss_gb()
    print(f"Phase 1 — Compressor created:")
    print(f"  RAM (RSS):  {rss_import:.3f} GB  (+{rss_import - rss_baseline:.3f})")
    print()

    # Phase 2: Load compressed (metadata + codebooks only)
    t0 = time.time()
    _, metadata = compressor.load_compressed(load_tensors=False)
    t_load = time.time() - t0
    gc.collect()
    rss_loaded = get_rss_gb()
    vram_loaded = get_vram_gb()
    print(f"Phase 2 — Metadata loaded ({t_load:.1f}s):")
    print(f"  RAM (RSS):  {rss_loaded:.3f} GB  (+{rss_loaded - rss_baseline:.3f})")
    print(f"  VRAM:       {vram_loaded:.3f} GB")
    print(f"  Tensors:    {metadata.get('tensor_count', '?')}")
    print()

    # Phase 3: Load model on device
    from chat import CompressedChatModel
    chat_model = CompressedChatModel(
        model_path,
        device=args.device,
        compression_mode=args.mode,
    )
    t0 = time.time()
    chat_model.load()
    t_full = time.time() - t0
    gc.collect()
    rss_final = get_rss_gb()
    vram_final = get_vram_gb()
    vram_peak = get_peak_vram_gb()

    print()
    print(f"Phase 3 — Model ready ({t_full:.1f}s):")
    print(f"  RAM (RSS):  {rss_final:.3f} GB  (+{rss_final - rss_baseline:.3f})")
    print(f"  VRAM:       {vram_final:.3f} GB")
    print(f"  VRAM peak:  {vram_peak:.3f} GB")
    print()

    # Summary
    print(f"{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Compressed model loaded: {vram_final:.2f} GB VRAM "
          f"(vs {uncompressed_gb:.2f} GB theoretical uncompressed)")
    if uncompressed_gb > 0:
        ratio = uncompressed_gb / max(vram_final, 0.001)
        print(f"Effective compression ratio: {ratio:.1f}x")
    print(f"Total load time: {t_full:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
