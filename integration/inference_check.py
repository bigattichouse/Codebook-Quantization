#!/usr/bin/env python3
"""
Compressed Inference Diagnostic

Tests the full compressed-inference pipeline end-to-end on a model you provide.
The model must already have a compression cache (run compress.py first).

Works on CPU, NVIDIA CUDA, and AMD ROCm — no hardware-specific assumptions.

Usage:
    python proofofconcept/integration/inference_check.py /path/to/model
    python proofofconcept/integration/inference_check.py /path/to/model --device cpu
    python proofofconcept/integration/inference_check.py /path/to/model --device cuda
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Allow imports from src/ regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def _section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Individual checks ──────────────────────────────────────────────────────────

def check_imports():
    """Verify all required packages and project modules are importable."""
    _section("Import Check")

    ok = True
    for name in ['torch', 'numpy', 'transformers']:
        try:
            __import__(name)
            print(f"  OK: {name}")
        except ImportError as e:
            print(f"  FAIL: {name} — {e}")
            ok = False

    for name in ['adaptive_compressor', 'model_loader',
                 'compressed_modules', 'gpu_accelerated_functions']:
        try:
            __import__(name)
            print(f"  OK: {name} (src)")
        except ImportError as e:
            print(f"  FAIL: {name} — {e}")
            ok = False

    return ok


def check_gpu(device: str):
    """Report GPU availability; always passes (CPU fallback is valid)."""
    _section("GPU Check")
    try:
        import torch
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        backend = f"ROCm/HIP {torch.version.hip}" if is_rocm else \
                  f"CUDA {getattr(torch.version, 'cuda', '?')}"
        avail = torch.cuda.is_available()
        print(f"  Backend: {backend if avail else 'CPU only'}")
        if avail:
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        if device == 'cuda' and not avail:
            print("  WARNING: --device cuda requested but no GPU found; "
                  "inference will run on CPU")
    except Exception as e:
        print(f"  Error: {e}")
    return True  # GPU absence is not a hard failure


def check_compression_cache(model_path: Path):
    """Verify that the model has a compression cache."""
    _section("Compression Cache")

    cache = model_path / 'codebook' / 'tensors'
    npz_files = list(cache.glob('*.npz')) if cache.exists() else []

    if npz_files:
        print(f"  Cache: {cache}")
        print(f"  Compressed tensors: {len(npz_files)}")
        return True
    else:
        print(f"  No cache found at: {cache}")
        print(f"  Run first:  python proofofconcept/compress.py {model_path}")
        return False


def check_inference(model_path: Path, device: str):
    """Run a short end-to-end generation via chat.py."""
    _section("End-to-End Inference")

    chat_script = Path(__file__).parent.parent / 'chat.py'
    if not chat_script.exists():
        print(f"  chat.py not found at: {chat_script}")
        return False

    cmd = [
        sys.executable, str(chat_script),
        str(model_path),
        '--device', device,
        '--prompt', 'Reply with only the word "hello".',
        '--max-tokens', '20',
    ]
    print(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - t0

        if result.returncode == 0:
            out = result.stdout.strip()
            print(f"  OK ({elapsed:.1f}s)")
            # Show last 300 chars of output to avoid flooding the terminal
            print(f"  Output: ...{out[-300:]}" if len(out) > 300 else f"  Output: {out}")
            return True
        else:
            print(f"  FAIL (exit code {result.returncode}, {elapsed:.1f}s)")
            stderr = result.stderr.strip()
            print(f"  Stderr: {stderr[-400:] if len(stderr) > 400 else stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (5 minutes)")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compressed inference diagnostic — works on CPU, CUDA, and ROCm'
    )
    parser.add_argument('model_path', type=Path,
                        help='Path to the model directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: model path not found: {args.model_path}")
        return 1

    print(f"Compressed Inference Diagnostic")
    print(f"  Model:  {args.model_path.resolve()}")
    print(f"  Device: {args.device}")

    imports_ok = check_imports()
    check_gpu(args.device)
    cache_ok = check_compression_cache(args.model_path)

    inference_ok = False
    if cache_ok:
        inference_ok = check_inference(args.model_path, args.device)
    else:
        print("\n  Skipping inference test — no compression cache.")

    _section("Summary")
    results = {
        'imports':   imports_ok,
        'cache':     cache_ok,
        'inference': inference_ok,
    }
    all_ok = True
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.capitalize()}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed.  See output above for details.")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
