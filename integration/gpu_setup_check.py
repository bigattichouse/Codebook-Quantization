#!/usr/bin/env python3
"""
GPU / ROCm Setup Diagnostic

Checks the GPU environment (NVIDIA CUDA or AMD ROCm), PyTorch integration,
and compression kernel compilation.  No model required.

Usage:
    python proofofconcept/integration/gpu_setup_check.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Allow imports from src/ regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def _section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def _run(cmd, label):
    print(f"\n  {label}:")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        out = (r.stdout or r.stderr).strip()[:300]
        if r.returncode == 0:
            print(f"  OK: {out}" if out else "  OK")
            return True, r.stdout.strip()
        else:
            print(f"  FAIL: {out}")
            return False, out
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (30s)")
        return False, "timeout"
    except Exception as e:
        print(f"  ERROR: {e}")
        return False, str(e)


def _find_rocm_root():
    """Find any ROCm installation, newest version first."""
    env = os.environ.get('ROCM_PATH', '')
    if env and Path(env).is_dir():
        return Path(env)
    candidates = sorted(Path('/opt').glob('rocm*'), reverse=True)
    return next((c for c in candidates if c.is_dir()), None)


# ── Individual checks ──────────────────────────────────────────────────────────

def check_environment():
    _section("Environment")

    rocm_root = _find_rocm_root()
    if rocm_root:
        print(f"  ROCm root: {rocm_root}")
    else:
        print("  ROCm: not found (OK for NVIDIA-only setups)")

    for var in ['ROCM_PATH', 'HIP_VISIBLE_DEVICES', 'HIP_PLATFORM',
                'CUDA_VISIBLE_DEVICES', 'ROCBLAS_LAYER']:
        val = os.environ.get(var)
        if val is not None:
            print(f"  {var}={val}")

    # Consider environment OK whether CUDA or ROCm
    return True


def check_tools():
    _section("GPU Tools")

    found = 0
    for tool in ['nvidia-smi', 'rocminfo', 'amd-smi', 'rocm-smi', 'hipcc']:
        ok, _ = _run(f"which {tool} 2>/dev/null", f"{tool}")
        if ok:
            found += 1

    # Try to pull GPU info from whichever tool is available
    _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || "
         "rocminfo 2>/dev/null | grep -E 'Name:|gfx' | head -6",
         "GPU hardware info")

    return found > 0


def check_pytorch():
    _section("PyTorch GPU Integration")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")

        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm:
            print(f"  Backend: ROCm/HIP {torch.version.hip}")
        else:
            cuda_ver = getattr(torch.version, 'cuda', None)
            print(f"  Backend: CUDA {cuda_ver or '(unknown)'}")

        avail = torch.cuda.is_available()
        print(f"  GPU available: {avail}")

        if avail:
            count = torch.cuda.device_count()
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1e9
                arch = getattr(props, 'gcnArchName', None)
                arch_str = f", arch={arch}" if arch else ""
                print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB{arch_str})")

        return avail
    except ImportError:
        print("  PyTorch not installed")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def check_kernel():
    _section("Compression Kernel Compilation")

    try:
        import torch
        if not torch.cuda.is_available():
            print("  Skipped — no GPU available")
            return True  # Not a failure; CPU path is valid

        from gpu_accelerated_functions import _load_extension
        print("  Compiling kernel (may take 30-60 s on first run)...")
        ext = _load_extension()

        if ext is not None:
            funcs = [a for a in dir(ext) if not a.startswith('_')]
            print(f"  Kernel ready. Functions: {funcs}")
            return True
        else:
            print("  Kernel compilation failed (see messages above)")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("GPU Setup Diagnostic  (NVIDIA CUDA / AMD ROCm / CPU)")

    results = {
        'environment': check_environment(),
        'tools':       check_tools(),
        'pytorch':     check_pytorch(),
        'kernel':      check_kernel(),
    }

    _section("Summary")
    all_ok = True
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.capitalize()}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll checks passed — GPU acceleration is ready.")
    else:
        print("\nSome checks failed.  See output above for details.")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
