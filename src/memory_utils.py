"""
memory_utils.py — System memory / VRAM accounting helpers.

Used by model_loader and chat to report before/after usage.
"""

from typing import Dict
import torch


def resolve_device(requested: str = 'cuda') -> str:
    """Return the effective PyTorch device string.

    Falls back to 'cpu' when CUDA/ROCm isn't visible to PyTorch.
    GPU compute via direct HIP ctypes kernels (AdaptiveCodebookLinear etc.)
    still runs on-device even when PyTorch tensors live in CPU memory.
    """
    if requested == 'cpu':
        return 'cpu'
    return requested if torch.cuda.is_available() else 'cpu'


def synchronize() -> None:
    """Synchronize CUDA/ROCm device if available; no-op otherwise."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_memory_stats() -> Dict[str, float]:
    """Return current CPU RSS and GPU allocated memory in GB."""
    stats: Dict[str, float] = {}
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    stats['cpu_rss_gb'] = float(line.split()[1]) / 1e6
                elif line.startswith('VmSize:'):
                    stats['cpu_virtual_gb'] = float(line.split()[1]) / 1e6
    except OSError:
        pass
    if torch.cuda.is_available():
        stats['gpu_mem_gb'] = torch.cuda.memory_allocated() / 1e9
    return stats


def format_memory_stats(stats: Dict[str, float]) -> str:
    lines = []
    if 'cpu_rss_gb' in stats:
        lines.append(f"  CPU RAM (RSS):   {stats['cpu_rss_gb']:.2f} GB")
    if 'gpu_mem_gb' in stats:
        lines.append(f"  GPU RAM:         {stats['gpu_mem_gb']:.2f} GB")
    return '\n'.join(lines)
