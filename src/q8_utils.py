"""
Q8 Quantization Utilities

Per-tensor Q8 quantization with scale and offset.
"""

import numpy as np
from typing import Tuple


def quantize_q8(weights: np.ndarray) -> Tuple[np.ndarray, np.float32, np.float32]:
    """
    Q8 quantization with per-tensor scale and offset.
    
    Quantizes weights to uint8 [0, 255] range with linear scaling.
    
    Args:
        weights: Input weights (any float type)
    
    Returns:
        q: Quantized weights as uint8 [0, 255]
        scale: Scale factor (range / 255)
        offset: Minimum value (zero point)
    
    Formula:
        q = round((weights - offset) / scale).clip(0, 255)
        weights ≈ q * scale + offset
    """
    w_min = np.float32(weights.min())
    w_max = np.float32(weights.max())
    
    # Handle edge case: all values same
    if w_max - w_min < 1e-10:
        scale = np.float32(1e-10)
    else:
        scale = (w_max - w_min) / 255.0
    
    # Quantize to [0, 255]
    q = np.round((weights.astype(np.float32) - w_min) / scale)
    q = q.clip(0, 255).astype(np.uint8)
    
    return q, scale, w_min


def dequantize_q8(q: np.ndarray, scale: np.float32, offset: np.float32) -> np.ndarray:
    """
    Dequantize Q8 back to float32.
    
    Args:
        q: Quantized weights (uint8 [0, 255])
        scale: Scale factor from quantization
        offset: Offset (zero point) from quantization
    
    Returns:
        Reconstructed weights as float32
    
    Formula:
        weights = q * scale + offset
    """
    return q.astype(np.float32) * scale + offset


def quantize_q8_error(weights: np.ndarray) -> Tuple[np.ndarray, np.float32, np.float32, np.float32]:
    """
    Q8 quantization with error measurement.
    
    Returns:
        q: Quantized weights
        scale: Scale factor
        offset: Offset
        error: Mean absolute error after dequantization
    """
    q, scale, offset = quantize_q8(weights)
    reconstructed = dequantize_q8(q, scale, offset)
    error = np.abs(weights.astype(np.float32) - reconstructed).mean()
    return q, scale, offset, error


def count_unique_q8(weights: np.ndarray) -> int:
    """
    Count unique values after Q8 quantization.
    
    This tells us how many of the 256 possible Q8 levels are actually used.
    If ≤128, we can use 7-bit packing. If ≤64, we can use 6-bit packing.
    
    Args:
        weights: Input weights
    
    Returns:
        Number of unique Q8 values (1-256)
    """
    q, _, _ = quantize_q8(weights)
    return len(np.unique(q))


def q8_histogram(weights: np.ndarray) -> np.ndarray:
    """
    Get histogram of Q8 quantized values.
    
    Shows which of the 256 levels are most/least used.
    
    Args:
        weights: Input weights
    
    Returns:
        Histogram array of size 256
    """
    q, _, _ = quantize_q8(weights)
    hist, _ = np.histogram(q, bins=256, range=(0, 256))
    return hist


def analyze_q8_distribution(weights: np.ndarray) -> dict:
    """
    Analyze Q8 quantization characteristics.
    
    Args:
        weights: Input weights
    
    Returns:
        Dictionary with analysis results
    """
    q, scale, offset = quantize_q8(weights)
    unique_q8 = len(np.unique(q))
    hist = q8_histogram(weights)
    
    # Find most/least used bins
    non_zero_bins = np.where(hist > 0)[0]
    most_used_bin = non_zero_bins[hist[non_zero_bins].argmax()] if len(non_zero_bins) > 0 else -1
    
    # Calculate sparsity (how many of 256 bins are unused)
    sparsity = 1.0 - (unique_q8 / 256.0)
    
    return {
        'unique_q8': unique_q8,
        'scale': scale,
        'offset': offset,
        'sparsity': sparsity,
        'most_used_bin': int(most_used_bin),
        'min_bin': int(non_zero_bins.min()) if len(non_zero_bins) > 0 else -1,
        'max_bin': int(non_zero_bins.max()) if len(non_zero_bins) > 0 else -1,
        'can_pack_7bit': unique_q8 <= 128,
        'can_pack_6bit': unique_q8 <= 64,
        'can_pack_4bit': unique_q8 <= 16,
    }
