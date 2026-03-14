"""
Tensor Analysis Module

Analyzes tensors to determine optimal compression strategy.
Full scan by default for exact decisions.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from q8_utils import quantize_q8, count_unique_q8, analyze_q8_distribution
from bitpack import calculate_packed_size


@dataclass
class TensorAnalysis:
    """Results of tensor analysis."""
    strategy: str  # 'direct_codebook', 'q8_packed', 'q8_codebook', 'exact', 'adaptive'
    codebook_size: int
    index_bits: int
    unique_bf16: int
    unique_q8: int
    tensor_size: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    sampled: bool = False
    reconstruction_error: float = 0.0
    is_lossless: bool = False


def analyze_tensor(
    weights: np.ndarray,
    tensor_name: str,
    sample_size: Optional[int] = None,
    max_codebook_size: int = 256
) -> TensorAnalysis:
    """
    Analyze a tensor and recommend compression strategy.
    """
    # Use full tensor or sample
    if sample_size is not None and weights.size > sample_size:
        indices = np.random.choice(weights.size, sample_size, replace=False)
        flat = weights.flatten()[indices]
        sampled = True
    else:
        flat = weights.flatten()
        sampled = False
    
    # Count EXACT unique values
    unique_bf16 = len(np.unique(flat))
    
    # Apply Q8 and count EXACT unique Q8 values
    unique_q8 = count_unique_q8(flat)
    
    # Classify tensor type - CRITICAL FIX FOR MoE ROUTERS
    name_low = tensor_name.lower()
    
    is_layernorm = 'norm' in name_low or 'ln' in name_low
    
    # Router detection: 
    # Qwen: model.layers.N.mlp.gate.weight (NO 'experts' in path)
    # Others: block_sparse_moe.gate
    is_router = 'router' in name_low or ('gate' in name_low and 'experts' not in name_low and 'gate_proj' not in name_low)
    
    is_ssm = 'a_log' in name_low or 'dt_bias' in name_low or 'conv1d' in name_low
    is_embedding = 'embed' in name_low or 'lm_head' in name_low
    is_small = weights.size < 500000  # < 1MB (approx 500k bf16 elements)
    
    # Determine strategy
    if is_layernorm or is_router or is_ssm or is_embedding or is_small:
        # Keep exact for critical layers or tiny tensors
        strategy = 'exact'
        codebook_size = 0
        index_bits = 16  # bfloat16
        compressed_bytes = weights.size * 2
    
    elif unique_bf16 <= 256:
        # Few unique values - direct codebook
        strategy = 'direct_codebook'
        codebook_size = next_power_of_2(unique_bf16)
        codebook_size = min(codebook_size, max_codebook_size)
        index_bits = 8 if codebook_size <= 256 else 16
        compressed_bytes = weights.size * (index_bits // 8) + codebook_size * 4
    
    elif unique_q8 <= 128:
        # Q8 has ≤128 unique - use 7-bit packing
        strategy = 'q8_packed_7bit'
        codebook_size = unique_q8  # Exact count
        index_bits = 7
        compressed_bytes = calculate_packed_size(weights.size, 7) + unique_q8 * 4
    
    elif unique_q8 <= 256:
        # Q8 has ≤256 unique - use uint8 indices
        strategy = 'q8_codebook'
        codebook_size = unique_q8  # Exact count
        codebook_size = min(codebook_size, max_codebook_size)
        index_bits = 8
        compressed_bytes = weights.size + codebook_size * 4
    
    else:
        # Many unique values - use larger codebook
        strategy = 'q8_codebook'
        codebook_size = max_codebook_size
        index_bits = 8 if codebook_size <= 256 else 16
        compressed_bytes = weights.size * (index_bits // 8) + codebook_size * 4
    
    # Calculate original size (bfloat16)
    original_bytes = weights.size * 2
    
    # Calculate compression ratio
    compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    
    return TensorAnalysis(
        strategy=strategy,
        codebook_size=codebook_size,
        index_bits=index_bits,
        unique_bf16=unique_bf16,
        unique_q8=unique_q8,
        tensor_size=weights.size,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=compression_ratio,
        sampled=sampled
    )


def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 ≥ n."""
    if n <= 0: return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1
