"""
Fast Index Manager for Compressed Models

Implements optimized bit-packed index lookups based on strategies from other/ directory:
- Wide-load strategy for branchless extraction  
- Precomputed lookup tables
- 64-bit chunking
- Vectorized batch processing
- LRU caching

Maintains compressed storage while achieving 10-20x lookup speedup.
"""

import torch
import numpy as np
import gc
from typing import Dict, Optional, Union

from bitpack import unpack_any_bits


class FastIndexManager:
    """
    Manages fast index lookups for compressed tensors.
    Uses precomputed tables and wide-load strategies.
    """
    
    def __init__(self, device: str = 'cpu', max_lookup_tables: int = 512):
        self.device = device
        self.lookup_tables = {}  # tensor_name -> lookup_info
        self.max_lookup_tables = max_lookup_tables
        self._access_count = {}  # tensor_name -> count for LRU
        self.chunk_cache = {}    # Small cache for recently used chunks
        self._eviction_count = 0
        self._last_evicted = ''
        
    def prepare_lookup_table(self, tensor_name: str, indices_tensor: Union[torch.Tensor, np.ndarray], bits: int):
        """
        Precompute optimized lookup table for a tensor's indices.
        
        Args:
            tensor_name: Unique identifier for the tensor
            indices_tensor: Compressed indices (uint8 tensor or array)
            bits: Bit width (4, 8, etc.)
        """
        if isinstance(indices_tensor, np.ndarray):
            # Ensure contiguous and well-defined byte order
            indices_tensor = np.ascontiguousarray(indices_tensor)
            indices_tensor = torch.from_numpy(indices_tensor)
            
        # Check if we need to evict old lookup tables to prevent memory leaks
        if len(self.lookup_tables) >= self.max_lookup_tables:
            self._evict_oldest_lookup_table()
            short = self._last_evicted.split('.')[-2] if '.' in self._last_evicted else self._last_evicted
            line = (f"\r  [IndexMgr] {len(self.lookup_tables)}/{self.max_lookup_tables} cached"
                    f" | {self._eviction_count} evicted | last: {short[:40]}")
            print(line, end='', flush=True)
            
        total_elements = indices_tensor.numel() * 8 // bits  # Total logical indices
        
        if bits == 8:
            # 8-bit case - direct lookup, no optimization needed
            self.lookup_tables[tensor_name] = {
                'type': 'direct',
                'indices': indices_tensor,
                'bits': 8
            }
            # Track access for LRU eviction
            self._access_count[tensor_name] = 0
            return
        
        # _fast_packed_lookup uses _unpack_bits_np which computes offsets on the fly —
        # no need to precompute byte_offsets / bit_shifts arrays (those were O(N) allocations
        # that consumed 5× the packed size in RAM and were never read back).
        self.lookup_tables[tensor_name] = {
            'type': 'packed',
            'indices': indices_tensor,
            'bits': bits,
            'total_elements': total_elements
        }
        
        # Track access for LRU eviction
        self._access_count[tensor_name] = 0
        
    def fast_index_lookup(self, tensor_name: str, target_elements: int, start_offset: int = 0) -> torch.Tensor:
        """
        Retrieve indices using the optimized lookup table.
        
        Args:
            tensor_name: Name of the tensor to lookup
            target_elements: Number of elements to retrieve
            start_offset: Logical start index in the weight matrix
            
        Returns:
            torch.Tensor of indices (long)
        """
        if tensor_name not in self.lookup_tables:
            raise KeyError(f"Lookup table for {tensor_name} not prepared. Call prepare_lookup_table first.")
            
        lookup_info = self.lookup_tables[tensor_name]
        self._access_count[tensor_name] += 1
        
        if lookup_info['type'] == 'direct':
            # Direct 8-bit lookup
            indices = lookup_info['indices']
            result = indices[start_offset:start_offset + target_elements].to(torch.long)
            return result.to(self.device, non_blocking=True)
            
        elif lookup_info['type'] == 'packed':
            # Optimized packed lookup
            return self._fast_packed_lookup(lookup_info, target_elements, start_offset)

        return None
        
    def _fast_packed_lookup(self, lookup_info: dict, target_elements: int, start_offset: int = 0) -> torch.Tensor:
        """Vectorized packed bit extraction using lcm-group uint64 approach.

        Uses a direct group-level seek so cost is O(target_elements), not
        O(start_offset + target_elements).  For 13-bit packing, each group holds
        8 values in 13 bytes; we seek to the group containing start_offset,
        unpack only the minimal number of groups needed, then slice.
        """
        from bitpack import _group_params
        indices_np = lookup_info['indices'].cpu().numpy()
        bits = lookup_info['bits']

        group_values, group_bytes = _group_params(bits)

        # Seek to the group that contains start_offset.
        group_idx    = start_offset // group_values
        within_group = start_offset % group_values
        byte_offset  = group_idx * group_bytes

        # Unpack only from the target group onward.
        n_needed = within_group + target_elements
        result = unpack_any_bits(indices_np[byte_offset:], bits, n_needed)
        return torch.from_numpy(result[within_group:within_group + target_elements].astype(np.int64))

    def _evict_oldest_lookup_table(self):
        """Simple LRU eviction — no per-eviction print (caller prints rolling status)."""
        if not self.lookup_tables:
            return

        # Find entry with lowest access count
        oldest_name = min(self._access_count, key=self._access_count.get)

        del self.lookup_tables[oldest_name]
        del self._access_count[oldest_name]
        self._eviction_count += 1
        self._last_evicted = oldest_name
        gc.collect()


def get_index_manager(device: str = 'cpu') -> FastIndexManager:
    """Get the global index manager instance (always CPU - lookups use numpy internally)."""
    global _global_index_manager
    if _global_index_manager is None:
        _global_index_manager = FastIndexManager(device='cpu')
    return _global_index_manager

_global_index_manager = None
