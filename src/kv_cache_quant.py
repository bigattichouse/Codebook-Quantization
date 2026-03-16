"""
kv_cache_quant.py — INT8 per-token-per-head KV cache quantization.

Plugs into the transformers 5.x Cache API (Cache / CacheLayerMixin).
Replaces DynamicCache with a drop-in that stores K and V as INT8 + a
float32 scale per head per token, then returns dequantized bf16 tensors
to the attention computation.

Memory layout per layer:
    keys_q  : int8   [B, H, S, D]   — 1 byte/element
    keys_s  : f32    [B, H, S, 1]   — 4 bytes/head/token (scale)
    values_q: int8   [B, H, S, D]
    values_s: f32    [B, H, S, 1]

Effective compression vs bf16:
    bf16 element = 2 bytes
    int8 element = 1 byte + 4/D bytes (scale overhead)
    For D=128: 1 + 0.03 ≈ 1.03 bytes → ~1.94× compression

Usage:
    from kv_cache_quant import INT8KVCache
    outputs = model.generate(..., past_key_values=INT8KVCache())

Enable in chat.py via --kv-quant flag.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


# ---------------------------------------------------------------------------
# Quantize / dequantize helpers
# ---------------------------------------------------------------------------

def _quantize(x: torch.Tensor):
    """Per-token-per-head symmetric INT8.  x: [B, H, S, D] → (int8, f32 scale)."""
    # scale shape [B, H, S, 1] — one scale per (batch, head, token)
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-8) / 127.0
    q = (x.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def _dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.float() * scale).to(dtype)


# ---------------------------------------------------------------------------
# Per-layer cache that stores INT8
# ---------------------------------------------------------------------------

class INT8Layer(CacheLayerMixin):
    """Single-layer KV cache backed by INT8 storage."""

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype  = key_states.dtype
        self.device = key_states.device
        self.keys_q  = torch.empty(0, dtype=torch.int8,   device=self.device)
        self.keys_s  = torch.empty(0, dtype=torch.float32, device=self.device)
        self.vals_q  = torch.empty(0, dtype=torch.int8,   device=self.device)
        self.vals_s  = torch.empty(0, dtype=torch.float32, device=self.device)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        k_q, k_s = _quantize(key_states)
        v_q, v_s = _quantize(value_states)

        if self.keys_q.numel() == 0:
            self.keys_q = k_q
            self.keys_s = k_s
            self.vals_q = v_q
            self.vals_s = v_s
        else:
            self.keys_q = torch.cat([self.keys_q, k_q], dim=-2)
            self.keys_s = torch.cat([self.keys_s, k_s], dim=-2)
            self.vals_q = torch.cat([self.vals_q, v_q], dim=-2)
            self.vals_s = torch.cat([self.vals_s, v_s], dim=-2)

        k_out = _dequantize(self.keys_q, self.keys_s, self.dtype)
        v_out = _dequantize(self.vals_q, self.vals_s, self.dtype)
        return k_out, v_out

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.keys_q.numel() == 0:
            return 0
        return self.keys_q.shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor):
        """Return (kv_length, kv_offset) for mask generation."""
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, 0

    def get_max_cache_shape(self) -> int:
        return -1  # unlimited

    def vram_bytes(self) -> int:
        b = 0
        for t in (self.keys_q, self.vals_q):
            if t.numel():
                b += t.numel()          # int8: 1 byte/element
        for t in (self.keys_s, self.vals_s):
            if t.numel():
                b += t.numel() * 4      # float32: 4 bytes/element
        return b

    def bf16_bytes(self) -> int:
        if not self.is_initialized or self.keys_q.numel() == 0:
            return 0
        return (self.keys_q.numel() + self.vals_q.numel()) * 2  # bf16: 2 bytes


# ---------------------------------------------------------------------------
# Full cache container
# ---------------------------------------------------------------------------

class INT8KVCache(Cache):
    """
    Drop-in replacement for DynamicCache that stores K/V as INT8.

    Pass to model.generate() via:
        model.generate(..., past_key_values=INT8KVCache())
    """

    def __init__(self):
        super().__init__(layer_class_to_replicate=INT8Layer)

    # --- stats helpers -------------------------------------------------------

    def vram_mb(self) -> float:
        return sum(
            layer.vram_bytes() for layer in self.layers if layer.is_initialized
        ) / 1e6

    def bf16_equivalent_mb(self) -> float:
        return sum(
            layer.bf16_bytes() for layer in self.layers if layer.is_initialized
        ) / 1e6

    def stats(self) -> dict:
        seq = self.get_seq_length()
        q_mb = self.vram_mb()
        b_mb = self.bf16_equivalent_mb()
        return {
            "seq_len":       seq,
            "n_layers":      len(self.layers),
            "quant_mb":      round(q_mb, 1),
            "bf16_equiv_mb": round(b_mb, 1),
            "ratio":         round(b_mb / q_mb, 2) if q_mb else 0.0,
        }

    def print_stats(self):
        s = self.stats()
        print(f"  KV cache (INT8): {s['seq_len']} tokens × {s['n_layers']} layers  "
              f"→ {s['quant_mb']:.0f} MB  (bf16 would be {s['bf16_equiv_mb']:.0f} MB, "
              f"{s['ratio']:.2f}× compression)")
