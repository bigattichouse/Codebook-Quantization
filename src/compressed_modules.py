"""
Compressed Model Modules

Layers that use codebook lookup during forward pass.
Weights stay compressed in memory - never decompressed permanently.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

# Try to import GPU accelerated versions
try:
    from gpu_accelerated_functions import GPUAcceleratedLinear, GPUAcceleratedEmbedding
    GPU_ACCELERATED_AVAILABLE = True
    HIP_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_ACCELERATED_AVAILABLE = False
    HIP_AVAILABLE = False

# Huffman + codebook kernel (separate file, never modifies the above)
try:
    from gpu_huffman_functions import HuffmanCodebookLinear, HuffmanCodebookEmbedding
    HUFFMAN_AVAILABLE = True
except ImportError:
    HUFFMAN_AVAILABLE = False

from fast_index_manager import get_index_manager

try:
    from compressed_matmul_cpu import (
        compressed_matmul      as _c_matmul,
        huffman_matmul         as _huff_matmul,
        huffman_embedding      as _huff_embedding,
        huffman_decode_weights as _huff_decode_weights,
        C_KERNEL_AVAILABLE,
    )
except ImportError:
    _c_matmul             = None
    _huff_matmul          = None
    _huff_embedding       = None
    _huff_decode_weights  = None
    C_KERNEL_AVAILABLE    = False

class AdaptiveCodebookLinear(nn.Module):
    """
    Linear layer that supports multi-tier compressed formats.
    """
    
    def __init__(self, name: str, shape: Tuple[int, ...], mode: str = 'exact'):
        super().__init__()
        self.name = name
        # Ensure shape is a tuple of ints
        self.shape = tuple(int(s) for s in shape)
        self.mode = mode
        self.register_buffer('bias', None, persistent=False)
        
        # Persistent buffers
        self.register_buffer('indices', None, persistent=False)
        self.register_buffer('codebook', None, persistent=False)
        self.register_buffer('weight', None, persistent=False)
        self.register_buffer('packed', None, persistent=False)
        self.register_buffer('unique_q8', None, persistent=False)
        self.register_buffer('scale', None, persistent=False)
        self.register_buffer('v_min', None, persistent=False)
        
        self.original_len = 0
        self.bits = 8
        self._gpu_func = None
        self._cached_weight = None
        self._mmap_buf = None   # MmappedPackedBuffer when use_mmap=True
        self._huff_data = None  # dict of numpy arrays for inference-time Huffman decode

    def forward(self, x):
        if self.mode == 'exact':
            w = self.weight
            if w.dtype != x.dtype:
                w = w.to(x.dtype)
            return F.linear(x, w, self.bias)
            
        if self.mode == 'direct_codebook' and self._gpu_func is not None:
            # Use GPU-accelerated version. Ensure output lands on the same device as
            # the input — on GPU→CPU fallback the model runs on CPU but _gpu_func
            # returns CUDA tensors, which corrupts subsequent CPU operations.
            return self._gpu_func(x).to(x.device)
            
        if self.mode == 'direct_codebook' and self._huff_data is not None:
            # Huffman inference-time path: Huffman bitstream stays compressed
            # in CPU RAM at all times — no packed index buffer ever materialised.
            # If x is on GPU, decode to float32 weights on CPU then GPU-matmul.
            # If x is on CPU, do the full matmul on CPU via C kernel.
            M, K = self.shape
            cb_cpu = self.codebook.cpu().float()
            orig_device = x.device
            orig_shape  = x.shape

            if orig_device.type == 'cuda' and _huff_decode_weights is not None:
                # Decode Huffman → float32 weight matrix [M, K] on CPU,
                # upload transiently to GPU, GPU matmul — VRAM holds only the
                # transient weight buffer (~M*K*4 bytes) rather than the full
                # packed index store.
                weights_np = _huff_decode_weights(
                    self._huff_data, cb_cpu.numpy(), M, K, C=len(cb_cpu))
                weights_t = torch.from_numpy(weights_np).to(
                    orig_device, non_blocking=True)
                out = torch.matmul(
                    x.reshape(-1, K).float(), weights_t.T)
                del weights_t  # free transient VRAM immediately
                if self.bias is not None:
                    out = out + self.bias.to(orig_device).float()
                out = out.reshape(*orig_shape[:-1], M)
                return out.to(x.dtype)
            else:
                # CPU-only path (device=cpu or C kernel unavailable)
                x_np   = x.cpu().reshape(-1, K).float().numpy()
                out_np = _huff_matmul(x_np, self._huff_data, cb_cpu.numpy(),
                                      M, K, C=len(cb_cpu))
                out = torch.from_numpy(out_np).reshape(*orig_shape[:-1], M)
                if self.bias is not None:
                    out = out + self.bias.cpu().float()
                if x.dtype != torch.float32:
                    out = out.to(x.dtype)
                return out.to(orig_device, non_blocking=True)

        if self.mode == 'direct_codebook':
            # True compressed matmul — no weight matrix ever materialised.
            # The C kernel (or numpy fallback) reads packed bits + codebook
            # directly; `w` is a scalar register in the inner loop.
            M, K = self.shape
            cb_cpu = self.codebook.cpu().float()  # float32; codebook may be bf16 after model.to()
            if self._mmap_buf is not None:
                packed_np = self._mmap_buf.as_numpy()
            else:
                packed_np = self.indices.cpu().numpy()

            x_cpu = x.cpu()
            orig_shape = x_cpu.shape
            x_np = x_cpu.reshape(-1, K).float().numpy()

            out_np = _c_matmul(x_np, packed_np, cb_cpu.numpy(), M, K, self.bits,
                               C=len(cb_cpu))

            out = torch.from_numpy(out_np).reshape(*orig_shape[:-1], M)
            if self.bias is not None:
                out = out + self.bias.cpu().float()
            if x.dtype != torch.float32:
                out = out.to(x.dtype)
            return out.to(x.device, non_blocking=True)
            
        elif self.mode == 'linear_quant':
            # Dequantize: indices are raw integer levels; reconstruct in float32
            target_shape = self.shape
            idx = self.indices.float()
            w = (idx * self.scale + self.v_min).reshape(target_shape)
            if x.dtype != torch.float32:
                w = w.to(x.dtype)
            return F.linear(x, w, self.bias)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    @classmethod
    def from_compressed(cls, name: str, data: dict, global_codebooks: dict,
                        use_gpu: bool = True, use_mmap: bool = False, idx_file=None):
        mode = data['mode']
        shape = data['shape']
        layer = cls(name, shape, mode)
        layer.bits = data.get('bits', 8)

        if mode == 'exact':
            raw = data['data']
            if raw.dtype == np.uint16:
                w = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                w = raw.astype(np.float32)
            layer.weight = nn.Parameter(torch.from_numpy(w.reshape(shape)), requires_grad=False)
        elif mode == 'direct_codebook':
            # Priority 1: Local per-tensor codebook (MUST BE FIRST)
            if 'codebook' in data and data['codebook'] is not None and data['codebook'].size > 0:
                cb = data['codebook']
                if isinstance(cb, np.ndarray):
                    cb = torch.from_numpy(cb)
                layer.register_buffer('codebook', cb.float(), persistent=False)
            # Priority 2: Global shared codebook
            else:
                ttype = data.get('codebook_type') or data.get('type')
                if ttype and ttype in global_codebooks:
                    cb = global_codebooks[ttype]
                    if isinstance(cb, np.ndarray):
                        cb = torch.from_numpy(cb)
                    layer.register_buffer('codebook', cb.float(), persistent=False)

            # Use reported bit-width
            layer.bits = data.get('bits', 8)

            # Setup GPU acceleration if possible and requested.
            # The GPU object stores indices in VRAM; no CPU RAM copy is needed when
            # GPU is active, so we skip register_buffer('indices') in that case.
            gpu_active = False

            # Huffman-encoded path: keep bitstream in CPU RAM; GPU matmul via
            # _huff_decode_weights (decode on CPU → upload transiently → GPU matmul).
            # This is the preferred path: VRAM holds only codebooks + SSM state,
            # not packed indices — enabling larger models on smaller cards.
            # The old VRAM-resident Phase 2 path (HuffmanCodebookLinear) is
            # intentionally bypassed so the stream never lives in VRAM.
            if data.get('encoding') == 'huffman' and (
                C_KERNEL_AVAILABLE
                and data.get('huff_row_bit_starts') is not None
                and data.get('huff_lut_sym') is not None
            ):
                _lens = np.asarray(data['huff_lengths'], dtype=np.uint8)
                _max  = int(_lens.max()) if _lens.size and _lens.max() > 0 else 1
                layer._huff_data = {
                    'huff_stream':    np.asarray(data['huff_stream'],         dtype=np.uint8),
                    'lut_sym':        np.asarray(data['huff_lut_sym'],        dtype=np.uint16),
                    'lut_len':        np.asarray(data['huff_lut_len'],        dtype=np.uint8),
                    'sl_first_code':  np.asarray(data['huff_sl_first_code'],  dtype=np.int64),
                    'sl_base_offset': np.asarray(data['huff_sl_base_offset'], dtype=np.int32),
                    'sl_sym':         np.asarray(data['huff_sl_sym'],         dtype=np.uint16),
                    'row_bit_starts': np.asarray(data['huff_row_bit_starts'], dtype=np.int64),
                    'sl_max_len':     _max,
                }
                layer.register_buffer('indices', None)
                # gpu_active stays False — no packed indices in VRAM

            if layer._huff_data is None and not gpu_active and use_gpu and GPU_ACCELERATED_AVAILABLE and HIP_AVAILABLE and hasattr(layer, 'codebook'):
                try:
                    layer._gpu_func = GPUAcceleratedLinear(
                        name, data['indices'], layer.codebook, shape, layer.bits
                    )
                    gpu_active = True
                except Exception as e:
                    print(f"Warning: Failed to setup GPU acceleration for {name}: {e}")

            if gpu_active:
                # Indices are in VRAM; no CPU RAM copy needed regardless of use_mmap.
                layer.register_buffer('indices', None)
            elif use_mmap and idx_file is not None:
                from pathlib import Path as _Path
                from compressed_matmul_cpu import MmappedPackedBuffer
                _idx = _Path(idx_file)
                if _idx.exists():
                    layer._mmap_buf = MmappedPackedBuffer(_idx)
                    layer.register_buffer('indices', None)  # no RAM copy
                else:
                    # .idx file missing — fall back to RAM load
                    layer.register_buffer('indices', torch.from_numpy(data['indices']))
            elif data.get('encoding') == 'huffman' and layer._huff_data is None:
                # Fallback: _huff_data was not set above (Phase-2 tables absent
                # or gcc unavailable) — decode at load time instead.
                from huffman_codebook import huffman_decode_indices
                from bitpack import pack_any_bits
                raw = huffman_decode_indices(
                    data['huff_stream'], data['huff_lengths'], int(data['huff_n'][0])
                )
                packed = pack_any_bits(raw, layer.bits)
                layer.register_buffer('indices', torch.from_numpy(packed))
            elif layer._huff_data is not None:
                pass  # indices already set to None; stream lives in CPU RAM
            else:
                layer.register_buffer('indices', torch.from_numpy(data['indices']))
        elif mode == 'linear_quant':
            layer.register_buffer('indices', torch.from_numpy(data['indices']))
            layer.register_buffer('scale', torch.tensor(data['scale'], dtype=torch.float32))
            layer.register_buffer('v_min', torch.tensor(data['v_min'], dtype=torch.float32))
        return layer

class AdaptiveCodebookEmbedding(nn.Module):
    def __init__(self, name: str, shape: Tuple[int, ...], mode: str = 'exact'):
        super().__init__()
        self.name = name
        # Ensure shape is a tuple of ints
        self.shape = tuple(int(s) for s in shape)
        self.mode = mode
        self.register_buffer('indices', None, persistent=False)
        self.register_buffer('codebook', None, persistent=False)
        self.register_buffer('weight', None, persistent=False)
        self.bits = 8
        self._gpu_func = None
        self._cached_weight = None
        self._mmap_buf  = None  # MmappedPackedBuffer when use_mmap=True
        self._huff_data = None  # dict of numpy arrays for inference-time Huffman decode
        # Some models (e.g. Gemma 3) scale embedding outputs by sqrt(hidden_size)
        # inside the embedding forward.  Preserve that behaviour.
        self.embed_scale: float = 1.0

    def forward(self, x):
        if self.mode == 'exact':
            out = F.embedding(x, self.weight)
            return out * self.embed_scale if self.embed_scale != 1.0 else out

        if self.mode == 'direct_codebook' and self._gpu_func is not None:
            out = self._gpu_func(x)
            # Cast to model dtype and ensure output device matches model device.
            # On GPU→CPU fallback, _gpu_func returns CUDA tensors but the rest of
            # the model is on CPU — must move back or subsequent ops produce garbage.
            target = self.codebook.dtype if self.codebook is not None else torch.bfloat16
            # x (token IDs) is always on the model device — use it as device reference.
            out = out.to(device=x.device, dtype=target)
            return out * self.embed_scale if self.embed_scale != 1.0 else out

        if self.mode == 'direct_codebook' and self._huff_data is not None:
            # Huffman inference-time embedding: decode only the requested token
            # rows on CPU (Huffman bitstream stays in CPU RAM), then move the
            # small decoded result to GPU.  Never decodes the full vocab table.
            vocab, hidden = self.shape
            cb_cpu = self.codebook.cpu().float()
            x_cpu  = x.cpu()

            unique_ids, inverse = torch.unique(x_cpu.reshape(-1), return_inverse=True)
            ids_np = unique_ids.numpy().astype(np.int32)

            out_unique = _huff_embedding(ids_np, self._huff_data,
                                         cb_cpu.numpy(), hidden, C=len(cb_cpu))
            out = torch.from_numpy(out_unique)[inverse].reshape(*x.shape, hidden)
            out = out.to(x.device, non_blocking=True)
            return out * self.embed_scale if self.embed_scale != 1.0 else out

        if self.mode == 'direct_codebook':
            # Per-token decoding: only decode the rows (token IDs) actually present.
            # Avoids unpacking the full vocab×hidden matrix (254M+ elements → minutes).
            cb_cpu = self.codebook.cpu() if self.codebook.device.type != 'cpu' else self.codebook
            index_manager = get_index_manager('cpu')
            if self._mmap_buf is not None:
                indices_cpu = torch.from_numpy(np.ascontiguousarray(self._mmap_buf.as_numpy()))
            else:
                indices_cpu = self.indices.cpu() if self.indices.device.type != 'cpu' else self.indices
            if self.name not in index_manager.lookup_tables:
                index_manager.prepare_lookup_table(self.name, indices_cpu, self.bits)

            hidden = self.shape[1]
            x_cpu = x.cpu()
            unique_ids, inverse = torch.unique(x_cpu.reshape(-1), return_inverse=True)
            rows = []
            for tok_id in unique_ids.tolist():
                start = int(tok_id) * hidden
                row_idx = index_manager.fast_index_lookup(self.name, hidden, start_offset=start)
                row_idx = torch.clamp(row_idx.long(), 0, len(cb_cpu) - 1)
                rows.append(torch.index_select(cb_cpu, 0, row_idx))  # [hidden]
            unique_embeds = torch.stack(rows, dim=0)  # [num_unique, hidden]
            # Reassemble to original shape [*, hidden]
            out = unique_embeds[inverse].reshape(*x.shape, hidden)
            out = out.to(x.device)
            return out * self.embed_scale if self.embed_scale != 1.0 else out

        return F.embedding(x, self.weight) # Fallback

    @classmethod
    def from_compressed(cls, name: str, data: dict, global_codebooks: dict,
                        use_gpu: bool = True, use_mmap: bool = False, idx_file=None):
        mode = data['mode']
        shape = data['shape']
        layer = cls(name, shape, mode)
        layer.bits = data.get('bits', 8)

        if mode == 'exact':
            raw = data['data']
            if raw.dtype == np.uint16:
                w = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                w = raw.astype(np.float32)
            layer.weight = nn.Parameter(torch.from_numpy(w.reshape(shape)), requires_grad=False)
        elif mode == 'direct_codebook':
            # Priority 1: Local per-tensor codebook (MUST BE FIRST)
            if 'codebook' in data and data['codebook'] is not None and data['codebook'].size > 0:
                cb = data['codebook']
                if isinstance(cb, np.ndarray):
                    cb = torch.from_numpy(cb)
                layer.register_buffer('codebook', cb.float(), persistent=False)
            # Priority 2: Global shared codebook
            else:
                ttype = data.get('codebook_type') or data.get('type') or 'embedding'
                if ttype in global_codebooks:
                    cb = global_codebooks[ttype]
                    if isinstance(cb, np.ndarray):
                        cb = torch.from_numpy(cb)
                    layer.register_buffer('codebook', cb.float(), persistent=False)

            # GPU object stores indices in VRAM; no CPU RAM copy needed when GPU active.
            gpu_active = False

            # Huffman path for embedding: keep bitstream in CPU RAM.
            # Decode only requested token rows on-the-fly; upload to GPU after.
            if data.get('encoding') == 'huffman' and (
                C_KERNEL_AVAILABLE
                and data.get('huff_row_bit_starts') is not None
                and data.get('huff_lut_sym') is not None
            ):
                _lens = np.asarray(data['huff_lengths'], dtype=np.uint8)
                _max  = int(_lens.max()) if _lens.size and _lens.max() > 0 else 1
                layer._huff_data = {
                    'huff_stream':    np.asarray(data['huff_stream'],         dtype=np.uint8),
                    'lut_sym':        np.asarray(data['huff_lut_sym'],        dtype=np.uint16),
                    'lut_len':        np.asarray(data['huff_lut_len'],        dtype=np.uint8),
                    'sl_first_code':  np.asarray(data['huff_sl_first_code'],  dtype=np.int64),
                    'sl_base_offset': np.asarray(data['huff_sl_base_offset'], dtype=np.int32),
                    'sl_sym':         np.asarray(data['huff_sl_sym'],         dtype=np.uint16),
                    'row_bit_starts': np.asarray(data['huff_row_bit_starts'], dtype=np.int64),
                    'sl_max_len':     _max,
                }
                layer.register_buffer('indices', None)
                # gpu_active stays False — no packed indices in VRAM

            if layer._huff_data is None and not gpu_active and use_gpu and GPU_ACCELERATED_AVAILABLE and HIP_AVAILABLE and hasattr(layer, 'codebook'):
                try:
                    layer._gpu_func = GPUAcceleratedEmbedding(
                        name, data['indices'], layer.codebook, shape, layer.bits
                    )
                    gpu_active = True
                except Exception as e:
                    print(f"Warning: Failed to setup GPU acceleration for embedding {name}: {e}")

            if gpu_active:
                layer.register_buffer('indices', None)
            elif use_mmap and idx_file is not None:
                from pathlib import Path as _Path
                from compressed_matmul_cpu import MmappedPackedBuffer
                _idx = _Path(idx_file)
                if _idx.exists():
                    layer._mmap_buf = MmappedPackedBuffer(_idx)
                    layer.register_buffer('indices', None)
                else:
                    layer.register_buffer('indices', torch.from_numpy(data['indices']))
            elif data.get('encoding') == 'huffman' and layer._huff_data is None:
                # Fallback: Phase-2 tables absent or gcc unavailable — decode at load.
                from huffman_codebook import huffman_decode_indices
                from bitpack import pack_any_bits
                raw = huffman_decode_indices(
                    data['huff_stream'], data['huff_lengths'], int(data['huff_n'][0])
                )
                layer.register_buffer('indices', torch.from_numpy(pack_any_bits(raw, layer.bits)))
            elif layer._huff_data is not None:
                pass  # indices already set to None; stream lives in CPU RAM
            else:
                layer.register_buffer('indices', torch.from_numpy(data['indices']))
        return layer
