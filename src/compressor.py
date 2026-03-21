"""
On-the-fly compression engine for LLM weights.

Compresses weights during loading without creating intermediate files.
Supports parallel compression using all available CPU cores.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import gc
from multiprocessing import cpu_count


def bfloat16_to_float32(raw_bytes: bytes) -> np.ndarray:
    """Convert raw bfloat16 bytes to float32 numpy array."""
    # bfloat16 is just the top 16 bits of a float32
    # We can convert by padding with 16 zero bits and viewing as float32
    uint16_data = np.frombuffer(raw_bytes, dtype=np.uint16)
    uint32_data = uint16_data.astype(np.uint32) << 16
    return uint32_data.view(np.float32)


def float32_to_bfloat16(data: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to bfloat16 (stored as uint16)."""
    # Rounding to nearest even would be better, but simple truncation is fast
    # and often sufficient for weights.
    uint32_data = data.astype(np.float32).view(np.uint32)
    # Right shift by 16 bits to get the bfloat16 representation
    return (uint32_data >> 16).astype(np.uint16)


def classify_tensor(name: str) -> str:
    """Classify tensor into a semantic category for codebook optimization.

    Supports naming conventions from:
      - Llama 3.x   (model.layers.N.self_attn.q_proj, mlp.gate_proj, ...)
      - Mistral/Devstral  (same as Llama)
      - Gemma 3     (same as Llama + pre_feedforward_layernorm, post_feedforward_layernorm)
      - Qwen 3.5    (model.language_model.layers.N.self_attn.q_proj, ...)
    """
    name_lower = name.lower()

    # 0. Layer norms (all families) — check first so 'norm' doesn't fall through
    # Covers: input_layernorm, post_attention_layernorm, rms_norm,
    #         pre_feedforward_layernorm, post_feedforward_layernorm (Gemma)
    if any(k in name_lower for k in ['layernorm', 'layer_norm', 'ln_', 'rms_norm',
                                       'pre_feedforward_layernorm',
                                       'post_feedforward_layernorm']):
        return 'layernorm'
    # Catch remaining 'norm' patterns (e.g. model.norm, final_layernorm)
    if 'norm' in name_lower:
        return 'layernorm'

    # 1. Routers/Gates (High precision needed for routing)
    if 'router' in name_lower:
        return 'router'
    if 'gate' in name_lower and 'expert' not in name_lower and 'mlp' not in name_lower:
        return 'router'

    # 2. SSM / Linear Attention / REAP components
    if any(k in name_lower for k in ['a_log', 'dt_bias', 'conv1d', 'o_norm']):
        return 'ssm_core'
    if any(k in name_lower for k in ['f_a_proj', 'f_b_proj', 'g_a_proj', 'g_b_proj', 'b_proj']):
        return 'ssm_core'

    # 3. Embedding / Head
    if 'embed' in name_lower or 'lm_head' in name_lower or 'wte' in name_lower:
        return 'embedding'

    # 4. MoE Experts (Mistral uses 'block_sparse', Kimi uses 'w1, w2, w3')
    if '.experts.' in name_lower or 'block_sparse' in name_lower or 'expert' in name_lower:
        return 'moe_expert'
    if any(f'.{i}.weight' in name_lower for i in ['w1', 'w2', 'w3']):
        return 'moe_expert'

    # 5. Attention Layers — Q/K/V/O projections (Llama, Mistral, Gemma, Qwen)
    if 'self_attn' in name_lower or 'attn' in name_lower:
        return 'attention'

    # 6. MLP / Feed-Forward Layers
    # Covers: mlp.gate_proj, mlp.up_proj, mlp.down_proj (Llama/Mistral/Gemma/Qwen)
    if 'mlp' in name_lower or 'ffn' in name_lower or 'feed_forward' in name_lower:
        return 'mlp_ffn'

    # Default fallback
    return 'mlp_ffn'


def kmeans_1d(data: np.ndarray, k: int, max_iters: int = 15,
              sample_size: int = 50000, seed: int = 42) -> np.ndarray:
    """
    Fast 1D k-means clustering using histogram-based approach.
    
    Returns sorted centroids.
    """
    np.random.seed(seed)
    
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample = data[indices]
    else:
        sample = data
    
    # Use percentile-based initialization (preserve input dtype)
    percentiles = np.linspace(0, 100, k + 2)[1:-1]
    centroids = np.percentile(sample, percentiles).astype(data.dtype)
    
    # Simple EM loop — O(N log K) assignment via searchsorted on sorted centroids.
    # Avoids the O(N×K) distance matrix which can be 1-2 GB for large k in lossless mode.
    for _ in range(max_iters):
        # Centroids are always sorted; midpoints define the Voronoi boundaries.
        midpoints = (centroids[:-1] + centroids[1:]) / 2   # shape (k-1,)
        labels = np.searchsorted(midpoints, sample)         # shape (N,), values in [0, k)

        # Vectorized O(N) centroid update via bincount — replaces O(N×K) loop.
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        sums   = np.bincount(labels, weights=sample.astype(np.float64), minlength=k)
        safe_counts = np.where(counts > 0, counts, 1.0)
        new_centroids = np.where(counts > 0, sums / safe_counts, centroids).astype(data.dtype)
        # Re-seed any empty cluster with a random sample point
        empty = np.where(counts == 0)[0]
        if len(empty):
            new_centroids[empty] = sample[np.random.randint(len(sample), size=len(empty))].astype(data.dtype)
        new_centroids = np.sort(new_centroids)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return np.sort(centroids)


def assign_to_codebook(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Assign values to nearest codebook entries."""
    # For large data, using searchsorted on midpoints is much faster than dist matrix
    midpoints = (codebook[:-1] + codebook[1:]) / 2
    return np.searchsorted(midpoints, data).astype(np.uint16)


def _load_cached_tensor(npz_file, tensor_info):
    """Load and decompress a tensor from cache."""
    data = np.load(npz_file)
    mode = str(data['mode'])
    original_name = str(data['name'])
    
    result = {
        'mode': mode,
        'name': original_name,
        'shape': data['shape']
    }
    
    if mode == 'exact':
        result['data'] = data['data']
    elif mode == 'codebook':
        result['type'] = str(data['type'][0])
        result['indices'] = data['indices']
    elif mode == 'direct_codebook':
        result['codebook_type'] = str(data['codebook_type'][0])
        result['bits'] = int(data['bits'][0]) if 'bits' in data.files else 8
        if 'codebook' in data.files:
            result['codebook'] = data['codebook']
        if 'encoding' in data.files and str(data['encoding'][0]) == 'huffman':
            result['encoding']     = 'huffman'
            result['huff_lengths'] = data['huff_lengths']
            result['huff_stream']  = data['huff_stream']
            result['huff_n']       = data['huff_n']
            for key in ('huff_row_bit_starts', 'huff_lut_sym', 'huff_lut_len',
                        'huff_sl_first_code', 'huff_sl_base_offset', 'huff_sl_sym'):
                if key in data.files:
                    result[key] = data[key]
        else:
            result['indices'] = data['indices']
    elif mode == 'q8_packed_7bit':
        result['packed'] = data['packed']
        result['original_len'] = int(data['original_len'][0])
        result['scale'] = float(data['scale'][0])
        result['offset'] = float(data['offset'][0])
        result['unique_q8'] = data['unique_q8']
        result['codebook_values'] = data['codebook_values']
    elif mode == 'linear_quant':
        result['indices'] = data['indices']
        result['scale'] = float(data['scale'][0])
        result['v_min'] = float(data['v_min'][0])
        result['bits'] = int(data['bits'][0]) if 'bits' in data.files else 8
    elif mode == 'q8_codebook':
        result['indices'] = data['indices']
        result['scale'] = float(data['scale'][0])
        result['offset'] = float(data['offset'][0])
        if 'codebook' in data:
            result['codebook'] = data['codebook']

    return original_name, result


def _save_cached_tensor(npz_file, name, data):
    """Save a compressed tensor to disk."""
    save_dict = {
        'name': name,
        'mode': data['mode'],
        'shape': data['shape']
    }
    
    # Add mode-specific data
    if data['mode'] == 'exact':
        save_dict['data'] = data['data']
    elif data['mode'] == 'codebook':
        save_dict['type'] = np.array([data['type']])
        save_dict['indices'] = data['indices']
    elif data['mode'] == 'direct_codebook':
        save_dict['codebook_type'] = np.array([data.get('codebook_type', 'mlp_ffn')])
        save_dict['bits'] = np.array([data.get('bits', 8)])
        if 'codebook' in data:
            save_dict['codebook'] = data['codebook']
        if data.get('encoding') == 'huffman':
            save_dict['encoding']      = np.array(['huffman'])
            save_dict['huff_lengths']  = data['huff_lengths']
            save_dict['huff_stream']   = data['huff_stream']
            save_dict['huff_n']        = data['huff_n']
            # Phase 2 GPU tables (stored when shape is passed to encoder)
            for key in ('huff_row_bit_starts', 'huff_lut_sym', 'huff_lut_len',
                        'huff_sl_first_code', 'huff_sl_base_offset', 'huff_sl_sym'):
                if key in data:
                    save_dict[key] = data[key]
        else:
            save_dict['indices'] = data['indices']
    elif data['mode'] == 'linear_quant':
        save_dict['indices'] = data['indices']
        save_dict['scale'] = np.array([data['scale']])
        save_dict['v_min'] = np.array([data['v_min']])
        save_dict['bits'] = np.array([data.get('bits', 8)])
    elif data['mode'] == 'q8_packed_7bit':
        save_dict['packed'] = data['packed']
        save_dict['original_len'] = np.array([data['original_len']])
        save_dict['scale'] = np.array([data['scale']])
        save_dict['offset'] = np.array([data['offset']])
        save_dict['unique_q8'] = data['unique_q8']
        save_dict['codebook_values'] = data['codebook_values']

    np.savez_compressed(npz_file, **save_dict)


def load_raw_tensor_data(f, offset, size, shape, dtype_str):
    """Low-level raw tensor data loader."""
    f.seek(offset)
    raw = f.read(size)
    try:
        if dtype_str == 'BF16':
            return bfloat16_to_float32(raw).reshape(shape)
        elif dtype_str == 'F16':
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        elif dtype_str == 'F32':
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        else:
            # Try bfloat16 first
            try:
                return bfloat16_to_float32(raw).reshape(shape)
            except:
                return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    except Exception as e:
        print(f"Error loading raw tensor: {e}")
        return np.zeros(shape, dtype=np.float32)


class OnTheFlyCompressor:
    """
    Compresses model weights on-the-fly during loading.
    """
    
    def __init__(self, model_path: Path, cache_dir: Optional[Path] = None,
                 embedding_size: int = 4096, mlp_size: int = 256,
                 force_rebuild: bool = False, compression_mode: str = 'balanced',
                 store_in_model: bool = True, snr_threshold_db: Optional[float] = None):
        self.model_path = model_path
        self.embedding_size = embedding_size
        self.mlp_size = mlp_size
        self.force_rebuild = force_rebuild
        self.snr_threshold_db = snr_threshold_db

        # Determine cache directory — named by quality level so multiple
        # quality tiers can coexist under the same model directory.
        if cache_dir:
            self.cache_dir = cache_dir
        elif store_in_model:
            if compression_mode == 'lossless':
                self.cache_dir = model_path / "codebook-lossless"
            elif snr_threshold_db is not None:
                self.cache_dir = model_path / f"codebook-{int(snr_threshold_db)}dB"
            else:
                self.cache_dir = model_path / "codebook"  # legacy fallback
        else:
            self.cache_dir = model_path.parent / f".{model_path.name}_cache"

        self.compression_mode = compression_mode
        self.codebooks = {}
        self.tensor_info = {}
        self.accuracy_stats = {}
        self.model_hash = ""
        self.num_workers = cpu_count()

    def _compute_model_hash(self) -> str:
        """Compute stable hash of the model files to validate cache."""
        st_files = sorted(self.model_path.glob("*.safetensors"))
        h = hashlib.md5()
        for f in st_files:
            # Just hash filename and size for speed
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
        return h.hexdigest()

    def _get_cache_file(self) -> Path:
        return self.cache_dir / "metadata.json"

    def _load_cache(self, load_tensors: bool = True) -> bool:
        """Load codebooks and metadata from cache."""
        cache_file = self._get_cache_file()
        if not cache_file.exists() or self.force_rebuild:
            return False
            
        try:
            with open(cache_file) as f:
                cache = json.load(f)
                
            if cache['model_hash'] != self.model_hash:
                return False
                
            self.tensor_info = cache['tensor_info']
            self.accuracy_stats = cache.get('accuracy_stats', {})
            
            # Load codebooks (.npy files)
            codebooks_dir = self.cache_dir / "codebooks"
            for npy_file in codebooks_dir.glob("*.npy"):
                # Handle both formats: 'ttype_codebook.npy' and 'ttype_codebook_e...m....npy'
                name = npy_file.name
                if '_codebook_' in name:
                    ttype = name.split('_codebook_')[0]
                elif '_codebook.npy' in name:
                    ttype = name.split('_codebook.npy')[0]
                else:
                    ttype = name.replace('.npy', '')
                
                self.codebooks[ttype] = np.load(npy_file)
            
            if load_tensors:
                # Load all .npz files from tensors/
                tensors_dir = self.cache_dir / "tensors"
                self._loaded_weights = {}
                
                print(f"  Loading {len(self.tensor_info)} compressed tensors from cache...")
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                # Use threads for fast file loading
                npz_files = list(tensors_dir.glob("*.npz"))
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(_load_cached_tensor, f, self.tensor_info) for f in npz_files]
                    for future in as_completed(futures):
                        name, data = future.result()
                        self._loaded_weights[name] = data
            
            return True
        except Exception as e:
            print(f"  Cache load failed: {e}")
            return False

    def _get_compressed_tensor_data(self, name: str) -> Optional[dict]:
        """Get raw compressed tensor data (without decompression)."""
        # 1. Check in-memory weights
        if hasattr(self, '_loaded_weights') and self._loaded_weights and name in self._loaded_weights:
            return self._loaded_weights[name].copy()
        else:
            # 2. Try loading from cache directory if not in memory
            safe_name = name.replace('.', '_').replace('/', '_')
            cache_path = self.cache_dir / "tensors" / f"{safe_name}.npz"
            
            if cache_path.exists():
                try:
                    _, data = _load_cached_tensor(cache_path, self.tensor_info)
                    return data
                except Exception as e:
                    print(f"Error loading compressed data for {name}: {e}")
                    return None
            else:
                return None

    def _save_cache(self):
        """Save codebooks and compressed tensors to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self._get_cache_file()
        cache = {
            'model_hash': self.model_hash,
            'embedding_size': self.embedding_size,
            'mlp_size': self.mlp_size,
            'compression_mode': self.compression_mode,
            'tensor_info': self.tensor_info,
            'accuracy_stats': self.accuracy_stats
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
        
        # Save codebooks
        codebooks_dir = self.cache_dir / "codebooks"
        codebooks_dir.mkdir(exist_ok=True)
        for ttype, cb in self.codebooks.items():
            npy_file = codebooks_dir / f"{ttype}_codebook_e{self.embedding_size}m{self.mlp_size}.npy"
            np.save(npy_file, cb.astype(np.float32))
        
        # Save compressed tensors (only if they exist in memory)
        if hasattr(self, '_loaded_weights') and self._loaded_weights:
            tensors_dir = self.cache_dir / "tensors"
            tensors_dir.mkdir(exist_ok=True)
            for name, data in self._loaded_weights.items():
                safe_name = name.replace('.', '_').replace('/', '_')
                npz_file = tensors_dir / f"{safe_name}.npz"
                _save_cached_tensor(npz_file, name, data)

    def _get_codebook_size(self, ttype: str) -> int:
        if ttype == 'embedding': return self.embedding_size
        if ttype == 'router': return 0 # Keep exact
        if ttype == 'ssm_core': return 0 # Keep exact
        return self.mlp_size

    def _track_accuracy(self, original: np.ndarray, reconstructed: np.ndarray, tensor_type: str):
        """Track reconstruction accuracy."""
        error = np.abs(original - reconstructed)
        self.accuracy_stats[tensor_type] = self.accuracy_stats.get(tensor_type, [])
        self.accuracy_stats[tensor_type].append(float(error.mean()))

    def load_compressed(self, callback=None, load_tensors: bool = True):
        """Load weights with metadata-only option."""
        print(f"\n{'='*80}")
        print(f"ON-THE-FLY COMPRESSION (load_tensors={load_tensors})")
        print(f"{'='*80}")
        print(f"Model: {self.model_path}")
        print(f"{'='*80}\n")
        
        self.model_hash = self._compute_model_hash()
        if self._load_cache(load_tensors=load_tensors):
            if not hasattr(self, '_loaded_weights'): self._loaded_weights = {}
            return self._loaded_weights, {
                'compression_method': 'cached',
                'tensor_count': len(self.tensor_info),
                'config': {}
            }
        
        # Rebuild required
        return self.compress_and_save()

    def compress_and_save(self):
        """Generic compression pass."""
        # ... logic implemented in AdaptiveCompressor ...
        pass

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        """Get decompressed tensor by name."""
        data = self._get_compressed_tensor_data(name)
        if data is None: return None
        
        mode = data['mode']
        if mode == 'exact':
            raw = data['data']
            return (raw.astype(np.uint32) << 16).view(np.float32).reshape(data['shape'])
        elif mode == 'direct_codebook':
            indices = data['indices']
            cb = data.get('codebook') or self.codebooks.get(data.get('codebook_type'))
            if cb is None: return None
            return cb[indices].reshape(data['shape'])
        return None
