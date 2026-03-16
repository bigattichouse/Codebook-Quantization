"""
Adaptive Compressor

Integrates Q8 quantization, bit-packing, and direct codebook compression
with per-tensor strategy selection.
Supports parallel compression for high performance.
"""

import json
import hashlib
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from multiprocessing import cpu_count

from compressor import (
    OnTheFlyCompressor,
    bfloat16_to_float32,
    float32_to_bfloat16,
    classify_tensor,
    kmeans_1d,
    assign_to_codebook
)
from q8_utils import quantize_q8, dequantize_q8, count_unique_q8
from bitpack import (
    pack_7bit_indices, unpack_7bit_indices,
    pack_6bit_indices, unpack_6bit_indices,
    pack_4bit_indices, unpack_4bit_indices,
    calculate_packed_size
)
from analyze_tensor import analyze_tensor


def _rss() -> str:
    """Return current process RSS as a short string, or '' if psutil unavailable."""
    try:
        import psutil
        return f" rss={psutil.Process().memory_info().rss/1e9:.1f}GB"
    except Exception:
        return ""


def _compress_adaptive_worker(name, file_path, offset, size, shape, dtype_str,
                               tensor_name, compression_mode, global_codebooks,
                               mse_threshold=0.0001, native_bits=16, unique_count=None,
                               target_bits=None):
    """
    Worker function for multi-tier meta-analysis.

    Loads its own tensor slice via mmap so the main thread's in_flight deque holds
    only lightweight metadata (no float32 arrays accumulate between submissions).
    Prioritizes global shared codebooks for maximum RAM efficiency.
    """
    import mmap as mmap_module
    import numpy as np
    from compressor import float32_to_bfloat16, classify_tensor, kmeans_1d, assign_to_codebook
    from q8_utils import quantize_q8, dequantize_q8
    from bitpack import pack_7bit_indices, pack_4bit_indices

    # Sub-step progress for large tensors (> 1M params) so the user sees activity.
    _n_params_approx = size // 2  # BF16: 2 bytes/param; good enough for threshold
    _verbose = _n_params_approx > 1_000_000
    def _step(msg: str):
        if _verbose:
            print(f"               {msg}", flush=True)

    # Load only the slice we need — seek+read avoids mapping the entire shard file
    # into virtual memory (a 19 GB file mmap'd for a 2 GB tensor bloats RSS by 19 GB).
    _step(f"reading {size/1e9:.2f}GB...")
    with open(file_path, 'rb') as _f:
        _f.seek(offset)
        raw = _f.read(size)

    # ── Fast lossless BF16 path ──────────────────────────────────────────────
    # Stay in uint16 throughout — identical to Pass 1 plus a LUT assign step.
    # No float32 conversion, no O(N log K) searchsorted.
    #
    # Memory:  raw (2B) shared with bf16_idx via frombuffer (zero-copy),
    #          then indices (2B), peak ~4 GB for 1B-param tensor.
    # Compare: normal path peaks at ~10 GB (raw + float32 + bf16_idx + indices).
    if compression_mode == 'lossless' and dtype_str == 'BF16' and (size // 2) >= 1000:
        bf16_idx = np.frombuffer(raw, dtype=np.uint16)   # zero-copy view of raw
        del raw                                           # Python ref gone; buffer kept alive by bf16_idx

        _step(f"histogram {len(bf16_idx)//1_000_000}M uint16 values...")
        hist = np.bincount(bf16_idx, minlength=65536)
        nonzero = np.where(hist > 0)[0].astype(np.uint16)
        actual_unique = len(nonzero)
        cb_f32 = (nonzero.astype(np.uint32) << 16).view(np.float32).copy()
        del hist

        # Signal power: convert only a tiny sample to float32
        _ns = min(50000, len(bf16_idx))
        _s_f32 = (bf16_idx[:_ns].astype(np.uint32) << 16).view(np.float32)
        signal_power = float(np.mean(_s_f32.astype(np.float64) ** 2))
        del _s_f32

        # LUT assignment: O(N) array index — no binary search, no log factor
        _step(f"assigning {len(bf16_idx)//1_000_000}M values via LUT...")
        lut = np.zeros(65536, dtype=np.uint16)
        lut[nonzero] = np.arange(actual_unique, dtype=np.uint16)
        indices = lut[bf16_idx]
        del lut, bf16_idx, nonzero

        bits = int(np.ceil(np.log2(max(actual_unique, 2))))
        if bits not in [8, 16]:
            _step(f"packing {len(indices)//1_000_000}M indices → {bits}-bit...")
            from bitpack import pack_any_bits
            indices = pack_any_bits(indices, bits)

        return name, {
            'mode': 'direct_codebook', 'indices': indices,
            'bits': bits, 'codebook': cb_f32, 'shape': shape, 'mse': 0.0,
        }, f'{bits}-bit (lossless)', actual_unique, 0.0, 100.0
    # ── end fast lossless BF16 path ─────────────────────────────────────────

    # Convert raw bytes → float32.  Explicitly del intermediate views so raw's
    # refcount drops to zero and its 2 GB are freed before the compression work starts.
    if dtype_str == 'BF16':
        _u16 = np.frombuffer(raw, dtype=np.uint16)
        flat = (_u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        del _u16
    elif dtype_str == 'F16':
        flat = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
    elif dtype_str == 'F32':
        flat = np.array(np.frombuffer(raw, dtype=np.float32).reshape(shape))  # force copy so raw can free
    else:
        try:
            _u16 = np.frombuffer(raw, dtype=np.uint16)
            flat = (_u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            del _u16
        except Exception:
            flat = np.zeros(shape, dtype=np.float32)
    del raw

    name_low = tensor_name.lower()

    # Ensure data is flattened for analysis
    flat = flat.flatten()
    # unique_count supplied by Pass 1 (free, from bincount) — avoids np.unique on the full tensor
    if unique_count is None:
        unique_count = len(np.unique(flat))

    # 1. Critical Layer Detection
    # These layers are sensitive and MUST be bit-perfect (0% loss)
    is_critical = any(k in name_low for k in ['norm', 'ln_', 'router', 'ssm_core'])
    if 'gate' in name_low and 'experts' not in name_low:
        is_critical = True
        
    # Early return only for tiny tensors (not worth the codebook overhead)
    if flat.size < 1000:
        return name, {
            'mode': 'exact', 'data': float32_to_bfloat16(flat), 'shape': shape
        }, 'exact (tiny)', unique_count, 0.0, 100.0

    # 2. Quality Thresholds
    # For balanced/lossy modes, use SNR (dB) instead of absolute MSE.
    # SNR is scale-invariant: a tensor with signal_power=0.001 and MSE=0.0002
    # has SNR ≈ 7 dB (terrible), while the same MSE on a normal-scale tensor
    # (signal_power=1.0) gives SNR ≈ 37 dB (excellent).
    # Lossless and critical tensors keep the absolute MSE check (≤ 1e-9).
    if is_critical or compression_mode == 'lossless':
        threshold = 1e-9
        snr_threshold_db = None  # use absolute MSE
    elif compression_mode == 'balanced':
        threshold = None
        snr_threshold_db = 30.0  # ≥ 30 dB SNR required
    elif compression_mode == 'lossy':
        threshold = None
        snr_threshold_db = 25.0  # ≥ 25 dB SNR required
    else:  # adaptive or custom mse_threshold
        threshold = mse_threshold
        snr_threshold_db = None

    def get_mse(reconstructed, original):
        return float(np.mean((original - reconstructed) ** 2))

    def get_snr(reconstructed, original):
        """Calculate Signal-to-Noise Ratio in dB."""
        signal = np.mean(original ** 2)
        noise = np.mean((original - reconstructed) ** 2)
        if noise < 1e-12: return 100.0 # Effectively perfect
        return 10 * np.log10(signal / noise)

    ttype = classify_tensor(tensor_name)

    # Pre-compute a sample for MSE estimation — full-tensor assign only happens once at selection.
    _n = len(flat)
    _sample_size = min(50000, _n)
    if _n > _sample_size:
        _idx = np.random.choice(_n, _sample_size, replace=False)
        flat_sample = flat[_idx]
        del _idx
    else:
        flat_sample = flat

    # Signal power for SNR calculations (scale-invariant quality metric).
    signal_power = float(np.mean(flat_sample.astype(np.float32) ** 2))

    def quality_passes(mse: float) -> bool:
        """Return True if MSE meets the quality bar (SNR-based or absolute)."""
        if snr_threshold_db is not None:
            if mse < 1e-12:
                return True  # essentially perfect
            if signal_power < 1e-12:
                return False  # near-zero tensor, can't compress meaningfully
            snr = 10.0 * np.log10(signal_power / mse)
            return snr >= snr_threshold_db
        else:
            return mse <= threshold

    # --- Targeted bit-width search (2–4 checks max, not a full scan) ---
    #
    # min_bits_exact = ceil(log2(unique_count)) — minimum bits to hold all unique values.
    # Lossless:  only one check at min_bits_exact.
    # Balanced:  try min_bits_exact-2, min_bits_exact-1; then Q8 linear fallback.
    # Lossy:     try min_bits_exact-3, min_bits_exact-2, min_bits_exact-1.
    _uc = max(unique_count, 2) if unique_count else 2
    min_bits_exact = int(np.ceil(np.log2(_uc)))

    if compression_mode == 'lossless':
        # Search 2 steps below min_bits_exact too: tail values may be so rare
        # (1-2 occurrences out of millions) that k-means at min_bits_exact-1 or -2
        # achieves MSE ≤ 1e-9, saving 1-2 bits per element with no practical loss.
        _raw_cands = list(range(max(8, min_bits_exact - 2), min_bits_exact + 1))
    elif compression_mode == 'balanced':
        # Search 8→12 bits ascending (most compressed first), stop at first pass.
        # This lets a hard layer step up to 9, 10, 11-bit k-means before falling
        # back to Q8 linear — much better than jumping from 12-bit k-means
        # straight to 8-bit uniform quantization.
        max_bal = min(min_bits_exact, 12)
        _raw_cands = list(range(8, max_bal + 1))
    else:  # lossy — hard cap at target_bits (default 8)
        cap = target_bits if target_bits is not None else 8
        # Try cap and cap-1 only; honour the hard ceiling the user requested.
        _raw_cands = [cap - 1, cap]
    search_candidates = sorted(set(w for w in _raw_cands if 3 <= w <= native_bits))

    best_strategy = None
    min_bits = native_bits + 1

    # 1. Check Global Shared Codebook first — use sample for MSE estimation.
    global_cb = global_codebooks.get(ttype)
    if global_cb is not None:
        bits_g = int(np.ceil(np.log2(len(global_cb))))
        mse_g = get_mse(global_cb[assign_to_codebook(flat_sample, global_cb)], flat_sample)
        if quality_passes(mse_g):
            min_bits = bits_g
            best_strategy = {
                'mode': 'direct_codebook', 'indices': None, 'bits': bits_g,
                'codebook_type': ttype, 'shape': shape, 'mse': mse_g, 'label': f'shared-{ttype}',
                '_cb': global_cb,
            }

    # 2. Targeted candidate search — ascending (most compressed first), stop at first pass.
    # Lossless uses threshold (1e-9) not strict 0.0: k-means at min_bits_exact-1 with
    # 99.999%+ coverage achieves MSE ~1e-15, well within the threshold.
    for bits in search_candidates:
        if bits >= min_bits: continue

        k = 2 ** bits
        if k > flat.size * 0.5: continue

        if compression_mode == 'lossless' and unique_count <= k:
            # Lossless: MSE=0 is guaranteed — O(N) histogram codebook, free flat immediately.
            _step(f"histogram {flat.size//1_000_000}M values...")
            bf16_idx = (flat.view(np.uint32) >> 16).astype(np.uint16)
            hist = np.bincount(bf16_idx, minlength=65536)
            nonzero = np.where(hist > 0)[0].astype(np.uint16)
            cb_km = (nonzero.astype(np.uint32) << 16).view(np.float32)
            _step(f"assigning {flat.size//1_000_000}M values ({bits}-bit lossless)...")
            indices = np.searchsorted(nonzero, bf16_idx).astype(np.uint16)
            del bf16_idx, hist, nonzero, flat
            flat = None
            min_bits = bits
            best_strategy = {
                'mode': 'direct_codebook', 'indices': indices,
                'bits': bits, 'codebook': cb_km, 'shape': shape, 'mse': 0.0,
                'label': f'{bits}-bit (lossless)'
            }
            break
        else:
            _step(f"k-means k={k} on {min(_n_params_approx, 50000)//1000}K sample...")
            cb_km = kmeans_1d(flat_sample, k, seed=42)
            idx_sample = assign_to_codebook(flat_sample, cb_km)
            mse_km = get_mse(cb_km[idx_sample], flat_sample)
            del idx_sample
            if quality_passes(mse_km):
                min_bits = bits
                _step(f"assigning {flat.size//1_000_000}M values ({bits}-bit)...")
                best_strategy = {
                    'mode': 'direct_codebook',
                    'indices': assign_to_codebook(flat, cb_km),
                    'bits': bits, 'codebook': cb_km, 'shape': shape, 'mse': mse_km,
                    'label': f'{bits}-bit (local)'
                }
                break  # first passing candidate wins
            del cb_km

    # 3. Linear quant Q8 fallback — balanced mode only, if no strategy found yet.
    if best_strategy is None and compression_mode == 'balanced' and flat is not None:
        bits, k = 8, 256
        v_min, v_max = flat_sample.min(), flat_sample.max()
        scale = (v_max - v_min) / (k - 1) if v_max > v_min else 1.0
        q_s = np.round((flat_sample - v_min) / scale).clip(0, k - 1)
        mse_lin = get_mse(q_s * scale + v_min, flat_sample)
        del q_s
        if quality_passes(mse_lin):
            v_min_f, v_max_f = flat.min(), flat.max()
            scale_f = (v_max_f - v_min_f) / (k - 1) if v_max_f > v_min_f else 1.0
            q_full = np.round((flat - v_min_f) / scale_f).clip(0, k - 1)
            min_bits = bits
            best_strategy = {
                'mode': 'linear_quant', 'bits': bits, 'v_min': v_min_f, 'scale': scale_f,
                'indices': q_full.astype(np.uint16), 'shape': shape, 'mse': mse_lin,
                'label': f'{bits}-bit (linear)'
            }
            del q_full

    # Resolve deferred full-tensor assign for global shared codebook hit
    if best_strategy and best_strategy.get('indices') is None:
        _cb = best_strategy.pop('_cb')
        if flat is not None:
            _step(f"assigning {flat.size//1_000_000}M values (shared codebook)...")
        best_strategy['indices'] = assign_to_codebook(flat, _cb) if flat is not None else None
        if flat is not None:
            del flat; flat = None

        # Final Decision
    if flat is not None:
        del flat  # free before packing if not already freed in lossless branch

    if best_strategy and min_bits < 16:
        from bitpack import pack_any_bits
        if min_bits not in [8, 16]:
            _step(f"packing {len(best_strategy['indices'])//1_000_000}M indices → {min_bits}-bit...")
            best_strategy['indices'] = pack_any_bits(best_strategy['indices'], min_bits)

        label = best_strategy.pop('label')
        mse = best_strategy.pop('mse')
        snr = 100.0
        if mse > 0 and signal_power > 0:
            snr = 10 * np.log10(signal_power / mse)

        return name, best_strategy, label, unique_count, mse, snr

    # Fallback to Exact — flat was freed above so reload from file
    with open(file_path, 'rb') as _f:
        _f.seek(offset)
        _raw = _f.read(size)
    if dtype_str == 'BF16':
        _u16 = np.frombuffer(_raw, dtype=np.uint16)
        exact_data = float32_to_bfloat16((_u16.astype(np.uint32) << 16).view(np.float32).flatten())
        del _u16
    else:
        exact_data = float32_to_bfloat16(np.frombuffer(_raw, dtype=np.float32).flatten())
    del _raw
    return name, {
        'mode': 'exact', 'data': exact_data, 'shape': shape
    }, 'exact (fallback)', unique_count, 0.0, 100.0


class AdaptiveCompressor(OnTheFlyCompressor):
    """Adaptive compression with parallel processing and per-tensor strategy selection."""
    
    def __init__(self, model_path: Path, cache_dir: Optional[Path] = None,
                 compression_mode: str = 'balanced', sample_size: Optional[int] = None,
                 force_rebuild: bool = False, store_in_model: bool = True,
                 num_workers: Optional[int] = None, mse_threshold: float = 0.005,
                 target_bits: Optional[int] = None, snr_db: Optional[float] = None):
        # Resolve SNR target: explicit --db overrides mode default.
        # Modes are convenience aliases for dB targets; lossless uses absolute MSE.
        if snr_db is not None:
            snr_threshold_db = float(snr_db)
        elif compression_mode == 'balanced':
            snr_threshold_db = 30.0
        elif compression_mode == 'lossy':
            snr_threshold_db = 25.0
        else:
            snr_threshold_db = None  # lossless / exact: MSE-based

        super().__init__(
            model_path=model_path,
            cache_dir=cache_dir,
            compression_mode=compression_mode,
            force_rebuild=force_rebuild,
            store_in_model=store_in_model,
            snr_threshold_db=snr_threshold_db,
        )
        self.snr_threshold_db = snr_threshold_db
        self._snr_values: list = []  # collect per-tensor actual SNR for reporting
        self.sample_size = sample_size
        self.num_workers = num_workers or cpu_count()
        self.mse_threshold = mse_threshold
        self.target_bits = target_bits  # hard cap for lossy mode; None = use mode default
        self.max_codebook_size = {
            'lossless': 8192,
            'balanced': 4096,   # balanced now searches 8-12 bit adaptively
            'lossy': 2 ** target_bits if target_bits else 256
        }.get(compression_mode, 256)
        
        self.strategy_stats = {
            'exact': 0,
            'direct_codebook': 0,
            'q8_packed_7bit': 0,
            'q8_codebook': 0
        }
        
        # Detect native bit depth (BF16=16, FP8/INT8=8)
        self.native_bits = 16
        self.config = {}
        config_file = self.model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    self.config = json.load(f)
                    q_config = self.config.get("quantization_config", {})
                    if q_config.get("quant_method") in ["fp8", "int8"]:
                        self.native_bits = 8
                        print(f"  Detected Native {q_config.get('quant_method').upper()} model (8-bit)")
            except: pass

    def get_tensor_compressed_data(self, name: str) -> dict:
        """
        Get compressed data for a tensor (for module creation).
        
        Args:
            name: Tensor name
            
        Returns:
            Dictionary with compressed data for layer creation
        """
        if name not in self.tensor_info:
            raise ValueError(f"Tensor {name} not found in compression cache")
        
        info = self.tensor_info[name]
        
        # Load the compressed data from cache
        try:
            from compressor import _load_cached_tensor
            npz_file = self._get_cache_file().parent / f"{name.replace('.', '_')}.npz"
            
            if npz_file.exists():
                _, data = _load_cached_tensor(npz_file, self.tensor_info)
                return data
            else:
                # Fallback to tensor_info data
                return info
                
        except Exception as e:
            print(f"Warning: Could not load cached data for {name}: {e}")
            return info

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        """Get a decompressed tensor by name. Supports streaming from cache."""
        # 1. Check in-memory weights
        if hasattr(self, '_loaded_weights') and self._loaded_weights and name in self._loaded_weights:
            data = self._loaded_weights[name]
        else:
            # 2. Try loading from cache directory if not in memory
            from compressor import _load_cached_tensor
            safe_name = name.replace('.', '_').replace('/', '_')
            cache_path = self.cache_dir / "tensors" / f"{safe_name}.npz"
            
            if cache_path.exists():
                try:
                    _, data = _load_cached_tensor(cache_path, self.tensor_info)
                except Exception as e:
                    print(f"Error streaming {name} from cache: {e}")
                    return None
            else:
                _tied = ('lm_head',)
                if not any(t in name for t in _tied):
                    print(f"Error: Cache file missing for {name} at {cache_path}")
                return None  # Don't fall back to on-demand compression
            
        try:
            return self._decompress_tensor_adaptive(data)
        except Exception as e:
            print(f"Error decompressing {name}: {e}")
            return None

    def _get_compressed_tensor_data(self, name: str) -> Optional[dict]:
        """Get raw compressed tensor data (without decompression) for compressed modules."""
        # 1. Check in-memory weights
        if hasattr(self, '_loaded_weights') and self._loaded_weights and name in self._loaded_weights:
            return self._loaded_weights[name].copy()
        else:
            # 2. Try loading from cache directory if not in memory
            from compressor import _load_cached_tensor
            safe_name = name.replace('.', '_').replace('/', '_')
            cache_path = self.cache_dir / "tensors" / f"{safe_name}.npz"
            
            if cache_path.exists():
                try:
                    _, data = _load_cached_tensor(cache_path, self.tensor_info)
                    return data.copy() if data else None
                except Exception as e:
                    print(f"Error loading compressed data for {name}: {e}")
                    return None
            else:
                # Suppress noise for known tied weights — caller handles the fallback.
                _tied = ('lm_head',)
                if not any(t in name for t in _tied):
                    print(f"Error: Cache file missing for {name} at {cache_path}")
                return None

    def _compress_and_decompress_on_demand(self, name: str) -> Optional[np.ndarray]:
        """Load, compress, and immediately decompress a tensor on demand."""
        if name not in self.tensor_info:
            return None
        
        try:
            info = self.tensor_info[name]
            st_file = self.model_path / info['file']
            
            from compressor import load_raw_tensor_data
            with open(st_file, 'rb') as f:
                # Load original tensor
                data = load_raw_tensor_data(f, info['offset'], info['size'], info['shape'], info['dtype'])
                
                # For critical tensors, return original data to avoid corruption
                if (data.size < 10000 or 
                    'norm' in name.lower() or 
                    'layernorm' in name.lower() or
                    'ln_' in name.lower() or
                    'lm_head' in name.lower() or
                    'embed_tokens' in name.lower() or
                    info['type'] in ['layernorm', 'router', 'embedding']):
                    return data
                
                # For larger tensors, try compression with quality check
                compressed = self._compress_single_tensor(data, name, info['type'])
                decompressed = self._decompress_tensor_adaptive(compressed)
                
                # Quality check - if MSE is too high, return original
                mse = np.mean((data.flatten() - decompressed.flatten()) ** 2)
                if mse > 0.001:  # Much stricter error threshold
                    print(f"Warning: High compression error for {name} (MSE={mse:.6f}), using original")
                    return data
                
                return decompressed
                
        except Exception as e:
            print(f"Error in on-demand compression for {name}: {e}")
            return None

    def _decompress_tensor_adaptive(self, data: dict) -> np.ndarray:
        """Decompress a tensor compressed with adaptive strategy."""
        mode = data['mode']
        if mode == 'exact':
            raw = data['data']
            w = (raw.astype(np.uint32) << 16).view(np.float32)
            return w.reshape(data['shape'])
        
        elif mode == 'direct_codebook':
            indices = data['indices']
            bits = data.get('bits', 8)
            
            # --- OPTIMIZED BITSTREAM UNPACKING ---
            if bits not in [8, 16]:
                n_elements = 1
                for dim in data['shape']: n_elements *= dim
                
                # Use FastIndexManager for large tensors (>10K elements)
                if n_elements > 10000:
                    try:
                        from fast_index_manager import get_index_manager
                        import torch
                        
                        print(f"  [FastDecompress] Using FastIndexManager for {n_elements:,} elements ({bits}-bit)")
                        
                        # Create a unique tensor name for this decompression
                        tensor_name = f"decompress_{hash(str(data.get('shape', [])))}"
                        
                        index_manager = get_index_manager()
                        
                        # Convert numpy to torch for FastIndexManager
                        indices_tensor = torch.from_numpy(indices)
                        index_manager.prepare_lookup_table(tensor_name, indices_tensor, bits)
                        
                        # Fast lookup
                        start_time = time.time()
                        unpacked_torch = index_manager.fast_index_lookup(tensor_name, n_elements)
                        fast_time = time.time() - start_time
                        
                        indices = unpacked_torch.cpu().numpy().astype(np.uint16)
                        print(f"  [FastDecompress] ✅ Completed in {fast_time:.3f}s")
                        
                    except Exception as e:
                        print(f"  [FastDecompress] ❌ Failed: {e}, falling back to slow method")
                        from bitpack import unpack_any_bits
                        indices = unpack_any_bits(indices, bits, n_elements)
                else:
                    # Use original method for small tensors
                    from bitpack import unpack_any_bits
                    indices = unpack_any_bits(indices, bits, n_elements)
            
            # Use per-tensor codebook if available, otherwise global
            codebook = data.get('codebook')
            if codebook is None:
                ttype = data.get('codebook_type') or classify_tensor(data.get('name', ''))
                codebook = self.codebooks.get(ttype)
                
                # FALLBACK CHAIN
                if codebook is None:
                    for fb in ['attention', 'moe_expert', 'mlp_ffn', 'embedding']:
                        if fb in self.codebooks:
                            codebook = self.codebooks[fb]; break
            
            if codebook is None:
                raise ValueError(f"No codebook for direct_codebook mode")
            return codebook[indices].reshape(data['shape'])
        
        elif mode == 'q8_packed_7bit':
            packed = data['packed']
            original_len = data['original_len']
            scale = data['scale']
            offset = data['offset']
            unique_q8 = data['unique_q8']
            indices = unpack_7bit_indices(packed, original_len)
            q8 = unique_q8[indices]
            return (q8.astype(np.float32) * scale + offset).reshape(data['shape'])
        
        elif mode == 'q8_codebook':
            indices = data['indices']
            # Use per-tensor codebook
            codebook = data.get('codebook')
            if codebook is None:
                # Fallback to global if missing
                ttype = classify_tensor(data.get('name', ''))
                codebook = self.codebooks.get(ttype)
                
                # FALLBACK CHAIN
                if codebook is None:
                    for fb in ['attention', 'moe_expert', 'mlp_ffn', 'embedding']:
                        if fb in self.codebooks:
                            codebook = self.codebooks[fb]; break
                
            if codebook is None:
                raise ValueError(f"No codebook for q8_codebook mode")
            
            # Reconstruct dequantized values from codebook
            return codebook[indices].reshape(data['shape'])
        
        elif mode == 'codebook':
            # Standard codebook mode fallback
            indices = data['indices']
            ttype = data.get('type') or data.get('codebook_type')
            
            # Map legacy type to new semantic categories if needed
            if ttype == 'mlp_attn':
                name = data.get('name', '').lower()
                ttype = 'attention' if 'attn' in name else 'mlp_ffn'
                
            codebook = self.codebooks.get(ttype)
            
            # FALLBACK CHAIN
            if codebook is None:
                for fb in ['attention', 'moe_expert', 'mlp_ffn', 'embedding']:
                    if fb in self.codebooks:
                        codebook = self.codebooks[fb]; break
                
            if codebook is None:
                # Emergency fallback: create a simple linear codebook
                max_idx = np.max(indices) if len(indices) > 0 else 255
                codebook = np.linspace(-1.0, 1.0, max_idx + 1)
                print(f"Warning: Using emergency linear codebook for {ttype}")
                
            # Safety check for index bounds
            if np.max(indices) >= len(codebook):
                print(f"Warning: Index out of bounds for {ttype}, clipping indices")
                indices = np.clip(indices, 0, len(codebook) - 1)
                
            return codebook[indices].reshape(data['shape'])
            
        elif mode == 'linear_quant':
            indices = data['indices'].astype(np.float32)
            scale = float(data['scale'])
            v_min = float(data['v_min'])
            return (indices * scale + v_min).reshape(data['shape'])

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_compressed(self, callback=None, load_tensors: bool = True):
        """Two-pass compression: 1) Analyze & build codebooks, 2) Stream compress & save."""
        print(f"\n{'='*80}")
        print(f"STREAMING ADAPTIVE COMPRESSION (mode={self.compression_mode})")
        print(f"{'='*80}")
        print(f"Model: {self.model_path}")
        print(f"Load tensors: {load_tensors}")
        print(f"{'='*80}\n")
        
        self.model_hash = self._compute_model_hash()
        
        # Check for existing cache
        if self._load_cache(load_tensors=load_tensors):
            print("✓ Loaded from existing cache")
            # Ensure _loaded_weights exists
            if not hasattr(self, '_loaded_weights'):
                self._loaded_weights = {}
            metadata = {
                'compression_method': 'adaptive_cached',
                'compression_mode': self.compression_mode,
                'tensor_count': len(self.tensor_info),
                'config': getattr(self, 'config', {}),
                'global_codebooks': self._load_global_codebooks()
            }
            return self._loaded_weights, metadata

        # PASS 1: Analysis and codebook building
        print("PASS 1: Analyzing model structure and building codebooks...")
        self._analyze_and_build_codebooks()
        
        # PASS 2: Stream compress and save
        print("\nPASS 2: Streaming compression and saving to disk cache...")
        weights, metadata = self._stream_compress_and_save()
        
        if not load_tensors:
            # In streaming mode, we don't want to keep the weights in RAM
            # but we definitely wanted them saved to disk (which Pass 2 does)
            print("  Discarding in-memory weights (Streaming Mode enabled)")
            self._loaded_weights = {}
            weights = {}
            metadata['compression_method'] = 'streaming_disk_ready'
            
        return weights, metadata

    def _analyze_and_build_codebooks(self):
        """PASS 1: Global Histogram Analysis - Samples EVERY parameter in the model."""
        import gc
        from concurrent.futures import ThreadPoolExecutor
        
        print("  Scanning all tensors for global histogram analysis...")
        
        self.tensor_info = {}
        # Histograms size based on native bit depth
        hist_size = 65536 if self.native_bits == 16 else 256
        categories = ['embedding', 'attention', 'mlp_ffn', 'moe_expert', 'router', 'ssm_core']
        category_hists = {k: np.zeros(hist_size, dtype=np.uint64) for k in categories}
        
        # Load config and scan metadata.
        # Prefer model.safetensors.index.json — it lists exactly the shard files
        # that HuggingFace uses, excluding alternative-format files like Mistral's
        # consolidated.safetensors (which uses different tensor naming conventions
        # and would cause double-counting if included).
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as _f:
                _idx = json.load(_f)
            _shard_names = sorted(set(_idx["weight_map"].values()))
            st_files = [self.model_path / n for n in _shard_names]
            print(f"  Using model.safetensors.index.json ({len(st_files)} shards)")
        else:
            st_files = sorted(self.model_path.glob("*.safetensors"))
        for st_file in st_files:
            with open(st_file, 'rb') as f:
                header_size = int.from_bytes(f.read(8), 'little')
                header = json.loads(f.read(header_size).decode('utf-8'))
                base_offset = f.tell()
                for name, info in header.items():
                    if name == '__metadata__': continue
                    self.tensor_info[name] = {
                        'shape': tuple(info['shape']), 'type': classify_tensor(name),
                        'file': st_file.name, 'offset': base_offset + info['data_offsets'][0],
                        'size': info['data_offsets'][1] - info['data_offsets'][0],
                        'dtype': info.get('dtype', 'BF16')
                    }

        print(f"  Analyzing {len(self.tensor_info)} tensors (100% coverage, {self.native_bits}-bit native)...")
        start_time = time.time()
        
        # Process files serially to build histograms — keeps RAM flat and progress visible.
        # (Parallel version used 8 threads which could accumulate multiple large raw reads.)
        total_tensors_p1 = len(self.tensor_info)
        tensor_count = 0
        for file_idx, st_file in enumerate(st_files, 1):
            fname = st_file.name
            tensors_in_file = [n for n, i in self.tensor_info.items() if i['file'] == fname]
            print(f"  Pass 1 [{file_idx}/{len(st_files)}]: {fname} ({len(tensors_in_file)} tensors)")
            with open(st_file, 'rb') as f:
                for name in tensors_in_file:
                    info = self.tensor_info[name]
                    ttype = info['type']
                    if ttype not in category_hists:
                        tensor_count += 1
                        continue
                    f.seek(info['offset'])
                    raw = f.read(info['size'])
                    if self.native_bits == 16:
                        if len(raw) % 2 == 0:
                            indices = np.frombuffer(raw, dtype=np.uint16)
                            counts = np.bincount(indices, minlength=65536).astype(np.uint64)
                            category_hists[ttype] += counts
                            unique_count = int(np.count_nonzero(counts))
                    else:
                        indices = np.frombuffer(raw, dtype=np.uint8)
                        counts = np.bincount(indices, minlength=256).astype(np.uint64)
                        category_hists[ttype] += counts
                        unique_count = int(np.count_nonzero(counts))
                    self.tensor_info[name]['unique_count'] = unique_count
                    del raw, indices, counts
                    min_bits = int(np.ceil(np.log2(unique_count))) if unique_count > 1 else 1
                    tensor_count += 1
                    elapsed = time.time() - start_time
                    print(f"    [{tensor_count:5d}/{total_tensors_p1}] {ttype:<12s} "
                          f"[{unique_count:6d} uniq / {min_bits:2d}-bit min]  "
                          f"{name}{_rss()}  ({elapsed:.1f}s)",
                          flush=True)

        # Build codebooks from histograms
        print("\n  Building optimal codebooks from global distribution...")
        self.codebooks = {}
        
        for ttype, hist in category_hists.items():
            total_samples = int(hist.sum())
            if total_samples == 0: continue
            
            # Find unique values and their frequencies
            nonzero_idx = np.where(hist > 0)[0]
            if self.native_bits == 16:
                nonzero_idx = nonzero_idx.astype(np.uint16)
            else:
                nonzero_idx = nonzero_idx.astype(np.uint8)
                
            unique_count = len(nonzero_idx)
            
            # Convert indices back to float values
            if self.native_bits == 16:
                unique_values = (nonzero_idx.astype(np.uint32) << 16).view(np.float32)
            else:
                unique_values = nonzero_idx.astype(np.float32) # For FP8, indices are proxies for now
            
            # FILTER OUT NaNs from codebook candidates
            nan_mask = np.isnan(unique_values)
            if nan_mask.any():
                print(f"    [WARN] Filtering {nan_mask.sum()} NaNs from {ttype} codebook candidates")
                unique_values = unique_values[~nan_mask]
                nonzero_idx = nonzero_idx[~nan_mask]
                unique_count = len(unique_values)
            
            frequencies = hist[nonzero_idx].astype(np.float32)
            val_min, val_max = float(unique_values.min()), float(unique_values.max())
            
            k = self._get_codebook_size(ttype)
            
            # Scale global codebook up to cover all unique values for lossless/balanced.
            # A 256-entry global codebook can't serve as a shared fallback when the data
            # has 7k+ unique values — per-tensor codebooks are needed anyway, and we want
            # the global codebook quality report to reflect whether it's actually usable.
            if self.compression_mode in ('lossless', 'balanced') and unique_count < 32768:
                if unique_count > k:
                    k = unique_count

            # Calculate Coverage %
            coverage = 100.0
            if unique_count > k:
                # Estimate coverage: sum of frequencies of top-k values
                top_k_idx = np.argsort(frequencies)[-k:]
                coverage = (frequencies[top_k_idx].sum() / total_samples) * 100
            
            # REPORTING (User requested format: samples -> uniques -> codebook (loss))
            if k == 0:
                loss_status = "exact"
            elif unique_count <= k:
                loss_status = "0% loss"
            else:
                loss_status = f"{100.0 - coverage:.2f}% loss"

            print(f"    {ttype:<12}: {total_samples:,} samples -> {unique_count:,} unique -> {k}-entry ({loss_status})")

            # Coverage curve: show % of value occurrences covered at each bit depth.
            # Helps reason about quality/size tradeoff for lossy compression targets.
            if unique_count > 1 and k > 0:
                sorted_freqs = np.sort(frequencies)[::-1]  # descending by frequency
                cumulative = np.cumsum(sorted_freqs)
                curve_parts = []
                for cbits in range(2, int(np.ceil(np.log2(unique_count))) + 1):
                    ck = min(2 ** cbits, unique_count)
                    cov = float(cumulative[ck - 1]) / total_samples * 100
                    truly_lossless = int(cumulative[ck - 1]) >= int(total_samples)
                    if truly_lossless:
                        curve_parts.append(f"{cbits}b=lossless")
                        break
                    else:
                        curve_parts.append(f"{cbits}b={cov:.1f}%")
                if curve_parts:
                    # Break the list into lines of ~6 entries each for readability
                    chunk = 6
                    lines = [curve_parts[i:i+chunk] for i in range(0, len(curve_parts), chunk)]
                    print(f"      coverage: {', '.join(lines[0])}")
                    for line in lines[1:]:
                        print(f"               {', '.join(line)}")
            
            if k == 0:
                continue
                
            if unique_count <= k:
                # Lossless - just use all unique values
                self.codebooks[ttype] = np.sort(unique_values)
            else:
                # Build codebook using weighted sampling for kmeans
                sample_size = min(200000, total_samples)
                probs = frequencies / frequencies.sum()
                sample = np.random.choice(unique_values, size=sample_size, p=probs)
                self.codebooks[ttype] = kmeans_1d(sample, k, max_iters=15)
            
            gc.collect()
        
        print(f"  Built {len(self.codebooks)} global codebooks")

    @staticmethod
    def _safe_parallel_workers(headroom_gb: float = 4.0, per_worker_gb: float = 1.5) -> int:
        """Return how many parallel compression workers RAM can safely support."""
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
            workers = max(1, int((avail_gb - headroom_gb) / per_worker_gb))
            # Cap at cpu_count to avoid over-subscribing
            from multiprocessing import cpu_count
            return min(workers, cpu_count())
        except Exception:
            return 1

    def _stream_compress_and_save(self):
        """PASS 2: Strictly serial streaming compression.

        Each tensor is loaded, compressed, written to disk, and freed before the next
        one starts.  This keeps peak RAM at roughly one tensor's worth of working memory
        regardless of model size.  No executor, no in-flight queue — one at a time.
        """
        import gc
        from compressor import _save_cached_tensor

        print(f"  Starting streaming compression (serial, one tensor at a time)...")

        # Ensure directories exist
        tensors_dir = self.cache_dir / "tensors"
        tensors_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_weights = {}  # Keep empty to save RAM

        # Group tensors by file for efficient sequential I/O
        files_to_tensors = {}
        for name, info in self.tensor_info.items():
            fname = info['file']
            if fname not in files_to_tensors:
                files_to_tensors[fname] = []
            files_to_tensors[fname].append(name)

        total_tensors = len(self.tensor_info)
        compressed_count = 0
        start_time = time.time()

        total_original_bytes = 0
        total_compressed_bytes = 0

        for file_idx, (fname, tensor_names) in enumerate(files_to_tensors.items(), 1):
            st_file = self.model_path / fname
            print(f"\n  File {file_idx}/{len(files_to_tensors)}: {fname} ({len(tensor_names)} tensors)")

            for name in tensor_names:
                info = self.tensor_info[name]
                n_params = int(np.prod(info['shape']))
                uniq = info.get('unique_count', '?')
                print(f"    [{compressed_count+1:5d}/{total_tensors}] compressing  "
                      f"{info['type']:<12s} {n_params:>12,}  {name}{_rss()}",
                      flush=True)

                # Process one tensor: load → compress → save → free
                name_r, result, strategy, unique, mse, snr = _compress_adaptive_worker(
                    name, str(st_file), info['offset'], info['size'],
                    info['shape'], info['dtype'],
                    name, self.compression_mode, self.codebooks,
                    self.mse_threshold, self.native_bits,
                    unique_count=info.get('unique_count'),
                    target_bits=self.target_bits,
                )

                safe_name = name_r.replace('.', '_').replace('/', '_')
                npz_file = tensors_dir / f"{safe_name}.npz"
                _save_cached_tensor(npz_file, name_r, result)
                compressed_count += 1

                orig_size = np.prod(result['shape']) * 2
                comp_size = 0
                bits_used = result.get('bits', 16)
                if result['mode'] == 'exact':
                    comp_size = result['data'].nbytes
                    bits_str = 'exact'
                elif result['mode'] == 'direct_codebook':
                    comp_size = calculate_packed_size(np.prod(result['shape']), bits_used)
                    if 'codebook' in result:
                        comp_size += result['codebook'].nbytes
                    bits_str = f'{bits_used}-bit'
                elif result['mode'] == 'linear_quant':
                    comp_size = calculate_packed_size(np.prod(result['shape']), bits_used)
                    bits_str = f'{bits_used}-bit (linear)'
                else:
                    bits_str = result['mode']

                total_original_bytes += orig_size
                total_compressed_bytes += comp_size
                self.strategy_stats[result['mode']] = self.strategy_stats.get(result['mode'], 0) + 1
                if snr < 99.0:  # exclude exact/lossless (reported as 100 dB)
                    self._snr_values.append(snr)

                del result  # free compressed data immediately after saving

                current_ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
                saved_gb = (total_original_bytes - total_compressed_bytes) / 1e9
                elapsed = time.time() - start_time
                print(f"    [{compressed_count:5d}/{total_tensors}] done         "
                      f"[{bits_str:>14s}] snr={snr:5.1f}dB ratio={current_ratio:.2f}x "
                      f"saved={saved_gb:.2f}GB{_rss()}  ({elapsed:.1f}s)",
                      flush=True)

            gc.collect()
        
        # Save final metadata
        print("\n  Saving compression metadata...")
        self._save_codebooks_and_metadata()
        
        total_time = time.time() - start_time
        print(f"  Compression complete: {compressed_count} tensors in {total_time:.1f}s")
        
        snr_vals = self._snr_values
        snr_summary = {}
        if snr_vals:
            snr_summary = {
                'snr_target_db':  self.snr_threshold_db,
                'snr_actual_min': round(min(snr_vals), 2),
                'snr_actual_mean': round(sum(snr_vals) / len(snr_vals), 2),
                'snr_actual_max': round(max(snr_vals), 2),
                'snr_n_tensors':  len(snr_vals),
            }

        metadata = {
            'compression_method': 'adaptive_streaming',
            'original_size_gb': total_original_bytes / 1e9,
            'compressed_size_gb': total_compressed_bytes / 1e9,
            'tensor_count': compressed_count,
            'compression_time_s': total_time,
            'strategy_stats': getattr(self, 'strategy_stats', {}),
            'config': getattr(self, 'config', {}),
            **snr_summary,
        }

        return self._loaded_weights, metadata

    def _compress_single_tensor(self, data: np.ndarray, name: str, ttype: str) -> dict:
        """Compress single tensor without parallel processing"""
        flat = data.flatten()
        
        # Track strategy usage
        if not hasattr(self, 'strategy_stats'):
            self.strategy_stats = {'exact': 0, 'codebook': 0, 'q8_codebook': 0}
        
        # Force exact for small/critical tensors
        if (flat.size < 10000 or 
            'norm' in name.lower() or 
            'layernorm' in name.lower() or
            'ln_' in name.lower() or
            'lm_head' in name.lower() or
            'embed_tokens' in name.lower() or
            'gate.weight' in name.lower() or  # MoE router gates
            ttype in ['layernorm', 'router', 'embedding']):
            self.strategy_stats['exact'] += 1
            return {
                'mode': 'exact',
                'data': float32_to_bfloat16(flat),
                'shape': data.shape
            }
        
        # Try codebook compression
        if ttype in self.codebooks:
            try:
                cb = self.codebooks[ttype]
                indices = assign_to_codebook(flat, cb)
                
                # Quality check
                reconstructed = cb[indices]
                mse = np.mean((flat - reconstructed) ** 2)
                
                if mse <= self.mse_threshold:
                    self.strategy_stats['codebook'] += 1
                    return {
                        'mode': 'codebook',
                        'type': ttype,
                        'indices': indices,
                        'shape': data.shape
                    }
            except Exception:
                pass
        
        # Fallback to exact
        self.strategy_stats['exact'] += 1
        return {
            'mode': 'exact',
            'data': float32_to_bfloat16(flat),
            'shape': data.shape
        }

    def _get_codebook_size(self, tensor_type: str) -> int:
        config = {
            'embedding': self.embedding_size,
            'attention': self.mlp_size,
            'mlp_ffn': self.mlp_size,
            'moe_expert': self.mlp_size,
            'shared_expert': 0, # Keep shared expert exact
            'router': 0,
            'layernorm': 0,
            'ssm_core': 0,
        }
        return config.get(tensor_type, self.mlp_size)

    def _save_codebooks_and_metadata(self):
        """Save codebooks and metadata for analysis-only mode"""
        # Ensure cache directory exists
        cache_dir = self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save codebooks
        codebooks_dir = cache_dir / "codebooks"
        codebooks_dir.mkdir(exist_ok=True)
        
        for ttype, codebook in self.codebooks.items():
            codebook_path = codebooks_dir / f"{ttype}_codebook.npy"
            np.save(codebook_path, codebook)
        
        # Save metadata
        snr_vals = getattr(self, '_snr_values', [])
        snr_summary = {}
        if snr_vals:
            snr_summary = {
                'snr_target_db':   getattr(self, 'snr_threshold_db', None),
                'snr_actual_min':  round(min(snr_vals), 2),
                'snr_actual_mean': round(sum(snr_vals) / len(snr_vals), 2),
                'snr_actual_max':  round(max(snr_vals), 2),
                'snr_n_tensors':   len(snr_vals),
            }
        metadata = {
            'compression_method': 'adaptive_v2',
            'compression_mode': self.compression_mode,
            'tensor_count': len(self.tensor_info),
            'codebook_sizes': {ttype: len(cb) for ttype, cb in self.codebooks.items()},
            'model_hash': self.model_hash,
            'tensor_info': self.tensor_info,
            'accuracy_stats': getattr(self, 'accuracy_stats', {}),
            'config': getattr(self, 'config', {}),
            **snr_summary,
        }
        
        metadata_path = cache_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved {len(self.codebooks)} codebooks and metadata to {cache_dir}")

    def _load_global_codebooks(self) -> Dict[str, torch.Tensor]:
        """Load global codebooks from the codebooks/ directory."""
        global_codebooks = {}
        
        codebooks_dir = self.cache_dir / "codebooks"
        if not codebooks_dir.exists():
            return global_codebooks
        
        # Load each .npy file as a codebook
        for codebook_file in codebooks_dir.glob("*.npy"):
            # Parse codebook type from filename (e.g., "embedding_codebook.npy" -> "embedding")
            codebook_name = codebook_file.stem.replace('_codebook', '')
            
            try:
                # Load numpy array and convert to PyTorch tensor
                codebook_np = np.load(codebook_file)
                codebook_tensor = torch.from_numpy(codebook_np).to(torch.bfloat16)
                global_codebooks[codebook_name] = codebook_tensor
                
            except Exception as e:
                print(f"Warning: Failed to load {codebook_file}: {e}")

        return global_codebooks


# ---------------------------------------------------------------------------
# Flat .idx exporter (for mmap inference — Phase 8)
# ---------------------------------------------------------------------------

def export_flat_idx(model_dir, tensors_subdir="codebook/tensors", verbose=True):
    """
    Export packed indices from .npz files to flat .idx binary files.

    Each .npz in tensors_subdir contains an 'indices' key (uint8 array).
    This function writes a matching .idx file with the raw bytes.

    .idx files can be memory-mapped directly into the process address space
    via MmappedPackedBuffer, letting the OS page data from disk on demand
    instead of loading all packed indices into RAM.

    Args:
        model_dir:       Path to model directory (e.g. ~/workspace/model/Qwen3.5-0.8B)
        tensors_subdir:  Subdirectory under model_dir containing .npz files
        verbose:         Print progress

    Returns:
        int: number of .idx files written
    """
    model_dir = Path(model_dir)
    tensors_dir = model_dir / tensors_subdir

    if not tensors_dir.exists():
        raise FileNotFoundError(f"Tensors directory not found: {tensors_dir}")

    npz_files = sorted(tensors_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {tensors_dir}")

    written = 0
    for npz_path in npz_files:
        idx_path = npz_path.with_suffix('.idx')
        if idx_path.exists():
            continue  # already exported

        try:
            data = np.load(npz_path)
        except Exception as e:
            if verbose:
                print(f"  [skip] {npz_path.name}: load failed ({e})")
            continue

        if 'indices' not in data:
            if verbose:
                print(f"  [skip] {npz_path.name}: no 'indices' key")
            continue

        packed = data['indices']
        if packed.dtype != np.uint8:
            packed = packed.astype(np.uint8)

        packed.tofile(str(idx_path))
        written += 1

        if verbose:
            size_kb = len(packed) / 1024
            print(f"  {npz_path.stem}: {size_kb:.0f} KB → {idx_path.name}")

    if verbose:
        print(f"export_flat_idx: wrote {written} .idx files to {tensors_dir}")

    return written

