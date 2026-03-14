import json
import numpy as np
from pathlib import Path
import sys
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from compressor import classify_tensor, load_raw_tensor_data

def deep_analyze_model(model_path):
    model_path = Path(model_path)
    
    # Smart file discovery to avoid counting multiple model versions
    # First try direct (non-recursive) search for main model files
    st_files = sorted(model_path.glob("*.safetensors"))
    
    # If no files found in root, or if we find very few files, try recursive
    if not st_files or len(st_files) < 5:
        all_files = sorted(model_path.rglob("*.safetensors"))
        if all_files:
            # Prefer main directory over subdirectories
            main_files = [f for f in all_files if f.parent == model_path]
            if main_files:
                st_files = main_files
            else:
                # If no files in main dir, use largest subdirectory
                from collections import defaultdict
                by_dir = defaultdict(list)
                for f in all_files:
                    by_dir[f.parent].append(f)
                # Choose directory with most files
                st_files = max(by_dir.values(), key=len)
    
    if not st_files:
        return None
        
    # Detect native bits per parameter from config
    native_bits = 16
    config_file = model_path / "config.json"
    if not config_file.exists():
        # Try checking subdirs for config
        configs = list(model_path.rglob("config.json"))
        if configs: config_file = configs[0]

    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                q_config = config.get("quantization_config", {})
                q_method = q_config.get("quant_method", "").lower()
                if q_method in ["fp8", "int8"]:
                    native_bits = 8
                elif q_method in ["mxfp4", "fp4", "q4_0", "q4_k"]:
                    native_bits = 4
                elif config.get("dtype") in ["float32"]:
                    native_bits = 32
        except: pass

    total_params = 0
    # Use a set to track processed tensor names to avoid double-counting shards/versions
    processed_tensors = set()
    
    cat_stats = {
        'embedding': {'params': 0, 'uniques': 0},
        'attention': {'params': 0, 'uniques': 0},
        'mlp_ffn': {'params': 0, 'uniques': 0},
        'moe_expert': {'params': 0, 'uniques': 0},
        'router': {'params': 0, 'uniques': 0},
        'other': {'params': 0, 'uniques': 0}
    }
    
    tensors_sampled = {k: 0 for k in cat_stats}
    for st_file in st_files:
        # If we already have 60GB+ of weights from one folder, skip other folders
        # to avoid counting multiple model versions (root vs original vs metal)
        # We prefer the root or largest folder
        with open(st_file, 'rb') as f:
            header_size = int.from_bytes(f.read(8), 'little')
            header = json.loads(f.read(header_size).decode('utf-8'))
            base_offset = f.tell()
            
            for name, info in header.items():
                if name == '__metadata__': continue
                if name in processed_tensors: continue
                processed_tensors.add(name)
                
                stored_elements = 1
                for dim in info['shape']: stored_elements *= dim
                
                params = stored_elements
                if '_blocks' in name.lower() and native_bits == 4:
                    params = stored_elements * 2
                
                ttype = classify_tensor(name)
                if ttype not in cat_stats: ttype = 'other'
                
                cat_stats[ttype]['params'] += params
                total_params += params
                
                # Sample for uniqueness
                if tensors_sampled[ttype] < 3 and params > 100000 and 'scale' not in name.lower() and 'bias' not in name.lower():
                    try:
                        f.seek(base_offset + info['data_offsets'][0])
                        if native_bits == 4:
                            raw = f.read(min(stored_elements, 100000))
                            data = np.frombuffer(raw, dtype=np.uint8)
                            u = len(np.unique(np.concatenate([data >> 4, data & 0x0F])))
                        elif native_bits == 8:
                            raw = f.read(min(stored_elements, 100000))
                            u = len(np.unique(np.frombuffer(raw, dtype=np.uint8)))
                        else:
                            raw = f.read(min(stored_elements * 2, 200000))
                            if len(raw) % 2 == 0:
                                u = len(np.unique(np.frombuffer(raw, dtype=np.uint16)))
                        
                        cat_stats[ttype]['uniques'] = max(cat_stats[ttype]['uniques'], u)
                        tensors_sampled[ttype] += 1
                    except: pass

    # Calculate theoretical sizes
    orig_gb = total_params * native_bits / 8 / 1e9
    lossless_gb = 0
    codebook_q8_gb = 0
    codebook_q4_gb = 0
    
    for ttype, stats in cat_stats.items():
        p = stats['params']
        u = stats['uniques'] if stats['uniques'] > 0 else (2**native_bits)

        # Default fallback
        ll_bits, q8_bits, q4_bits = native_bits, native_bits, native_bits

        if ttype != 'other':
            # ESTIMATE LOSSLESS BITS
            ll_bits = int(np.ceil(np.log2(u))) if u > 0 else native_bits

            # CEILING: If using >50% of possible values, assume standard entropy
            if u > (2**native_bits * 0.5):
                ll_bits = native_bits

            # MODE-SPECIFIC BITS
            is_sens = ttype in ['router']
            q8_bits = ll_bits if is_sens else 8
            q4_bits = ll_bits if is_sens else 4

            # Correction: Codebook 8b can't be larger than native 8b
            if native_bits == 8:
                q8_bits = 8

        lossless_gb += (p * ll_bits / 8) / 1e9
        codebook_q8_gb += (p * q8_bits / 8) / 1e9
        codebook_q4_gb += (p * q4_bits / 8) / 1e9

    return {
        'name': model_path.name,
        'params': total_params / 1e9,
        'orig': orig_gb,
        'll': lossless_gb,
        'cq8': codebook_q8_gb,
        'cq4': codebook_q4_gb,
        'other_gb': (cat_stats['other']['params'] * native_bits / 8) / 1e9,
        'native_bits': native_bits
    }

def main():
    model_paths = [
        "Kimi-Linear-REAP-35B-A3B-Instruct",
        "Qwen3.5-35B-A3B",
        "Ministral-3-14B-Instruct-2512",
        "Qwen3-0.6B",
        "Qwen3-Coder-30B-A3B-Instruct",
        "Devstral-Small-2-24B-Instruct-2512",
        "phi-4-mini-instruct",
        "Qwen3-1.7B",
        "Qwen3-Coder-Next",
        "gemma-3-270M-it",
        "gpt-oss-120b"
    ]
    
    base_dir = Path("/home/bigattichouse/workspace/model")
    
    # Collect all results first for sorting by size
    all_results = []
    for m_name in model_paths:
        p = base_dir / m_name
        if not p.exists(): continue
        res = deep_analyze_model(p)
        if res:
            # Add missed GB calculation (typically small or zero for our models)
            missed_gb = max(0, res['other_gb'])  # Use the 'other' category as missed
            all_results.append({
                'name': res['name'],
                'params': res['params'],
                'original': res['orig'],
                'lossless': res['ll'], 
                'codebook_q8': res['cq8'],
                'codebook_q4': res['cq4'],
                'missed': missed_gb
            })
    
    # Sort by model name for consistent output (case-insensitive)
    all_results.sort(key=lambda x: x['name'].lower())

    print(f"\n{'='*145}")
    print(f"COMPRESSION ANALYSIS: ORIGINAL VS. LOSSLESS VS. CODEBOOK MODES")
    print(f"{'='*145}")
    print(f"{'Model Name':<35} | {'Params':>6} | {'Original':>9} | {'Lossless':>9} | {'CodebkQ8':>9} | {'CodebkQ4':>9} | {'Missed':>8}")
    print(f"{'-'*145}")
    
    for res in all_results:
        print(f"{res['name']:<35} | {res['params']:>5.1f}B | {res['original']:>8.1f}GB | {res['lossless']:>8.1f}GB | {res['codebook_q8']:>8.1f}GB | {res['codebook_q4']:>8.1f}GB | {res['missed']:>7.1f}GB")

    print(f"{'='*145}")
    print("Column Definitions:")
    print(" - Original: Current disk storage size (native format)")
    print(" - Lossless: Perfect reconstruction with optimal bit-width compression")
    print(" - CodebkQ8: 8-bit adaptive codebook compression (balanced accuracy/size)")
    print(" - CodebkQ4: 4-bit adaptive codebook compression (maximum size reduction)")
    print(" - Missed: Parameters not covered by compression heuristics")
    print(f"{'='*145}\n")

if __name__ == "__main__":
    main()
