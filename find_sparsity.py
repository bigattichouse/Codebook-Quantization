import numpy as np
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from bitpack import unpack_any_bits

def find_sparse_layers(tensors_dir):
    files = sorted(Path(tensors_dir).glob("*.npz"))
    print(f"{'File':<60} | {'Zero %':<8} | {'Longest Run':<12} | {'Bits':<4}")
    print("-" * 95)
    
    for f in files:
        try:
            data = np.load(f)
            if 'indices' not in data: continue
            
            bits = int(data['bits'][0])
            original_len = np.prod(data['shape'])
            
            # For speed, only unpack if it's large and we care
            # But let's just do it for all since we want to find THE layer
            unpacked = unpack_any_bits(data['indices'], bits, original_len)
            
            zero_count = np.sum(unpacked == 0)
            zero_pct = zero_count / original_len * 100
            
            # Find longest run
            if len(unpacked) > 0:
                diffs = np.where(unpacked[1:] != unpacked[:-1])[0]
                if len(diffs) == 0:
                    longest = len(unpacked)
                else:
                    runs = np.diff(np.concatenate([[-1], diffs, [len(unpacked)-1]]))
                    longest = np.max(runs)
            else:
                longest = 0
            
            if zero_pct > 1.0 or longest > 10:
                print(f"{f.name:<60} | {zero_pct:<8.2f} | {longest:<12} | {bits:<4}")
                
        except Exception as e:
            # print(f"Error processing {f.name}: {e}")
            continue

if __name__ == "__main__":
    tensors_dir = '/home/bigattichouse/workspace/model/Qwen3.5-9B/codebook-lossless/tensors/'
    find_sparse_layers(tensors_dir)
