#!/usr/bin/env python3
"""
Diagnostic: compare CPU-RAM vs GPU Phase 2 through AdaptiveCodebookLinear.from_compressed.
"""
import sys, os
import numpy as np
import torch
import time

sys.path.insert(0, 'src')

from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding

CACHE = '/home/bigattichouse/workspace/model/Qwen3.5-9B/codebook-lossless-huffman/tensors'

def load_data(name):
    path = os.path.join(CACHE, name + '.npz')
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))

global_codebooks = {}  # per-layer codebooks embedded in data dict

def test_linear_layer(layer_name, T=4):
    data = load_data(layer_name)
    if data is None:
        print(f"SKIP {layer_name}: not found"); return

    enc_val = data.get('encoding', '?')
    encoding = str(enc_val[0]) if hasattr(enc_val, '__len__') and not isinstance(enc_val, str) else str(enc_val)
    if encoding != 'huffman':
        print(f"SKIP {layer_name}: encoding={encoding}"); return

    shape = tuple(int(x) for x in data['shape'])
    M, K = shape

    print(f"\nLayer: {layer_name}  shape={shape}")

    # Load GPU Phase 2
    layer_gpu = AdaptiveCodebookLinear.from_compressed(
        layer_name, data, global_codebooks, use_gpu=True
    )
    p2_active = layer_gpu._gpu_func is not None and 'Huffman' in type(layer_gpu._gpu_func).__name__
    print(f"  GPU P2 active: {p2_active}  (type: {type(layer_gpu._gpu_func).__name__ if layer_gpu._gpu_func else None})")

    # Load CPU-RAM
    layer_cpu = AdaptiveCodebookLinear.from_compressed(
        layer_name, data, global_codebooks, use_gpu=False
    )
    cpu_active = layer_cpu._huff_data is not None
    print(f"  CPU-RAM active: {cpu_active}")

    if not p2_active or not cpu_active:
        print(f"  SKIP: could not set up both paths")
        return

    x = torch.randn(T, K, dtype=torch.float32).cuda()

    t0 = time.time()
    out_gpu = layer_gpu(x)
    torch.cuda.synchronize()
    t_gpu = time.time() - t0

    t0 = time.time()
    out_cpu = layer_cpu(x)
    t_cpu = time.time() - t0

    err = (out_gpu.cpu().float() - out_cpu.cpu().float()).abs()
    print(f"  GPU P2:   range=[{out_gpu.float().min():.4f}, {out_gpu.float().max():.4f}]  time={t_gpu*1000:.1f}ms")
    print(f"  CPU-RAM:  range=[{out_cpu.float().min():.4f}, {out_cpu.float().max():.4f}]  time={t_cpu*1000:.1f}ms")
    print(f"  Max err: {err.max():.6f}  Mean err: {err.mean():.6f}")
    if err.max() < 0.05:
        print("  PASS ✓")
    else:
        print("  FAIL ✗ — paths disagree!")

def test_embedding(layer_name, token_ids=None):
    data = load_data(layer_name)
    if data is None:
        print(f"SKIP {layer_name}: not found"); return

    enc_val = data.get('encoding', '?')
    encoding = str(enc_val[0]) if hasattr(enc_val, '__len__') and not isinstance(enc_val, str) else str(enc_val)
    if encoding != 'huffman':
        print(f"SKIP {layer_name}: encoding={encoding}"); return

    shape = tuple(int(x) for x in data['shape'])
    vocab, H = shape

    if token_ids is None:
        token_ids = [0, 1, 100, 1000, 10000, 50000]

    print(f"\nEmbedding: {layer_name}  shape={shape}  test_tokens={len(token_ids)}")

    # Load CPU-RAM
    layer_cpu = AdaptiveCodebookEmbedding.from_compressed(
        layer_name, data, global_codebooks, use_gpu=False
    )
    cpu_active = layer_cpu._huff_data is not None
    print(f"  CPU-RAM active: {cpu_active}")

    # Load GPU Phase 2
    layer_gpu = AdaptiveCodebookEmbedding.from_compressed(
        layer_name, data, global_codebooks, use_gpu=True
    )
    p2_active = layer_gpu._gpu_func is not None and 'Huffman' in type(layer_gpu._gpu_func).__name__
    print(f"  GPU P2 active: {p2_active}")

    if not cpu_active:
        print("  SKIP: CPU-RAM path not set up")
        return

    ids = torch.tensor(token_ids, dtype=torch.long).cuda()

    # CPU-RAM reference
    t0 = time.time()
    out_cpu = layer_cpu(ids)
    t_cpu = time.time() - t0
    print(f"  CPU-RAM: range=[{out_cpu.float().min():.4f}, {out_cpu.float().max():.4f}]  time={t_cpu*1000:.1f}ms")

    if p2_active:
        print(f"  GPU P2: decoding all {vocab}×{H} symbols per forward call...")
        t0 = time.time()
        out_gpu = layer_gpu(ids)
        torch.cuda.synchronize()
        t_gpu = time.time() - t0
        print(f"  GPU P2: range=[{out_gpu.float().min():.4f}, {out_gpu.float().max():.4f}]  time={t_gpu:.2f}s")

        err = (out_gpu.cpu().float() - out_cpu.cpu().float()).abs()
        print(f"  Max err: {err.max():.6f}  Mean err: {err.mean():.6f}")
        if err.max() < 0.05:
            print("  PASS ✓")
        else:
            print("  FAIL ✗ — paths disagree!")

print("=" * 60)
print("Linear layer tests")
print("=" * 60)
test_linear_layer('model_language_model_layers_0_linear_attn_in_proj_a_weight')
test_linear_layer('model_language_model_layers_0_mlp_gate_proj_weight')

print()
print("=" * 60)
print("Embedding test")
print("=" * 60)
test_embedding('model_language_model_embed_tokens_weight',
               token_ids=[0, 1, 100, 1000, 10000, 50000])

print("\nDone.")
