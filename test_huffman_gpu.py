"""
Diagnostic test for GPU Huffman decode kernel.

Run from the repo root:
    cd /path/to/Codebook-Quantization
    python test_huffman_gpu.py

Tests a tiny known stream end-to-end, printing each intermediate value so the
exact failure point is visible.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch

from huffman_codebook import (
    huffman_encode_indices,
    huffman_decode_indices,
    _canonical_codes,
    build_gpu_lut,
)

# ─── 1. Build a tiny known Huffman stream ────────────────────────────────────

# 8 symbols from a small range (max index = 7)
TEST_INDICES = np.array([3, 1, 0, 3, 2, 1, 0, 3], dtype=np.uint16)

# Also include a second row so we exercise per-row bit offsets
TEST_INDICES2 = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint16)

ALL_INDICES = np.concatenate([TEST_INDICES, TEST_INDICES2])
M, K = 2, 8

print("=" * 60)
print("1. Encoding")
print("=" * 60)
result = huffman_encode_indices(ALL_INDICES, shape=(M, K))

huff_stream  = result['huff_stream']
huff_lengths = result['huff_lengths']
huff_n       = int(result['huff_n'][0])
row_bs       = result['huff_row_bit_starts']
lut_sym      = result['huff_lut_sym']
lut_len      = result['huff_lut_len']
sl_fc        = result['huff_sl_first_code']
sl_bo        = result['huff_sl_base_offset']
sl_sym       = result['huff_sl_sym']

print(f"  huff_n       = {huff_n}")
print(f"  stream len   = {len(huff_stream)} bytes")
print(f"  row_bit_starts = {row_bs}")
print(f"  huff_lengths = {huff_lengths}")

codes = _canonical_codes(huff_lengths)
print("  Canonical codes:")
for sym, (code, L) in sorted(codes.items(), key=lambda x: (x[1][1], x[0])):
    print(f"    sym={sym}  code={bin(code)[2:].zfill(L)}  len={L}")

# ─── 2. CPU round-trip ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. CPU decode round-trip")
print("=" * 60)
cpu_decoded = huffman_decode_indices(huff_stream, huff_lengths, huff_n)
match = np.array_equal(cpu_decoded, ALL_INDICES)
print(f"  CPU decoded : {cpu_decoded}")
print(f"  Expected    : {ALL_INDICES}")
print(f"  Match       : {match}")
if not match:
    print("  *** CPU decode FAILED — check huffman_codebook.py ***")
    sys.exit(1)

# ─── 3. LUT spot-check ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. LUT spot-check (first key from stream)")
print("=" * 60)
# Compute what key the GPU should see at bit_pos=0
b0 = int(huff_stream[0])
b1 = int(huff_stream[1]) if len(huff_stream) > 1 else 0
b2 = int(huff_stream[2]) if len(huff_stream) > 2 else 0
w24 = (b0 << 16) | (b1 << 8) | b2
expected_key = (w24 >> (24 - 0 - 12)) & 0xFFF  # bit_off=0, n=12
print(f"  stream bytes[0:3] = {b0:02x} {b1:02x} {b2:02x}")
print(f"  w24 = {w24:#08x}")
print(f"  expected key (first 12 bits) = {expected_key}")
print(f"  lut_sym[{expected_key}] = {lut_sym[expected_key]}  lut_len[{expected_key}] = {lut_len[expected_key]}")
print(f"  first expected symbol = {ALL_INDICES[0]}")
assert lut_sym[expected_key] == ALL_INDICES[0], \
    f"LUT mismatch: lut_sym[{expected_key}]={lut_sym[expected_key]}, expected {ALL_INDICES[0]}"
print("  LUT check PASSED")

# ─── 4. GPU decode ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("4. GPU decode")
print("=" * 60)

if not torch.cuda.is_available():
    print("  No CUDA device — skipping GPU test")
    sys.exit(0)

from gpu_huffman_functions import (
    _load_huffman_extension, _np_to_i32_tensor, _ROCmHuffmanWrapper,
)

ext = _load_huffman_extension()
if ext is None:
    print("  GPU extension failed to compile — skipping GPU test")
    sys.exit(1)

device = torch.device('cuda')
is_rocm_ext = isinstance(ext, _ROCmHuffmanWrapper)
print(f"  Extension type: {'ROCm ctypes (_ROCmHuffmanWrapper)' if is_rocm_ext else 'NVIDIA load_inline'}")

if is_rocm_ext:
    # ROCm ctypes path: native HIP kernel reads raw uint8 bytes + int64 row_bit_starts
    stream_np = np.asarray(huff_stream, dtype=np.uint8)
    rbs_dtype = np.int64
else:
    # NVIDIA load_inline path: kernel reads stream as int32 words + int32 row_bit_starts
    raw_len = len(huff_stream)
    pad_len = ((raw_len + 3) // 4) * 4
    padded_u8 = np.zeros(pad_len, dtype=np.uint8)
    padded_u8[:raw_len] = huff_stream
    stream_np = padded_u8.view(np.int32)
    rbs_dtype = np.int32

stream_gpu = torch.from_numpy(stream_np.copy()).to(device)
torch.cuda.synchronize()

print(f"  stream_gpu.device        = {stream_gpu.device}")
print(f"  stream_gpu.dtype         = {stream_gpu.dtype}")
print(f"  stream_gpu.is_contiguous = {stream_gpu.is_contiguous()}")
print(f"  stream_gpu.data_ptr()    = {hex(stream_gpu.data_ptr())}")

rbs_gpu     = torch.from_numpy(np.asarray(row_bs, dtype=rbs_dtype)).to(device)
lut_sym_gpu = _np_to_i32_tensor(lut_sym, device)
lut_len_gpu = _np_to_i32_tensor(lut_len, device)
sl_fc_gpu   = _np_to_i32_tensor(sl_fc, device)
sl_bo_gpu   = _np_to_i32_tensor(sl_bo, device)
sl_sym_gpu  = _np_to_i32_tensor(sl_sym, device)

max_len = int(huff_lengths.max())
print(f"  max_code_len = {max_len}")
print(f"  M={M}  K={K}")
print(f"  stream_gpu dtype  = {stream_gpu.dtype}, shape = {stream_gpu.shape}")
print(f"  rbs_gpu dtype     = {rbs_gpu.dtype}, shape = {rbs_gpu.shape}")
print(f"  lut_sym_gpu dtype = {lut_sym_gpu.dtype}, shape = {lut_sym_gpu.shape}")
print(f"  sl_fc_gpu dtype   = {sl_fc_gpu.dtype}, shape = {sl_fc_gpu.shape}")

print(f"  lut_sym_gpu[{expected_key}] = {lut_sym_gpu[expected_key].item()} (expected {ALL_INDICES[0]})")
print(f"  lut_len_gpu[{expected_key}] = {lut_len_gpu[expected_key].item()}")

# ── Diagnostic: verify stream pointer is readable from kernel ────────────────
if not is_rocm_ext and hasattr(ext, 'diag_stream_read'):
    print()
    print("  [diag] Testing direct stream read from kernel...")
    diag_out = ext.diag_stream_read(stream_gpu, stream_gpu.numel())
    torch.cuda.synchronize()
    diag_np = diag_out.cpu().numpy()
    stream_cpu = stream_gpu.cpu().numpy()
    print(f"  [diag] stream_gpu values (Python): {[hex(int(x) & 0xFFFFFFFF) for x in stream_cpu]}")
    print(f"  [diag] kernel read values:         {[hex(int(x) & 0xFFFFFFFF) for x in diag_np]}")
    stream_match = all(diag_np[i] == int(stream_cpu[i]) for i in range(len(diag_np)))
    print(f"  [diag] Match: {stream_match}")
else:
    print("  [diag] ROCm ctypes path — skipping diag_stream_read (not applicable)")

# Run the kernel
out = ext.huffman_decode_to_i32(
    stream_gpu, rbs_gpu, lut_sym_gpu, lut_len_gpu,
    sl_fc_gpu, sl_bo_gpu, sl_sym_gpu,
    M, K, max_len,
)
torch.cuda.synchronize()

out_np = out.cpu().numpy().reshape(M, K)
print()
print(f"  GPU output row 0 : {out_np[0]}")
print(f"  Expected row 0   : {ALL_INDICES[:K]}")
print(f"  GPU output row 1 : {out_np[1]}")
print(f"  Expected row 1   : {ALL_INDICES[K:]}")

gpu_match = np.array_equal(out_np.ravel(), ALL_INDICES)
print()
if gpu_match:
    print("  *** GPU decode PASSED ***")
else:
    print("  *** GPU decode FAILED ***")
    print()
    # Extra diagnostics: what does lut_sym[0] contain?
    print(f"  lut_sym[0]  = {lut_sym_gpu[0].item()}  (HUFF_NO_SYM = -1)")
    print(f"  lut_len[0]  = {lut_len_gpu[0].item()}")
    print(f"  rbs_gpu values = {rbs_gpu.cpu().numpy()}")
    print()
    print("  If GPU output is all zeros and lut_sym[0]==0 with lut_len[0]>0:")
    print("  → huff_read_bits is returning 0 for all reads")
    print("  If GPU output is all zeros and lut_sym[0]==-1:")
    print("  → slow path is not matching any code, outputting default 0")
    print("  Check that the kernel recompiled (v5 suffix in build dir).")

# ─── 5. Real model smoke test (optional) ─────────────────────────────────────
print()
print("=" * 60)
print("5. Larger random test (1024 symbols, 8 rows × 128 cols)")
print("=" * 60)

rng = np.random.default_rng(42)
# Skewed distribution like real LLM weights (symbol 0 is most common)
probs = 1.0 / (np.arange(1, 257) ** 1.5)
probs /= probs.sum()
big_idx = rng.choice(256, size=8 * 128, p=probs).astype(np.uint16)
big_M, big_K = 8, 128

big_result = huffman_encode_indices(big_idx, shape=(big_M, big_K))

# CPU decode
cpu_big = huffman_decode_indices(
    big_result['huff_stream'], big_result['huff_lengths'], int(big_result['huff_n'][0])
)
assert np.array_equal(cpu_big, big_idx), "CPU big test failed"
print("  CPU big test: PASSED")

# GPU decode
if is_rocm_ext:
    snp2 = np.asarray(big_result['huff_stream'], dtype=np.uint8)
    rb2_dtype = np.int64
else:
    raw_len2 = len(big_result['huff_stream'])
    pad_len2 = ((raw_len2 + 3) // 4) * 4
    pu2 = np.zeros(pad_len2, dtype=np.uint8)
    pu2[:raw_len2] = big_result['huff_stream']
    snp2 = pu2.view(np.int32)
    rb2_dtype = np.int32
sg2 = torch.from_numpy(snp2.copy()).to(device)
rb2 = torch.from_numpy(np.asarray(big_result['huff_row_bit_starts'], dtype=rb2_dtype)).to(device)
ls2 = _np_to_i32_tensor(big_result['huff_lut_sym'], device)
ll2 = _np_to_i32_tensor(big_result['huff_lut_len'], device)
sf2 = _np_to_i32_tensor(big_result['huff_sl_first_code'], device)
sb2 = _np_to_i32_tensor(big_result['huff_sl_base_offset'], device)
ss2 = _np_to_i32_tensor(big_result['huff_sl_sym'], device)
ml2 = int(big_result['huff_lengths'].max())

out2 = ext.huffman_decode_to_i32(sg2, rb2, ls2, ll2, sf2, sb2, ss2, big_M, big_K, ml2)
torch.cuda.synchronize()
gpu_big = out2.cpu().numpy()
big_match = np.array_equal(gpu_big, big_idx)
print(f"  GPU big test: {'PASSED' if big_match else 'FAILED'}")
if not big_match:
    diff = np.where(gpu_big != big_idx)[0]
    print(f"  First 5 mismatches at positions {diff[:5]}:")
    for pos in diff[:5]:
        print(f"    [{pos}] gpu={gpu_big[pos]}  expected={big_idx[pos]}")

print()
print("Done.")
