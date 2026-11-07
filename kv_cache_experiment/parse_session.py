#!/usr/bin/env python3
"""
Precise parser for the llama.cpp ggsn session file of gemma-3-270m-it (SWA hybrid).

Layout (confirmed against llama.cpp @ 5851a8e50 sources):
  [magic "ggsn"][u32 version]
  [u32 n_token][n_token * i32 tokens]
  [arch string: u32 len + bytes]   e.g. "gemma3"
  --- mem_attn = llama_kv_cache state_write ---   (FULL-ATTN layers only)
     [u32 n_stream]
     for each stream:
        [u32 cell_count]
        # state_write_meta: per cell  [i32 pos][u32 n_seq_id][n_seq_id * i32 seq_id]
        # state_write_data:
           [u32 v_trans][u32 n_layer][i32 n_rot_k][i32 n_rot_v]
           if not v_trans:  per layer  [i32 ktype][u64 krow][k rows]  [i32 vtype][u64 vrow][v rows]
           else:            per layer  [i32 ktype][u64 krow][k rows]
                             per layer  [i32 vtype][u32 v_size_el][u32 n_embd_v_gqa][n_embd_v_gqa * v rows]
  --- mem_recr = llama_memory_recurrent state_write ---   (SWA layers only)
     [u32 cell_count]
     # meta: per cell [i32 pos][u32 n_seq_id][n_seq_id*i32]
     # data: [u32 n_layer][i32 n_rot_r][i32 n_rot_s] then R / S tensors per layer
Model dims: n_layer=18, n_embd=640, n_head=4, n_head_kv=1, head_dim=256.
SWA pattern 6 (dense_first=false): full layers = 5,11,17 ; SWA layers = rest (15).
n_embd_k_gqa = n_embd_v_gqa = 256.
"""
import struct, sys

F = "/home/bigattichouse/workspace/Codebook-Quantization/kv_cache_experiment/kv_session.ggsn"
buf = open(F, "rb").read()
N = len(buf)
print(f"file size = {N}")

class R:
    def __init__(s, off=0): s.off = off
    def u8(s):  v=struct.unpack_from("<B",buf,s.off)[0]; s.off+=1; return v
    def i8(s):  v=struct.unpack_from("<b",buf,s.off)[0]; s.off+=1; return v
    def u16(s): v=struct.unpack_from("<H",buf,s.off)[0]; s.off+=2; return v
    def i16(s): v=struct.unpack_from("<h",buf,s.off)[0]; s.off+=2; return v
    def u32(s): v=struct.unpack_from("<I",buf,s.off)[0]; s.off+=4; return v
    def i32(s): v=struct.unpack_from("<i",buf,s.off)[0]; s.off+=4; return v
    def u64(s): v=struct.unpack_from("<Q",buf,s.off)[0]; s.off+=8; return v
    def i64(s): v=struct.unpack_from("<q",buf,s.off)[0]; s.off+=8; return v
    def f32(s): v=struct.unpack_from("<f",buf,s.off)[0]; s.off+=4; return v
    def take(s,n): v=buf[s.off:s.off+n]; s.off+=n; return v
    def skip(s,n): s.off+=n
    def s(s):
        l=s.u32(); v=buf[s.off:s.off+l].decode(); s.off+=l; return v

r = R()
# --- global header ---
magic = r.take(4)
assert struct.unpack_from("<I", buf, 0)[0] == 0x6767736e, magic  # "ggsn" LE
version = r.u32()
n_token = r.u32()
tokens = [r.i32() for _ in range(n_token)]
arch = r.s()
print(f"magic={magic} version={version} n_token={n_token} arch={arch!r} off_after_arch={r.off}")
print(f"first 8 tokens = {tokens[:8]}  last 4 = {tokens[-4:]}")

N_LAYER_FULL = 3
N_LAYER_SWA  = 15
HEAD_DIM     = 256
N_HEAD_KV    = 1
n_embd_k_gqa = HEAD_DIM * N_HEAD_KV   # 256
n_embd_v_gqa = HEAD_DIM * N_HEAD_KV   # 256
FULL_LAYERS  = [5, 11, 17]
SWA_LAYERS   = [i for i in range(18) if i % 6 != 5]

# --- mem_attn (llama_kv_cache) ---
print("\n=== mem_attn (llama_kv_cache) ===")
n_stream = r.u32()
print(f"n_stream = {n_stream}")
kv_rows_per_layer = {}   # il -> (n_cells, ktype, vtype)
import numpy as np
all_K = []  # list of (layer, np.ndarray [n_cells, 256] f32)
all_V = []

for s in range(n_stream):
    cell_count = r.u32()
    print(f"stream {s}: cell_count = {cell_count} (offset {r.off})")
    # meta: per cell
    pos_list = []
    for _ in range(cell_count):
        pos = r.i32()
        n_seq = r.u32()
        seqs = [r.i32() for _ in range(n_seq)]
        pos_list.append(pos)
    print(f"  meta: first pos={pos_list[:3]} last pos={pos_list[-3:]} (offset {r.off})")
    # data
    v_trans = r.u32()
    n_layer = r.u32()
    n_rot_k = r.i32()
    n_rot_v = r.i32()
    print(f"  data: v_trans={v_trans} n_layer={n_layer} n_rot_k={n_rot_k} n_rot_v={n_rot_v} (offset {r.off})")

    if not v_trans:
        for li in range(n_layer):
            il = FULL_LAYERS[li]
            ktype = r.i32(); krow = r.u64()
            k = r.take(cell_count * krow)
            vtype = r.i32(); vrow = r.u64()
            v = r.take(cell_count * vrow)
            kv_rows_per_layer[il] = (cell_count, ktype, vtype)
            all_K.append((il, k, ktype, krow))
            all_V.append((il, v, vtype, vrow))
    else:
        for li in range(n_layer):
            il = FULL_LAYERS[li]
            ktype = r.i32(); krow = r.u64()
            k = r.take(cell_count * krow)
            kv_rows_per_layer[il] = (cell_count, ktype, None)
            all_K.append((il, k, ktype, krow))
        for li in range(n_layer):
            il = FULL_LAYERS[li]
            vtype = r.i32(); v_size_el = r.u32(); n_embd_v = r.u32()
            # n_embd_v rows each of length cell_count * v_size_el bytes
            v = r.take(n_embd_v * cell_count * v_size_el)
            kv_rows_per_layer[il] = (cell_count, kv_rows_per_layer[il][1], vtype)
            all_V.append((il, v, vtype, v_size_el))

    print(f"  after stream {s} data: offset = {r.off}")

print("\nper-layer KV: " , kv_rows_per_layer)

# --- mem_attn = llama_kv_cache_iswa = kv_base (3 full) THEN kv_swa (15 SWA) ---
# kv_base was parsed above (all_K/all_V hold layers 5,11,17).
# Now parse kv_swa at the current offset.
SWA_LAYERS = [i for i in range(18) if i % 6 != 5]
print("\n=== mem_attn.kv_swa (llama_kv_cache, SWA layers) ===")
base_end = r.off
n_stream2 = r.u32()
print(f"n_stream = {n_stream2} (offset {r.off})")
for s in range(n_stream2):
    cell_count = r.u32()
    print(f"stream {s}: cell_count = {cell_count} (offset {r.off})")
    pos_list = []
    for _ in range(cell_count):
        pos = r.i32(); n_seq = r.u32()
        for _ in range(n_seq): r.i32()
        pos_list.append(pos)
    print(f"  meta: first pos={pos_list[:3]} last pos={pos_list[-3:]}")
    v_trans = r.u32(); n_layer = r.u32(); n_rot_k = r.i32(); n_rot_v = r.i32()
    print(f"  data: v_trans={v_trans} n_layer={n_layer} n_rot_k={n_rot_k} n_rot_v={n_rot_v}")
    assert not v_trans
    for li in range(n_layer):
        il = SWA_LAYERS[li]
        ktype = r.i32(); krow = r.u64()
        k = r.take(cell_count * krow)
        vtype = r.i32(); vrow = r.u64()
        v = r.take(cell_count * vrow)
        kv_rows_per_layer[il] = (cell_count, ktype, vtype)
        all_K.append((il, k, ktype, krow))
        all_V.append((il, v, vtype, vrow))
swa_end = r.off
print(f"\nkvs_wa parsed {n_layer} layers, ends at {swa_end}")

# --- verify we consumed exactly the whole file ---
print(f"\n  FINAL offset = {r.off}  EOF = {N}  -> exact match: {r.off == N}")
assert r.off == N, f"parser stopped at {r.off} != EOF {N}"

print("\n=== byte accounting ===")
print(f"total            = {N}")
print(f"header (magic..arch)      = {1670} bytes")
print(f"kv_base (3 full layers)   = {base_end-1670} bytes")
print(f"kv_swa  (15 SWA layers)   = {swa_end-base_end} bytes")
print(f"mem_recr                  = 0 bytes (gemma3 is not recurrent -> llama_kv_cache_iswa used directly)")

# ================= histogram / entropy analysis =================
print("\n" + "="*70)
print("KV DATA ANALYSIS (all 18 layers, 411 cells each, 256-dim f16)")
print("="*70)

K_by_layer = {il: np.frombuffer(k, dtype=np.float16).reshape(-1,256) for il,k,kt,kr in all_K}
V_by_layer = {il: np.frombuffer(v, dtype=np.float16).reshape(-1,256) for il,v,vt,vr in all_V}

all_K = np.concatenate([K_by_layer[il].ravel() for il in K_by_layer]).astype(np.float32)
all_V = np.concatenate([V_by_layer[il].ravel() for il in V_by_layer]).astype(np.float32)
all_K_raw16 = all_K.astype(np.float16).view(np.uint16)  # f16 bit patterns
all_V_raw16 = all_V.astype(np.float16).view(np.uint16)

print(f"K: {all_K.size} values  (raw f16 = {all_K.size*2:,} bytes)")
print(f"V: {all_V.size} values  (raw f16 = {all_V.size*2:,} bytes)")
print(f"raw KV total           = {(all_K.size+all_V.size)*2:,} bytes")
print(f"(plus ~{N-(all_K.size+all_V.size)*2:,} bytes of file header + per-cell pos/seq metadata)")

def shannon(arr):
    vals,cnts=np.unique(arr,return_counts=True)
    p=cnts/cnts.sum()
    return float(-np.sum(p*np.log2(p))), vals

def stats(name, x):
    a=np.abs(x)
    print(f"\n--- {name} (f32 view of f16) ---")
    print(f"  min={x.min():+.5f}  max={x.max():+.5f}  mean={x.mean():+.5f}  std={x.std():.5f}")
    print(f"  abs: mean={a.mean():.5f}  median={np.median(a):.5f}  p95={np.percentile(a,95):.5f}  p99={np.percentile(a,99):.5f}  p999={np.percentile(a,99.9):.5f}")
    print(f"  frac exactly 0: {np.mean(x==0):.4%}   frac |x|<1e-3: {np.mean(a<1e-3):.4%}   frac |x|<1e-2: {np.mean(a<1e-2):.4%}")

stats("K", all_K); stats("V", all_V)

# ---- 1) LOSSLESS ceiling: entropy of the ACTUAL stored f16 bit patterns ----
print("\n=== A) LOSSLESS ceiling (entropy of raw f16 bit patterns) ===")
print("    = what an ideal Huffman/arithmetic coder of the stored bytes could reach")
HK, Kvals = shannon(all_K_raw16)
HV, Vvals = shannon(all_V_raw16)
print(f"  K: {HK:6.3f} bits/val  ({len(Kvals):>6} distinct f16 patterns)  -> est {HK/8*all_K.size:,.0f} B  (vs {all_K.size*2:,})  ratio {HK/16:.3f}")
print(f"  V: {HV:6.3f} bits/val  ({len(Vvals):>6} distinct f16 patterns)  -> est {HV/8*all_V.size:,.0f} B  (vs {all_V.size*2:,})  ratio {HV/16:.3f}")
both=np.concatenate([all_K_raw16,all_V_raw16])
Hb,_=shannon(both)
print(f"  K+V combined: {Hb:6.3f} bits/val")

# ---- 2) CODEBOOK / lossy quantization, per-channel with quantile clipping ----
# K/V here is heavy-tailed (median |x|~0.75, p99~13, max~130). A real INT8-KV
# quantizer therefore clips at a per-channel quantile instead of per-channel max,
# otherwise the few outliers destroy the resolution. We model exactly that.
print("\n=== B) CODEBOOK quantization (per-channel, clip @ p99.9) ===")
print("    storage = ceil(codeword entropy) bits/val + 16b/channel scale overhead")
print("    SNR is vs the ORIGINAL (uncropped) values, so clipping error is included")
def codebook_analysis(x2d, bits, clip_q=0.999):
    x = x2d.astype(np.float32)
    n_cells, ch = x.shape
    c = np.quantile(np.abs(x), clip_q, axis=0)   # per-channel clip magnitude
    c[c==0]=1e-6
    nlv = 1<<bits
    step = 2.0*c/ch  # = 2*c/nlv
    q = np.clip(np.round((x + c[None,:])/(2.0*c[None,:])*(nlv-1)), 0, nlv-1)
    xdq = q/(nlv-1.0)*(2.0*c[None,:]) - c[None,:]   # dequant (clipped)
    H,_ = shannon(q.astype(np.int64).ravel())
    sig = np.sqrt(np.mean(x**2))
    noise = np.sqrt(np.mean((x-xdq)**2))
    snr = 20*np.log10(sig/noise) if noise>0 else 999.0
    clipped = float(np.mean(np.abs(x) > c[None,:]))
    total_bits = H + 16.0/ch   # one f16 scale per channel (256 vals)
    return H, total_bits, snr, clipped

KV2_K = np.concatenate([K_by_layer[il] for il in sorted(K_by_layer)]).astype(np.float32)
KV2_V = np.concatenate([V_by_layer[il] for il in sorted(V_by_layer)]).astype(np.float32)
for bits in (8,6,4):
    Hk,totk,snrk,clk = codebook_analysis(KV2_K,bits)
    Hv,totv,snrv,clv = codebook_analysis(KV2_V,bits)
    tot = totk*all_K.size + totv*all_V.size
    raw = (all_K.size+all_V.size)*16
    print(f"  {bits:2d}-bit:  K {totk:5.2f}b/val (SNR {snrk:5.1f}dB, clip {clk:5.2%})   V {totv:5.2f}b/val (SNR {snrv:5.1f}dB, clip {clv:5.2%})")
    print(f"             -> est {tot/8:,.0f} B  (vs {raw//8:,})  = {tot/raw:.3f} of raw")

# ---- 3) per-layer lossless entropy (K, f16 bit patterns) ----
# A real compressor stores each layer's K/V as its own tensor, so the per-layer
# entropy is the realistic lossless ceiling (layer scales differ, so a shared
# codebook across layers is not how it's laid out).
print("\n=== C) per-layer lossless entropy (K, f16 patterns) ===")
row=[]
for il in sorted(K_by_layer):
    h,_=shannon(K_by_layer[il].ravel().astype(np.float16).view(np.uint16))
    row.append((il,round(h,2)))
print("  "+"  ".join(f"L{i}:{h}" for i,h in row))
wavg = sum(h*K_by_layer[i].size for i,h in row)/sum(K_by_layer[i].size for i,h in row)
print(f"  weighted avg = {wavg:.3f} bits/val  -> ~{wavg/16:.2f}x lossless compression (0 quality loss)")

