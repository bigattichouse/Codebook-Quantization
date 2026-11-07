#!/usr/bin/env python3
"""
Bit-depth sweep for the gemma-3 KV session (kv_session.ggsn, 18 layers x 411 x 256 f16).

Three distinct quantization cases are measured and plotted:

  A. CODEBOOK (this repo's approach)
     From the raw f16: per-channel clip at p99.9 (robust) or per-channel max (no clip),
     uniform quantization to 2**bits levels per channel, one f16 scale per 411-value
     channel.  Arbitrary bit depths 4..16 (9, 10, 12 included).
     Storage: raw = bits + scale overhead; huffman = Shannon(indices) + overhead.

  B. BUILT-IN Q8_0 (llama.cpp ggml)
     From the raw f16, exactly as llama.cpp stores a Q8_0 tensor:
     blocks of 32 values, per-block amax, d = amax/127, ql = clamp(round(x/d), -127, 127),
     stored qh = ql + 128 (u8), d as f16.  33 bytes / 32 values = 8.25 b/val raw.
     No clipping: the scale is the block max, so the worst value in every block is exact.

  C. BUILT-IN Q8_0 THEN HUFFMAN ON ITS INDICES
     Same blocks, but the 8-bit index stream is entropy-coded.  Zero extra distortion
     relative to B; the only question is the index entropy (H).  Total = H + 16/32 ovh.
     This is what a future llama.cpp codebook/Huffman backend would actually do.

Storage overhead: one f16 (16 b) scale per 411-value channel -> 16/411 = 0.0389 b/val
for the codebook; 16/32 = 0.5 b/val for Q8_0 blocks.

Writes chart.html (SNR vs effective bits/value; codebook clipped + no-clip curves,
built-in Q8_0 raw and Q8_0+Huffman points, codebook Q8/Q9 points, f16 baseline).
"""
import numpy as np, io, contextlib

# --- pull the parsed per-layer tensors from parse_session.py (runs it quietly) ---
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = {}
    exec(open("/home/bigattichouse/workspace/Codebook-Quantization/kv_cache_experiment/parse_session.py").read(), g)

K = np.concatenate([g['K_by_layer'][il] for il in sorted(g['K_by_layer'])]).astype(np.float32)
V = np.concatenate([g['V_by_layer'][il] for il in sorted(g['V_by_layer'])]).astype(np.float32)

CH = 411            # per-channel = per 411-row column (n_pos), 256 channels per layer
OVH = 16.0 / CH     # one f16 scale per 411-value channel  = 0.0389 b/val
QK8 = 32            # llama.cpp Q8_0 block size
OVH8 = 16.0 / QK8   # one f16 scale per 32-value block      = 0.5 b/val

def shannon_bits(vals):
    v, c = np.unique(vals, return_counts=True)
    c = c.astype(np.float64); c /= c.sum()
    return float(-(c * np.log2(c)).sum())

def snr_of(x, xq):
    sig = np.sqrt(np.mean(x ** 2))
    noise = np.sqrt(np.mean((x - xq) ** 2))
    return 20 * np.log10(sig / noise) if noise > 0 else 999.0

# ---------- A. codebook from f16 ----------
def codebook(x, bits, clip_q=0.999):
    """per-channel clip at quantile, uniform quant to 2**bits levels.
    returns (snr_db, H_bits_per_val, clipped_frac, q)"""
    c = np.quantile(np.abs(x), clip_q, axis=0)
    c[c == 0] = 1e-6
    nlv = 1 << bits
    q = np.clip(np.round((x + c[None, :]) / (2.0 * c[None, :]) * (nlv - 1)), 0, nlv - 1).astype(np.int64)
    xdq = q / (nlv - 1.0) * (2.0 * c[None, :]) - c[None, :]
    H = shannon_bits(q.ravel())
    clip = float(np.mean(np.abs(x) > c[None, :]))
    return snr_of(x, xdq), H, clip, q

# ---------- B/C. exact llama.cpp Q8_0 ----------
def q8_0(x):
    """exact built-in Q8_0: 32-value blocks, f16 per-block scale.
    returns (snr_db, H_bits_per_val_of_indices, qh_u8)"""
    n = x.size
    xb = x.reshape(-1, QK8)
    amax = np.abs(xb).max(axis=1, keepdims=True)          # abs max (no clip)
    d = np.where(amax > 0, amax / 127.0, 1e-9)
    ql = np.clip(np.rint((xb / d).astype(np.float32)), -127, 127).astype(np.int32)
    qh = (ql + 128).astype(np.uint8)                       # stored form, 0..255
    dq = qh.astype(np.float32) - 128.0
    xq = (dq * d).reshape(x.shape)
    H = shannon_bits(qh.ravel())
    return snr_of(x, xq), H, qh

raw_bytes = (K.size + V.size) * 2
nb = K.size + V.size
BITS = (4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16)
print(f"raw f16 KV = {raw_bytes:,} bytes; codebook scale overhead = {OVH:.4f} b/val (1 f16 / 411); q8_0 block overhead = {OVH8:.3f} b/val (1 f16 / 32)")

def table(clip_q, label):
    print(f"\n### A. codebook: {label} ###")
    hdr = f"{'bits':>4} | {'SNR K':>6} {'SNR V':>6} | {'raw b/val':>9} {'size MB':>7} {'x':>5} | {'huff b/val':>10} {'size MB':>7} {'x':>5} | {'clip%':>5}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for bits in BITS:
        sk, Hk, ck, qk = codebook(K, bits, clip_q)
        sv, Hv, cv, qv = codebook(V, bits, clip_q)
        Hpool = shannon_bits(np.concatenate([qk.ravel(), qv.ravel()]))
        raw_bpv = bits + OVH
        huf_bpv = Hpool + OVH
        raw_sz = raw_bpv / 8 * nb
        huf_sz = huf_bpv / 8 * nb
        rows.append((bits, sk, sv, raw_bpv, raw_sz, huf_bpv, huf_sz, ck))
        print(f"{bits:>4} | {sk:6.1f} {sv:6.1f} | "
              f"{raw_bpv:9.2f} {raw_sz/1e6:7.2f} {raw_bytes/raw_sz:5.2f} | "
              f"{huf_bpv:10.2f} {huf_sz/1e6:7.2f} {raw_bytes/huf_sz:5.2f} | {ck*100:5.2f}")
    return rows

rows_clip   = table(0.999, "per-channel clip @ p99.9  (robust; outlier-tolerant)")
rows_noclip = table(1.0,   "per-channel max scale, NO clip  (best SNR; one outlier per channel eats resolution)")

print()
print("=== B/C. built-in Q8_0 (llama.cpp ggml, blocks of 32, f16 scale) ===")
sk8, Hk8, qk8 = q8_0(K)
sv8, Hv8, qv8 = q8_0(V)
H8 = shannon_bits(np.concatenate([qk8.ravel(), qv8.ravel()]))
snr8 = min(sk8, sv8)
raw8_bpv = 8.0 + OVH8
raw8_sz = raw8_bpv / 8 * nb
huf8_bpv = H8 + OVH8
huf8_sz = huf8_bpv / 8 * nb
print(f"B. built-in Q8_0 raw:        8.00 + {OVH8:.3f} ovh = {raw8_bpv:.2f} b/val -> {raw8_sz/1e6:.2f} MB ({raw_bytes/raw8_sz:.2f}x), SNR K {sk8:.1f} / V {sv8:.1f} dB")
print(f"C. built-in Q8_0 + Huffman:  {H8:.2f} + {OVH8:.3f} ovh = {huf8_bpv:.2f} b/val -> {huf8_sz/1e6:.2f} MB ({raw_bytes/huf8_sz:.2f}x), SNR unchanged (K {sk8:.1f} / V {sv8:.1f} dB)")
print(f"   index entropy savings: {8.0 - H8:.2f} bits/val = {(8.0 - H8)/8*100:.1f}% of the raw index width")
v8, c8 = np.unique(np.concatenate([qk8.ravel(), qv8.ravel()]), return_counts=True)
top = np.sort(c8 / c8.sum())[::-1]
for f in (0.10, 0.25, 0.50):
    print(f"   top {f*100:.0f}% of distinct indices carry {top[:int(f*len(top))].sum()*100:.1f}% of values")

# direct head-to-head at the 8-bit family
r8c = [r for r in rows_clip if r[0] == 8][0]
r8n = [r for r in rows_noclip if r[0] == 8][0]
print()
print("=== head-to-head, 8-bit family (SNR dB, Huffman b/val, MB, x) ===")
print(f"A codebook Q8 clipped  : SNR {min(r8c[1],r8c[2]):.1f}  huff {r8c[5]:.2f} b/val  {r8c[6]/1e6:.2f} MB  x{raw_bytes/r8c[6]:.2f}")
print(f"A codebook Q8 no-clip  : SNR {min(r8n[1],r8n[2]):.1f}  huff {r8n[5]:.2f} b/val  {r8n[6]/1e6:.2f} MB  x{raw_bytes/r8n[6]:.2f}")
print(f"B built-in Q8_0 raw    : SNR {snr8:.1f}  raw  {raw8_bpv:.2f} b/val  {raw8_sz/1e6:.2f} MB  x{raw_bytes/raw8_sz:.2f}")
print(f"C built-in Q8_0 +Huff  : SNR {snr8:.1f}  huff {huf8_bpv:.2f} b/val  {huf8_sz/1e6:.2f} MB  x{raw_bytes/huf8_sz:.2f}")

# ================= SVG chart =================
def svg_plot(out):
    W, Hh, ml, mr, mt, mb = 1000, 640, 90, 30, 50, 70
    pw, ph = W - ml - mr, Hh - mt - mb
    x0, x1 = 0.0, 16.6
    y0, y1 = 0.0, 45.0
    def X(v): return ml + (v - x0) / (x1 - x0) * pw
    def Y(v): return mt + ph - (v - y0) / (y1 - y0) * ph
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="monospace" font-size="14">']
    s.append(f'<rect width="{W}" height="{Hh}" fill="white"/>')
    s.append(f'<text x="{W/2}" y="28" text-anchor="middle" font-size="19">KV-cache quantization: SNR vs effective bits/value (18 layers x 411 x 256)</text>')
    import math
    def ticks(lo, hi, n=8):
        step = (hi - lo) / n
        mag = 10 ** math.floor(math.log10(step))
        for m in (1, 2, 2.5, 5, 10):
            if step / mag <= m:
                step = m * mag; break
        t = math.ceil(lo / step) * step
        out = []
        while t <= hi + 1e-9:
            out.append(round(t, 6)); t += step
        return out
    for t in ticks(x0, x1):
        s.append(f'<line x1="{X(t):.1f}" y1="{mt}" x2="{X(t):.1f}" y2="{mt+ph}" stroke="#e3e3e3"/>')
        s.append(f'<text x="{X(t):.1f}" y="{mt+ph+20}" text-anchor="middle">{t:g}</text>')
    for t in ticks(y0, y1):
        s.append(f'<line x1="{ml}" y1="{Y(t):.1f}" x2="{ml+pw}" y2="{Y(t):.1f}" stroke="#e3e3e3"/>')
        s.append(f'<text x="{ml-8}" y="{Y(t)+4:.1f}" text-anchor="end">{t:g}</text>')
    s.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="black"/>')
    s.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt}" stroke="black"/>')
    s.append(f'<text x="{ml+pw/2}" y="{Hh-16}" text-anchor="middle">effective bits per value (after Huffman, incl. per-channel/per-block scale overhead)</text>')
    s.append(f'<text x="24" y="{mt+ph/2}" text-anchor="middle" transform="rotate(-90 24 {mt+ph/2})">SNR (dB)</text>')
    # f16 baseline
    s.append(f'<line x1="{X(16):.1f}" y1="{mt}" x2="{X(16):.1f}" y2="{mt+ph}" stroke="#999" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{X(16)-4:.1f}" y="{mt+18}" text-anchor="end" fill="#666">raw f16 = 16 b/val (exact)</text>')

    def curve(rows, color, marker, dy, max_bits=None):
        keep = [r for r in rows if max_bits is None or r[0] <= max_bits]
        pts = " ".join(f"{X(r[5]):.1f},{Y(min(max(r[1],r[2]), y1)):.1f}" for r in keep)
        s.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>')
        for r in keep:
            x, y = X(r[5]), Y(min(max(r[1], r[2]), y1))
            if marker == 'c':
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            else:
                s.append(f'<rect x="{x-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" fill="{color}"/>')
            s.append(f'<text x="{x:.1f}" y="{y+dy:.1f}" text-anchor="middle" fill="{color}" font-size="12">{r[0]}</text>')

    def point(bpv, snr, color, shape, label, dx, dy, anchor="middle", fs=12.5):
        x, y = X(bpv), Y(snr)
        if shape == 'diamond':
            s.append(f'<path d="M {x:.1f} {y-7:.1f} L {x+7:.1f} {y:.1f} L {x:.1f} {y+7:.1f} L {x-7:.1f} {y:.1f} Z" fill="{color}"/>')
        elif shape == 'c':
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}"/>')
        else:
            s.append(f'<rect x="{x-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" fill="{color}"/>')
        s.append(f'<text x="{x+dx:.1f}" y="{y+dy:.1f}" text-anchor="{anchor}" fill="{color}" font-size="{fs}">{label}</text>')

    # A: clipped codebook curve, all bits (max SNR 27.3 -> well in axis)
    curve(rows_clip,   "#1f6fd1", 'c', -9)
    # A: no-clip codebook curve, only while it stays on-axis (<= 9 bits -> 39 dB);
    #    annotate that it keeps climbing off the top.
    curve(rows_noclip, "#d1491f", 's', 18, max_bits=9)
    r9n = [r for r in rows_noclip if r[0] == 9][0]
    s.append(f'<text x="{X(r9n[5])+8:.1f}" y="{Y(min(r9n[1],r9n[2]))+22:.1f}" fill="#d1491f" font-size="12">-> climbs to {max(rows_noclip[-1][1],rows_noclip[-1][2]):.0f} dB @ 16 b</text>')

    # the three-case decision points (B/C labels stacked above/below to avoid overlap)
    r9c = [r for r in rows_clip if r[0] == 9][0]
    point(raw8_bpv, snr8, "#2e9e4f", 'c', f"B  built-in Q8_0: {snr8:.1f} dB / {raw8_bpv:.2f} b", -12, -14, anchor="end")
    point(huf8_bpv, snr8, "#0b5c2e", 'diamond', f"C  Q8_0 + Huffman: {snr8:.1f} dB / {huf8_bpv:.2f} b", 12, 22, anchor="start")
    point(r8c[5], min(r8c[1], r8c[2]), "#1f6fd1", 'c', f"A  codebook Q8 clip: {min(r8c[1],r8c[2]):.0f} dB / {r8c[5]:.2f} b", -12, 20, anchor="end")
    point(r9c[5], min(r9c[1], r9c[2]), "#1f6fd1", 'c', f"A  codebook Q9 clip: {min(r9c[1],r9c[2]):.0f} dB / {r9c[5]:.2f} b", 12, 20, anchor="start")

    # legend: bottom-right, away from the top-clustered B/C and the left curves
    lx, ly = W - mr - 430, mt + ph - 150
    s.append(f'<rect x="{lx}" y="{ly}" width="430" height="128" fill="white" stroke="#ccc" opacity="0.92"/>')
    s.append(f'<circle cx="{lx+16}" cy="{ly+20}" r="5" fill="#1f6fd1"/>')
    s.append(f'<text x="{lx+30}" y="{ly+24}">A: codebook from f16 (per-channel scale, Huffman)</text>')
    s.append(f'<circle cx="{lx+16}" cy="{ly+42}" r="5" fill="#1f6fd1" opacity="0.55"/>')
    s.append(f'<text x="{lx+30}" y="{ly+46}" fill="#666">clipped p99.9 (robust) / red squares = no-clip (max)</text>')
    s.append(f'<circle cx="{lx+16}" cy="{ly+64}" r="5" fill="#2e9e4f"/>')
    s.append(f'<text x="{lx+30}" y="{ly+68}">B: built-in Q8_0 raw (ggml blocks of 32, f16 scale)</text>')
    s.append(f'<path d="M {lx+16} {ly+78} L {lx+21} {ly+83} L {lx+16} {ly+88} L {lx+11} {ly+83} Z" fill="#0b5c2e"/>')
    s.append(f'<text x="{lx+30}" y="{ly+88}">C: built-in Q8_0 + Huffman on its 8-bit indices</text>')
    s.append(f'<text x="{lx+30}" y="{ly+108}" fill="#666">x = bits/value AFTER entropy coding; y = min(K,V) SNR</text>')
    s.append(f'<text x="{lx+30}" y="{ly+122}" fill="#666">raw f16 = 16 b/val (right dashed line, exact)</text>')
    s.append('</svg>')
    open(out, "w").write("<!doctype html><meta charset=utf-8><title>KV codebook SNR vs bits</title>\n" + "\n".join(s))
    print(f"\nchart written: {out}")

svg_plot("/home/bigattichouse/workspace/Codebook-Quantization/kv_cache_experiment/chart.html")
