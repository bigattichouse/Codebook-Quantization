# KV-Cache / Session-Attention Compression Feasibility

Target: `gemma-3-270m-it-Q8_0.gguf` (18 layers, n_embd=640, n_head=4,
n_head_kv=1, head_dim=256, sliding_window=512) running in the standalone
CPU-only runner. Session = 28 prompt + 384 generated = 412 tokens, saved via
`llama_state_save_file` to `kv_session.ggsn` (7,587,566 bytes).

Separate experiment from the project's weight Codebook-Quantization / INT8-KV
weight pipeline. Question: is codebook / Huffman compression of session/attention
(KV-cache) data feasible or worthwhile?

## 1. Exact ggsn layout for this build (llama.cpp @ 5851a8e50)

The earlier "parse misalignment" is resolved. The file is:

    [ "ggsn" ][ u32 version=11 ]
    [ u32 n_token=412 ][ 412 x i32 tokens ]
    [ arch string: u32 len + bytes ]              # "gemma3", ends @1670
    ---- mem_attn = llama_kv_cache_iswa::state_write ----
        llama_kv_cache_iswa writes two llama_kv_cache sub-caches back to back:
        (a) kv_base : full-attention layers only  -> 3 layers (5, 11, 17)
        (b) kv_swa  : sliding-window layers       -> 15 layers (all others)
    ---- mem_recr = llama_memory_recurrent::state_write ----
        EMPTY for gemma3 (it is not a recurrent arch): writes cell_count=0 + header

Each llama_kv_cache section:
    [ u32 n_stream=1 ]
    [ u32 cell_count=411 ]
    # meta, per cell: [ i32 pos ][ u32 n_seq_id ][ n_seq_id x i32 ]
    # data:
    [ u32 v_trans=0 ][ u32 n_layer ][ i32 n_rot_k=0 ][ i32 n_rot_v=0 ]
    # per layer:
    [ i32 k_type=1 (f16) ][ u64 k_row_size=512 ][ cell_count x k_row bytes ]
    [ i32 v_type=1 (f16) ][ u64 v_row_size=512 ][ cell_count x v_row bytes ]

Key facts:
  - `n_embd_k_gqa = n_embd_v_gqa = n_head_kv * head_dim = 1 * 256 = 256`
    -> one f16 row = 256 * 2 = 512 bytes per K (and V) per cell.
  - SWA pattern 6, dense_first=false => is_swa[il] = (il%6 < 5). Full layers
    are il where (il%6)==5 => {5,11,17}. That is why the first data header
    shows n_layer=3 (it is kv_base, not the whole model).
  - `has_cell_ext()` is FALSE for gemma3 (n_pos_per_embd==1, ple_n_heads==0).
  - cell_count = 411 = 412 tokens - 1 (the final generated token's KV is not
    stored yet at save time; the 412 tokens include the last sampled token).

Byte accounting (parser lands exactly on EOF):
    header (magic..arch)         1,670  B
    kv_base   (3 full layers)  1,267,620 B   (3 layers x 411 x 2 x 512 + meta)
    kv_swa    (15 SWA layers)  6,318,276 B   (15 layers x 411 x 2 x 512 + meta)
    mem_recr                           0 B
    TOTAL                          7,587,566 B   == file size  (exact)

Per-layer payload = 411 cells * (K 512 B + V 512 B) = 411 * 1024 = 420,864 B.

## 2. Data statistics (all 18 layers, 411 cells, 256-dim f16)

    K: 1,893,888 values   V: 1,893,888 values   (raw f16 = 3,787,776 B each)
    raw KV data total    = 7,575,552 B  (+ ~12 KB of pos/seq metadata)

  K:  min=-90.4  max=+132.5  std=3.11   abs median=0.75  p95=5.1  p99=12.9  p999=31.5
  V:  min=-99.2  max=+112.4  std=3.69   abs median=0.80  p95=5.3  p99=15.5  p999=39.8

  Distribution is heavy-tailed: median |x| ~0.75 but a few per-channel
  outliers reach ~100+. Fraction exactly zero is ~0; |x|<1e-2 is ~1.2%.
  K and V have comparable scale and shape; nothing is degenerate.

  Per layer, each K (and V) tensor uses only ~16k-23k of the 65536 possible
  f16 bit patterns (values are bounded per layer, e.g. L0 K in [-20,20]).

## 3. Compression ceilings

A) LOSSLESS (ideal Huffman/arithmetic on the stored bytes, per-layer):
     Per-layer f16-pattern entropy: 13.4 - 14.0 bits/value
     Weighted average (K): 13.65 bits/value
     -> ~0.85x  (saves ~15% of raw, 0 quality loss)
   The pooled K+V entropy is 13.96 bits/val; per-layer is the realistic figure
   because each layer's tensor is stored separately with its own scale.

   NOTE on prior confusion: a naive f32->u16 cast (or a shared cross-layer
   codebook) gives meaningless numbers. Per-layer f16-pattern entropy is the
   correct, achievable lossless ceiling.

B) LOSSY CODEBOOK (per-channel, clip at per-channel p99.9; storage =
   codeword-entropy + one f16 scale per 411-val channel = 0.039 b/val):
     8-bit:  K 5.70 b/val (SNR 26.0 dB, clip 0.11%)   V 5.41 b/val (26.6 dB)
             -> ~0.35x of raw
     6-bit:  K 3.70 b/val (SNR 21.7 dB)               V 3.41 b/val (21.7 dB)
             -> ~0.22x of raw
     4-bit:  K 1.91 b/val (SNR 9.7 dB)                V 1.73 b/val (9.2 dB)
             -> ~0.11x of raw
   (This is the regime of the project's existing INT8-KV weight path, applied
   here to the session KV data. 8-bit with clipping is the sweet spot: ~3x
   smaller with high SNR. 4-bit is aggressive.)

## 4. Bit-depth sweep (4..16, arbitrary) + the three quantization cases

`bit_sweep.py` re-runs the parse, then measures three distinct cases at every
bit depth and writes `chart.html` (pure-SVG, no plotting lib).

  A. CODEBOOK from f16 (this repo's approach). Per-channel scale, one f16 per
     411-val channel, uniform quant to 2**bits, optional clip. Arbitrary bit
     depths (9/10/12 included). Storage = bits + 0.039 ovh (raw) or
     Shannon(indices) + ovh (Huffman).
  B. BUILT-IN Q8_0 (llama.cpp ggml). From raw f16, exactly as ggml stores a
     Q8_0 tensor: 32-value blocks, per-block amax, d=amax/127, q=clamp(round,
     -127,127), stored qh=q+128, d as f16. 33 B / 32 val = 8.50 b/val raw.
     NO clip: the scale is the block max, so the worst value per block is exact.
  C. BUILT-IN Q8_0 + HUFFMAN on its 8-bit indices. Same blocks, index stream
     entropy-coded. Zero extra distortion vs B; only question is index entropy.

Head-to-head, 8-bit family (SNR = min(K,V), b/val after Huffman):

    A codebook Q8 clip :  26.0 dB   5.53 b/val   2.62 MB   x2.89
    A codebook Q8 max  :  33.0 dB   5.06 b/val   2.40 MB   x3.16
    B built-in Q8_0    :  41.8 dB   8.50 b/val (raw) 4.02 MB  x1.88
    C Q8_0 + Huffman   :  41.8 dB   7.85 b/val   3.72 MB   x2.04

Key findings:
  * SIGNAL: built-in Q8_0 wins by a landslide (41.8 vs 26 dB for clipped codebook).
    Not the bit depth - it is scale granularity + NO clipping. Q8_0 uses a
    per-32-value amax scale so each outlier is isolated to its own 32-val block
    and the block max is exact. The codebook's robust variant clips at p99.9 and
    shares one scale across 411 values, so clipping error + one outlier per
    channel dominate. Even the codebook's own no-clip per-channel variant only
    reaches 33 dB.
  * SIZE: codebook wins. Q8_0's block-level quantization spreads indices over
    all 256 levels (top 10% of indices = only 32.8% of values), so its index
    entropy is high and Huffman shaves just 0.65 bits/val. The codebook's
    per-channel indices are far more concentrated (top 10% = 80.8%), so Huffman
    shaves 2.5 bits/val. The codebook's size advantage is entirely from its
    index distribution, not from Huffman being "better."
  * Q9 (clipped codebook) is not worth it: +0.4 dB for +1 MB over Q8 (same
    plateau). Clipped SNR saturates at ~26.5/27.3 dB from 8 bits up because
    clipping error dominates; to climb you must loosen/remove the clip, which
    trades outlier robustness for resolution.
  * No-clip codebook climbs ~6 dB/bit (33 @ 8b, 39 @ 9b, 45 @ 10b, 57 @ 12b,
    81 @ 16b) - but one outlier per 411-val channel eats resolution, so it is
    fragile on real, drifting KV streams.

Verdict on the three cases (for this PoC):
  * Optimize quality (and the model's weights already run Q8_0, so there is no
    custom decode cost): use B, built-in Q8_0. 41.8 dB at 4.02 MB. C adds
    Huffman for only 0.3 MB, no SNR gain - marginal.
  * Optimize minimum size at acceptable quality: A, codebook Q8 clipped -
    2.62 MB but 26 dB (~16 dB below Q8_0).
  * The "third line" (C) confirms Q8_0's indices are not very compressible;
    the codebook is smaller purely because its per-channel indices cluster.

## 5. Verdict: feasible but not compelling for THIS use case

Feasible: yes. KV-cache values are a well-behaved, bounded, per-layer-scaled
float distribution. Per-channel codebook quantization is straightforward and
gives ~3x at 8-bit (26 dB SNR) or ~9x at 4-bit. Lossless Huffman alone gives
~15%.

Worthwhile: weak, for three reasons specific to session/attention data:

  1. It is already stored at 16-bit f16 (not 32-bit). The lossless ceiling is
     only ~0.85x because f16 is a dense, near-uniform entropy source per layer
     -- there is little redundancy to squeeze without losing bits. Unlike the
     weight pipeline (Q8_0/INT4 weights with lots of structural redundancy),
     KV values carry real signal, not a compressible quantization pattern.

  2. KV-cache is a working buffer, not an artifact. It is rewritten every
     step and only the current window (SWA=512) matters for most layers.
     Compress/decompress on the hot path costs CPU that the cache exists to
     save. The only place it clearly pays is *persistence*: shrinking the
     .ggsn session file (here 7.6 MB) for checkpoint/restore or transfer.
     For a 412-token session that's ~5 MB saved at 8-bit -- nice, not critical.

  3. The big win (8/4-bit) is lossy and needs per-channel scales + clip
     calibration, i.e. it is exactly the INT8-KV machinery the project already
     has for weights. Reusing it for the session cache is possible but the
     quality/throughput trade-off must be validated against actual generation,
     which is out of scope for this PoC.

Recommendation:
  - Do NOT pursue a standalone codebook/Huffman scheme for the live KV cache.
    The lossless gain (~15%) does not justify the encode/decode overhead on
    the hot path, and the lossy gain is already available via the existing
    INT8-KV path if the user opts into quantized KV.
  - IF the goal is purely to shrink persisted .ggsn session files, an 8-bit
    per-channel codebook (or plain INT8 with per-channel scale + p99.9 clip)
    is a clean ~3x win with ~0 quality risk (26 dB). That is the only
    scenario where this is clearly worth building. If quality is the priority
    instead, built-in Q8_0 already delivers 41.8 dB at 4.02 MB with no custom
    decode - the codebook's job is strictly to go smaller at a quality cost.

Repro:
    python3 kv_cache_experiment/parse_session.py
    # exact-EOF parse + section A/B/C analysis; prints byte accounting and
    # all entropy/SNR figures above.
    python3 kv_cache_experiment/bit_sweep.py
    # bit-depth sweep 4..16 + the three cases (A codebook / B built-in Q8_0 /
    # C Q8_0+Huffman); writes chart.html (open in a browser, pure SVG).
