"""
gguf_codes.py — CPU-only lossless-headroom measurement for already-quantized GGUFs.

GGUF quant blocks store small integer codes (4-bit, 5-bit, 8-bit) plus per-block
scales.  The *dequantized* floats are not a small codebook (every block has its
own fp16 scale), but the integer codes are.  This module samples blocks from
each tensor via mmap (no GPU, no full read), decodes the code symbols, and
measures what per-tensor Huffman coding of those codes would save.  Block
scales/mins are counted raw at their stored size.

Finding on k-quants (2026-11): code histograms are near-identical across a whole
model but nearly flat (Q4_K ≈ 3.86 / 4 bits), because the quantizer fits each
block's scale to fill the code range — lossless headroom is only ~2%.
"""

from __future__ import annotations

import heapq
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Block decoders: raw uint8 blocks [n, block_bytes] -> (symbols, n_symbol_values)
# ---------------------------------------------------------------------------

def _nibbles(qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return qs & 0x0F, qs >> 4


def _dec_q4_k(b):  # 144 B / 256: d, dmin (f16), scales[12], qs[128]
    lo, hi = _nibbles(b[:, 16:144])
    return np.concatenate([lo.ravel(), hi.ravel()]), 16


def _dec_q5_k(b):  # 176 B / 256: d, dmin, scales[12], qh[32], qs[128]
    qh = b[:, 16:48]
    qs = b[:, 48:176].reshape(-1, 4, 32)
    out = []
    for j in range(4):
        lo, hi = _nibbles(qs[:, j])
        out.append(lo | (((qh >> (2 * j)) & 1) << 4))
        out.append(hi | (((qh >> (2 * j + 1)) & 1) << 4))
    return np.concatenate([o.ravel() for o in out]), 32


def _dec_q5_1(b):  # 24 B / 32: d, m (f16), qh (u32), qs[16]
    qh = b[:, 4:8].copy().view("<u4")  # [n,1]
    lo, hi = _nibbles(b[:, 8:24])
    j = np.arange(16, dtype=np.uint32)
    x0 = lo | (((qh >> j) & 1) << 4).astype(np.uint8)
    x1 = hi | (((qh >> (j + 16)) & 1) << 4).astype(np.uint8)
    return np.concatenate([x0.ravel(), x1.ravel()]), 32


def _dec_q4_nibbles(off):  # IQ4_NL / Q4_0: 18 B / 32, Q4_1: 20 B / 32
    def f(b):
        lo, hi = _nibbles(b[:, off:])
        return np.concatenate([lo.ravel(), hi.ravel()]), 16
    return f


def _dec_q8_0(b):  # 34 B / 32: d (f16), qs[32] int8
    return b[:, 2:34].ravel(), 256


# name -> (block_bytes, weights_per_block, code_bits_per_weight, decoder)
DECODERS = {
    "Q4_K":   (144, 256, 4, _dec_q4_k),
    "Q5_K":   (176, 256, 5, _dec_q5_k),
    "Q5_1":   (24,  32,  5, _dec_q5_1),
    "IQ4_NL": (18,  32,  4, _dec_q4_nibbles(2)),
    "Q4_0":   (18,  32,  4, _dec_q4_nibbles(2)),
    "Q4_1":   (20,  32,  4, _dec_q4_nibbles(4)),
    "Q8_0":   (34,  32,  8, _dec_q8_0),
}


# ---------------------------------------------------------------------------
# Entropy / Huffman
# ---------------------------------------------------------------------------

def huffman_bits(counts: np.ndarray) -> float:
    """Average Huffman code length (bits/symbol) for a histogram."""
    nz = [int(c) for c in counts if c > 0]
    if len(nz) <= 1:
        return 1.0
    heap = [(c, i) for i, c in enumerate(nz)]
    heapq.heapify(heap)
    total_cost, tie = 0, len(nz)
    while len(heap) > 1:
        a, _ = heapq.heappop(heap)
        b, _ = heapq.heappop(heap)
        total_cost += a + b  # each merge adds one bit to every symbol beneath it
        heapq.heappush(heap, (a + b, tie))
        tie += 1
    return total_cost / sum(nz)


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def sample_blocks(raw: np.ndarray, block_bytes: int, max_blocks: int,
                  rng: np.random.Generator, chunk: int = 1024) -> np.ndarray:
    """Sample contiguous chunks of blocks spread across the tensor."""
    blocks = raw.reshape(-1, block_bytes)
    n = blocks.shape[0]
    if n <= max_blocks:
        return np.asarray(blocks)
    # few large sequential reads: thousands of small random seeks stall on disk
    n_chunks = max(1, max_blocks // chunk)
    starts = np.sort(rng.integers(0, n - chunk, size=n_chunks))
    return np.concatenate([np.asarray(blocks[s:s + chunk]) for s in starts])


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def gguf_files_for(path: Path) -> list[Path]:
    """GGUF file itself, or the sorted shards in a directory."""
    if path.is_file() and ".gguf" in path.name:
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.gguf"))
    return []


def scan_gguf(path: Path, max_blocks: int = 8192, seed: int = 0,
              progress: bool = True) -> dict:
    """
    Measure Huffman headroom of the quant codes in a GGUF (file or shard dir).

    Returns {"types": {type: stats}, "other_bytes", "raw_bytes", "huff_bytes",
             "roles": {(type, name pattern): hist}}.
    """
    import gguf

    rng = np.random.default_rng(seed)
    types = defaultdict(lambda: {"raw": 0, "huff": 0, "n": 0, "ent": 0.0,
                                 "ent_w": 0, "hist": 0, "pair": None, "quad": None})
    roles = {}
    other_bytes = 0

    for fn in gguf_files_for(path):
        for t in gguf.GGUFReader(str(fn)).tensors:
            tname = t.tensor_type.name
            if tname not in DECODERS:
                other_bytes += int(t.n_bytes)
                continue
            bb, wpb, cbits, dec = DECODERS[tname]
            syms, nvals = dec(sample_blocks(t.data.reshape(-1), bb, max_blocks, rng))
            s64 = syms.astype(np.int64)
            counts = np.bincount(s64, minlength=nvals)

            n_blocks = int(t.n_bytes) // bb
            meta_bytes = int(t.n_bytes) - n_blocks * wpb * cbits / 8
            a = types[tname]
            a["raw"] += int(t.n_bytes)
            a["huff"] += meta_bytes + n_blocks * wpb * huffman_bits(counts) / 8 + nvals * 4
            a["ent"] += entropy_bits(counts) * t.n_elements
            a["ent_w"] += t.n_elements
            a["n"] += 1
            a["hist"] = a["hist"] + counts

            # context tests: adjacent pairs, and 4-code words for 4-bit types
            if nvals <= 32:
                n2 = len(s64) // 2 * 2
                ph = np.bincount(s64[:n2:2] * nvals + s64[1:n2:2], minlength=nvals * nvals)
                a["pair"] = ph if a["pair"] is None else a["pair"] + ph
            if nvals == 16:
                s4 = s64[: len(s64) // 4 * 4].reshape(-1, 4)
                w = (s4[:, 0] << 12) | (s4[:, 1] << 8) | (s4[:, 2] << 4) | s4[:, 3]
                qh = np.bincount(w, minlength=65536)
                a["quad"] = qh if a["quad"] is None else a["quad"] + qh

            key = (tname, re.sub(r"blk\.\d+\.", "blk.N.", t.name))
            roles[key] = roles.get(key, 0) + counts
        if progress:
            print(f"  scanned {fn.name}", file=sys.stderr, flush=True)

    raw = sum(a["raw"] for a in types.values()) + other_bytes
    huff = sum(a["huff"] for a in types.values()) + other_bytes
    return {"types": dict(types), "other_bytes": other_bytes,
            "raw_bytes": raw, "huff_bytes": huff, "roles": roles}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_headroom(res: dict, indent: str = "  ", context: bool = True,
                   roles: bool = False) -> None:
    """Print the per-type table, and optionally context tests / role histograms."""
    types = res["types"]
    print(f"{indent}{'type':<7} {'tensors':>7} {'raw GB':>8} {'code b/w':>8} {'entropy':>8}"
          f" {'huff GB':>8} {'saved':>7}")
    for tname, a in sorted(types.items(), key=lambda kv: -kv[1]["raw"]):
        ent = a["ent"] / a["ent_w"]
        print(f"{indent}{tname:<7} {a['n']:>7} {a['raw']/1e9:>8.2f} {DECODERS[tname][2]:>8}"
              f" {ent:>8.3f} {a['huff']/1e9:>8.2f} {100*(1-a['huff']/a['raw']):>6.1f}%")
    ob = res["other_bytes"] / 1e9
    print(f"{indent}{'other':<7} {'':>7} {ob:>8.2f} {'':>8} {'':>8} {ob:>8.2f}   (kept raw)")
    print(f"{indent}{'TOTAL':<7} {'':>7} {res['raw_bytes']/1e9:>8.2f} {'':>8} {'':>8}"
          f" {res['huff_bytes']/1e9:>8.2f} {100*(1-res['huff_bytes']/res['raw_bytes']):>6.1f}%")

    if context:
        print()
        print(f"{indent}Shared-dictionary / context tests (bits per code, sampled):")
        print(f"{indent}{'type':<7} {'raw':>5} {'per-tensor H':>12} {'shared H':>9}"
              f" {'pair H/2':>9} {'quad H/4':>9} {'distinct quads':>15}")
        for tname, a in sorted(types.items()):
            pr = entropy_bits(a["pair"]) / 2 if a["pair"] is not None else float("nan")
            if a["quad"] is not None:
                qd, nd = entropy_bits(a["quad"]) / 4, f"{int((a['quad'] > 0).sum()):,}/65,536"
            else:
                qd, nd = float("nan"), "-"
            print(f"{indent}{tname:<7} {DECODERS[tname][2]:>5} {a['ent']/a['ent_w']:>12.3f}"
                  f" {entropy_bits(a['hist']):>9.3f} {pr:>9.3f} {qd:>9.3f} {nd:>15}")

    if roles:
        print()
        print(f"{indent}Pooled code histograms by tensor role (fraction of symbols):")
        for (tname, pat), c in sorted(res["roles"].items()):
            p = c / c.sum()
            line = (" ".join(f"{x:.3f}" for x in p) if len(p) <= 32
                    else f"top8 share={np.sort(p)[-8:].sum():.3f}")
            print(f"{indent}{tname:<6} {pat:<36} H={entropy_bits(c):.3f}  {line}")
