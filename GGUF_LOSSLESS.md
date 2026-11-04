# Lossless Recompression of Already-Quantized GGUFs

**Question (2026-11-04):** a good Q4 GGUF is a ready-made starting point, but
it can be larger than VRAM/RAM. Can we shrink it *losslessly* — codebook /
Huffman style, as this project does for fp16 weights — so it fits better
with no additional quality loss? And can we answer that without touching the
GPU?

**Answer: not meaningfully.** Measured headroom is ~2%. The tooling is kept
because it answers the question in minutes for any GGUF.

---

## What gets measured

The dequantized floats in a GGUF are *not* a small codebook: every block
(32 or 256 weights) carries its own fp16 scale, so one tensor holds millions
of distinct float values. Re-running the float codebook on them gains nothing.

The **integer codes inside each block** are a small alphabet (16 symbols for
4-bit, 32 for 5-bit, 256 for Q8_0). Building a histogram of those codes and a
Huffman dictionary from it — the same histogram → dictionary logic as
`src/huffman_codebook.py` — is exactly lossless. Block scales / mins are left
raw and counted at their stored size.

Supported block layouts (`src/gguf_codes.py`): Q4_K, Q5_K, Q5_1, Q4_0, Q4_1,
IQ4_NL, Q8_0. Other types (F32, BF16, …) are counted as raw.

Beyond per-tensor Huffman, the scan also tests whether codes are "more common
across the model than we think":

| Test | Meaning |
|---|---|
| per-tensor H | entropy with one dictionary per tensor |
| shared H | one dictionary per quant type for the whole model |
| pair H/2 | entropy of adjacent code pairs, per code |
| quad H/4 | entropy of 4-code words, per code; plus how many of the 65,536 words occur |

## Results

### Qwen3.8-Flash-Next UD-Q4_K_XL (111 GB, 4 shards, MoE 512 experts / 10 active)

| Type | Tensors | Raw GB | Code bits | Entropy | Huffman GB | Saved |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K (expert gate/up) | 94 | 44.35 | 4 | 3.864 | 43.39 | 2.2% |
| IQ4_NL (`per_layer_token_embd`) | 1 | 28.80 | 4 | 3.914 | 28.58 | 0.8% |
| Q5_1 (expert down) | 43 | 27.05 | 5 | 4.866 | 26.63 | 1.6% |
| Q8_0 (attention, shared experts, embed/output) | 503 | 9.61 | 8 | 7.628 | 9.22 | 4.1% |
| Q5_K | 2 | 1.15 | 5 | 4.859 | 1.13 | 1.8% |
| other (F32/BF16, raw) | | 0.35 | | | 0.35 | |
| **Total** | | **111.32** | | | **109.30** | **1.8%** |

Context tests:

| Type | per-tensor H | shared H | pair H/2 | quad H/4 | distinct quads |
|---|---:|---:|---:|---:|---:|
| Q4_K | 3.864 | 3.864 | 3.861 | 3.856 | 65,536 / 65,536 |
| IQ4_NL | 3.914 | 3.914 | 3.912 | 3.725 | 36,114 / 65,536 |
| Q5_1 | 4.866 | 4.866 | 4.863 | – | – |
| Q8_0 | 7.628 | 7.541 | – | – | – |

### Qwen3.8-27B-DFlash2 Q4_K_M (1.1 GB): 1.9% saved, Q4_K entropy 3.864.
### gemma-3-270m-it Q8_0 (0.3 GB): 4.6% saved, Q8_0 entropy 7.58.

## Why there is so little headroom

- **The histograms are shared model-wide** — every tensor role has almost the
  same 16-bin bell curve (Q4_K, all roles):
  `0.041 0.026 0.038 0.052 0.068 0.084 0.096 0.103 0.102 0.094 0.081 0.065 0.049 0.035 0.029 0.034`.
  One shared dictionary costs nothing vs per-tensor ones.
- **But the curve is nearly flat.** k-quant scale fitting picks each block's
  scale so its values span the whole code range, which maximizes code usage
  by design. The quantizer has already removed what Huffman would find.
  The bumps at codes 0 and 15 are clipping at the block edges.
- **No repeated code vocabulary.** Pairs and 4-code words add ~nothing; every
  possible Q4_K 4-code word occurs.
- **Q8_0 has the most slack** (~4–5%) because its single scale per 32 weights
  is set by the block max, leaving the bell shape intact — but Q8_0 is a small
  share of a Q4 model.

The one outlier is the IQ4_NL `per_layer_token_embd` table: only 36k of 65k
4-code words occur (quad H 3.725 vs 3.914), worth roughly another 7% / ~2 GB
on that tensor with a context model. Even counting that, total lossless
headroom stays under ~4%.

## Where the real fit wins are (not yet implemented)

Structural, still lossless with respect to the GGUF:

1. **`per_layer_token_embd` is 28.8 GB (~26% of the file)** and is a lookup
   table, not a matmul. Each token reads only its own rows, so it can stay
   mmap'd on disk instead of resident.
2. **Experts are ~70 GB, but only 10 of 512 run per token.** Keep attention
   and shared-expert weights on the card and experts in host RAM.

Together these put the hot working set well within 32 GB VRAM + host RAM.

## Usage

All CPU-only; tensors are mmap'd and sampled in large sequential chunks
(small random seeks across 100 GB stall on disk). The full 111 GB model scans
in a few minutes.

```bash
# Full report: per-type table, context tests, per-role histograms
./venv/bin/python analysis/gguf_entropy.py ../gguf/Qwen3.8-Flash-Next
./venv/bin/python analysis/gguf_entropy.py model.gguf --blocks 4096

# survey.py / analyze.py detect a GGUF file or shard directory automatically
./venv/bin/python survey.py ../gguf/Qwen3.8-Flash-Next      # 2048 blocks/tensor
./venv/bin/python analyze.py ../gguf/Qwen3.8-Flash-Next     # 8192 blocks/tensor
./venv/bin/python analyze.py ../gguf/Qwen3.8-Flash-Next --quick  # metadata only
```

`--blocks` is the maximum number of quant blocks sampled per tensor; small
tensors are read whole.
