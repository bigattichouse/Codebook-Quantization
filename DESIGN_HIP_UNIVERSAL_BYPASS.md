# Design: Universal HIP Kernel Bypass for Inference

**Goal:** Route all layers — compressed *and* uncompressed — through our own HIP kernels,
completely bypassing PyTorch's ROCm dispatch and the slow Python fallbacks for SSM ops.
Deliverable: `--mode uncompressed` that loads a stock model and still hits the fast path.

---

## Implementation Status (2026-03-14)

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — Raw Linear/Embedding bypass (`ck_linear_raw_bf16`) | ✅ Done | 13.3 tok/s uncompressed |
| 2 — HIP causal conv1d (prefill + decode) | ✅ Kernel done | Not yet wired to GatedDeltaNet (see §2b below) |
| 2b — HIPCausalConv1d proper GatedDeltaNet integration | 🔲 Deferred | Needs forward() + causal_conv1d_update monkey-patch |
| 3a — HIP GDR decode step | ✅ Kernel done | Not injected by default: wrapper overhead > compute savings at seq_len=1 |
| 3b — HIP chunk_gated_delta_rule (prefill) | 🔲 Pending | Highest remaining value; prefill latency not yet measured |

**Actual vs predicted speeds (MI50, Qwen3.5-9B):**

| Mode | Predicted | Actual |
|------|-----------|--------|
| Stock PyTorch | ~1.4 tok/s | 1.4 tok/s ✓ |
| Phase 1 only | ~1.6 tok/s | — |
| Phase 1 + 3a | ~5–8 tok/s | — |
| Uncompressed HIP (Phase 1 + 2 + 3a injected) | — | **13.3 tok/s** |
| Compressed HIP baseline | 7.7 tok/s | **8.6 tok/s** |

Phase 1 alone (raw bf16 linear) delivered the majority of the gain.
The GDR decode kernel adds overhead at seq_len=1 and is disabled by default.
Phase 3b (prefill chunk scan) is the next significant opportunity.

---

---

## 1. Motivation

### Why the compressed model beats uncompressed (7.7 vs 1.4 tok/s)

Two reasons, not one:

| Root cause | Compressed | Uncompressed |
|---|---|---|
| Linear matmuls | our HIP kernel (bitpack → lookup) | PyTorch → rocBLAS |
| SSM ops (GatedDeltaNet) | our HIP kernel (same matmul path, bypasses SSM) | PyTorch Python fallback (nested loops) |

The SSM layers are the dominant bottleneck for uncompressed inference. Qwen3.5-9B alternates
`full_attention` and `linear_attention` layers. The `linear_attention` type uses
`Qwen3_5GatedDeltaNet`, which needs two specialised kernels that PyTorch falls back to pure
Python when `flash-linear-attention` is absent:

```
torch_chunk_gated_delta_rule   — prefill: O(seq_len/chunk_size) Python iterations,
                                  each doing O(chunk_size²) tensor ops
torch_recurrent_gated_delta_rule — decode: O(seq_len) Python iterations (seq_len=1
                                   in steady-state, so this is fine, but prefill kills it)
```

A single prompt of 50 tokens runs 50 iterations of a Python loop per SSM layer.
At 40+ SSM layers, that is 2 000+ Python loop trips just for prefill.

### What we already have

`compressed_kernel.hip` already exports:
- `ck_linear_raw_f32 / ck_linear_raw_bf16` — raw (uncompressed) matmul, no codebook
- `ck_embedding_raw_f32` — raw embedding lookup
- `ck_upload_weights_f32 / ck_upload_weights_bf16` — device-side weight upload

`gpu_accelerated_functions.py` already wraps these behind `GPUAcceleratedLinear.from_weight()`
and `GPUAcceleratedEmbedding.from_weight()`.

So **Phase 1 is essentially free**. Phases 2 and 3 add new HIP kernels.

---

## 2. Architecture

```
chat.py --mode uncompressed
    └── UncompressedKernelLoader.create_and_load()
            ├── Phase 1: replace every nn.Linear  → RawKernelLinear
            │           replace every nn.Embedding → RawKernelEmbedding
            ├── Phase 2: replace nn.Conv1d (depthwise) → HIPCausalConv1d
            └── Phase 3: monkey-patch GatedDeltaNet.forward()
                         to call HIPGatedDeltaRule instead of
                         torch_chunk_gated_delta_rule /
                         torch_recurrent_gated_delta_rule
```

For the compressed path, the same Phase 2 and Phase 3 are applied on top of what
`CompressedModelLoader` already does.  The compressed linears already use our kernel;
we just add the SSM ops.

---

## 3. Phase 1 — Raw Kernel for All Linear / Embedding

### What changes

`model_loader._replace_modules_recursive` currently skips nn.Linear if there is no
compressed cache entry for it.  Add a `replace_uncompressed: bool` flag:

```python
# current (simplified)
if not (data and data['mode'] == 'direct_codebook'):
    return          # ← skip non-compressed layers

# new
if not (data and data['mode'] == 'direct_codebook'):
    if self.replace_uncompressed and use_gpu:
        weight = child.weight.data          # already on CPU at this point
        new_layer = AdaptiveCodebookLinear.from_weight(full_name, weight, child.weight.shape)
        if child.bias is not None:
            new_layer.bias = child.bias.data.clone()
        setattr(parent, attr_name, new_layer)
        self.modules_replaced += 1
    return
```

`from_weight()` calls `ck_upload_weights_f32` to move the weight to the GPU and stores
it there.  Subsequent forward calls use `ck_linear_raw_f32`, bypassing PyTorch dispatch.

**Memory cost:** identical to the PyTorch baseline — full float32/bfloat16 weights on GPU.
**Speed benefit:** removes PyTorch dispatch overhead; more importantly, makes all layer
types use the same kernel, which will matter more once Phase 3 lands.

### Why raw matmul may not beat rocBLAS for large GEMM

For square/large M×K GEMM, rocBLAS uses tiled algorithms with better wavefront utilisation
than our one-block-per-output-row kernel.  The raw kernel will be competitive or faster for
GEMV (T=1 decode, where rocBLAS path has dispatch overhead per call).  During decode this is
the dominant regime, so Phase 1 is a net win regardless.

For prefill (T > 1, true GEMM), consider falling back to `torch.nn.functional.linear` for
non-SSM layers and only bypass for SSM.  This is a tuning decision, not an architectural one.

---

## 4. Phase 2 — HIP Causal Conv1d

### What it replaces

`Qwen3_5GatedDeltaNet` contains:

```python
self.conv1d = nn.Conv1d(
    in_channels=conv_dim,
    out_channels=conv_dim,
    kernel_size=conv_kernel_size,   # typically 4
    groups=conv_dim,                # depthwise
    bias=False,
    padding=conv_kernel_size - 1,
)
```

Forward path (prefill, no fast path):
```python
mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
```

Forward path (decode, no fast path):
```python
# causal_conv1d_update does a single-step conv state update
mixed_qkv = self.causal_conv1d_update(mixed_qkv, conv_state, weight, bias, activation)
```

### New HIP kernel signature

```c
// compressed_kernel.hip additions:

// Prefill: depthwise causal conv1d + silu activation
// x:      [B, C, L]  (bfloat16 or float32)
// weight: [C, kernel_size]
// bias:   [C] or NULL
// out:    [B, C, L]
int ck_causal_conv1d_f32(
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C, int L, int kernel_size,
    hipStream_t stream
);

// Decode: single-step conv state update
// conv_state: [B, C, kernel_size-1]  (ring buffer of past inputs)
// x_t:        [B, C]                 (single new timestep)
// out:         [B, C]
int ck_causal_conv1d_update_f32(
    float* conv_state, const float* x_t,
    const float* weight, const float* bias,
    float* out,
    int B, int C, int kernel_size,
    hipStream_t stream
);
```

### Kernel design (prefill)

```
Grid(B × C, ceil(L/64)),  Block(64 threads)

Each block computes one (batch, channel) strip of L output elements.
Thread tid handles output positions tid, tid+64, tid+128, ...

For each output position t:
  acc = 0
  for k in range(kernel_size):
      in_pos = t - k
      acc += (in_pos >= 0) ? x[b, c, in_pos] * w[c, k] : 0
  out[b, c, t] = silu(acc + bias[c])
```

kernel_size is 4 (tiny), so the inner loop is unrolled by the compiler.
All C channels are independent — grid dimension covers them.

### Python wrapper

```python
class HIPCausalConv1d(nn.Module):
    """Drop-in replacement for nn.Conv1d(groups=C) with causal masking + SiLU."""

    def __init__(self, conv1d_module: nn.Conv1d, ext: ROCmKernelExtension):
        super().__init__()
        self._ext  = ext
        self.C     = conv1d_module.in_channels
        self.ksz   = conv1d_module.kernel_size[0]
        # Upload weight to GPU once
        w = conv1d_module.weight.squeeze(1).to(torch.float32).cuda()
        self._weight = w.contiguous()
        self._bias   = None  # conv1d has bias=False in GatedDeltaNet

    def forward(self, x):          # x: [B, C, L]
        B, C, L = x.shape
        out = torch.empty_like(x)
        self._ext.ck_causal_conv1d_f32(x, self._weight, self._bias, out, B, C, L, self.ksz)
        return out
```

Module injection: walk model tree, find `nn.Conv1d` where `groups == in_channels` inside
`GatedDeltaNet` and replace with `HIPCausalConv1d`.

---

## 5. Phase 3 — HIP Gated Delta Rule

This is the core SSM recurrence and the biggest win.

### Mathematics (decode, single step)

State `H ∈ ℝ^{heads × k_dim × v_dim}`, per-step update:

```
H_t = H_{t-1} * g_t                          # decay
kv  = (H_t * k_t[...,None]).sum(-2)           # project state → value space
δ   = (v_t - kv) * β_t                        # delta rule correction
H_t = H_t + outer(k_t, δ)                    # rank-1 state update
y_t = (H_t * q_t[...,None]).sum(-2)           # readout
```

For Qwen3.5-9B typical dims: heads=64, k_dim=64, v_dim=128.
State per SSM layer: `64 × 64 × 128 × 4 bytes = 2 MB`.

### Decode kernel (seq_len = 1)

```
Grid(B × heads, ceil(v_dim / 32)),  Block(32 threads)

Each block handles one (batch, head) pair, covering v_dim output elements.
q, k, v, g, beta: [B, heads, dim] — one vector each.

// Pseudocode, shared memory for H column reuse
for kd in range(k_dim):
    h_col = H[b, h, kd, :]        // load v_dim-length column (one global read)
    kv += k[kd] * h_col            // accumulate kv projection
    H[b, h, kd, :] *= g           // decay entire column
// delta = (v - kv) * beta
for kd in range(k_dim):
    H[b, h, kd, :] += k[kd] * delta   // rank-1 update (k_dim writes)
y = (H * q).sum(kd)               // readout
```

Memory: H lives in GPU global memory, one slice at a time.  k_dim=64 global reads per
block, each of v_dim=128 float32 = 512 bytes — fits in L1/L2 easily.

### Prefill kernel (seq_len > 1)

The chunk-parallel algorithm is identical to `torch_chunk_gated_delta_rule` but the outer
loop over chunks and the inner attention-within-chunk GEMM are all on-device.

Key observation: within each chunk (size=64), the intra-chunk attention is a standard
causal masked GEMM: `[heads, chunk, k_dim] × [heads, k_dim, chunk] → [heads, chunk, chunk]`.
This fits naturally into a tiled HIP kernel or can reuse rocBLAS for the GEMM portions
while the scan (inter-chunk state update) runs as a separate kernel.

Staged approach for prefill:
1. **Intra-chunk kernel**: masked GEMM, one block per chunk per head.  Can call
   `rocblas_sgemm_strided_batched` from inside the host wrapper.
2. **Inter-chunk scan kernel**: sequential over chunks (small count), each updating the
   recurrent state `H` using the same decode kernel above.

This splits complexity: the GEMM uses the existing optimised library; we write only the
scan kernel (which has no good library equivalent).

### Python wrapper

```python
class HIPGatedDeltaRule:
    """Replaces torch_chunk_gated_delta_rule and torch_recurrent_gated_delta_rule."""

    def __init__(self, ext: ROCmKernelExtension):
        self._ext = ext

    def __call__(self, query, key, value, g, beta,
                 initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True, chunk_size=64):
        seq_len = query.shape[1]
        if seq_len == 1:
            return self._decode(query, key, value, g, beta, initial_state, output_final_state)
        else:
            return self._prefill(query, key, value, g, beta, initial_state,
                                 output_final_state, chunk_size)
    ...
```

Injection: after model creation, walk all `GatedDeltaNet` modules and replace:

```python
for mod in model.modules():
    if isinstance(mod, Qwen3_5GatedDeltaNet):
        hip_rule = HIPGatedDeltaRule(ext)
        mod.chunk_gated_delta_rule     = hip_rule
        mod.recurrent_gated_delta_rule = hip_rule
```

No subclassing needed — `chunk_gated_delta_rule` and `recurrent_gated_delta_rule` are
ordinary Python attributes on the module (set in `__init__`), so direct assignment works.

---

## 6. Mode: `--mode uncompressed`

### chat.py changes

```python
parser.add_argument('--mode', default='balanced',
                    choices=['balanced', 'lossless', 'uncompressed'])
```

New branch in `CompressedChatModel.load()`:

```python
if self.compression_mode == 'uncompressed':
    return self._load_uncompressed()
```

```python
def _load_uncompressed(self):
    """Load stock model and apply HIP kernel bypass to all layers."""
    from transformers import AutoModelForCausalLM
    config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    # Standard HuggingFace load onto CPU
    model = AutoModelForCausalLM.from_pretrained(
        self.model_path,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        trust_remote_code=True,
    )

    loader = UncompressedKernelLoader(device=self.device)
    self.model = loader.apply(model)
    return self
```

### UncompressedKernelLoader

```python
class UncompressedKernelLoader:
    """Apply HIP kernel bypass to a fully-loaded (uncompressed) model."""

    def __init__(self, device='cuda'):
        self.device = device
        self._ext   = _get_rocm_extension()   # same singleton used by AdaptiveCodebookLinear

    def apply(self, model):
        self._replace_linears(model)
        self._replace_conv1d(model)
        self._replace_ssm_ops(model)
        model.to(self.device)
        model.eval()
        return model

    def _replace_linears(self, model):
        for name, child in list(model.named_modules()):
            if isinstance(child, nn.Linear):
                # Re-use from_weight() which uploads to GPU and uses ck_linear_raw_*
                parent, attr = _parent_and_attr(model, name)
                new = AdaptiveCodebookLinear.from_weight(name, child.weight, child.weight.shape)
                if child.bias is not None:
                    new.bias = child.bias.data.clone()
                setattr(parent, attr, new)

    def _replace_conv1d(self, model):
        for name, child in list(model.named_modules()):
            if isinstance(child, nn.Conv1d) and child.groups == child.in_channels:
                parent, attr = _parent_and_attr(model, name)
                setattr(parent, attr, HIPCausalConv1d(child, self._ext))

    def _replace_ssm_ops(self, model):
        hip_rule = HIPGatedDeltaRule(self._ext)
        for mod in model.modules():
            if type(mod).__name__ == 'Qwen3_5GatedDeltaNet':
                mod.chunk_gated_delta_rule     = hip_rule
                mod.recurrent_gated_delta_rule = hip_rule
```

---

## 7. Implementation Order

| Phase | Effort | Expected gain for uncompressed | Actual / Status |
|---|---|---|---|
| 1 — Raw Linear bypass | Low | ~1.5× | ✅ 9.5× actual (raw bf16 vs Python SSM fallback) |
| 2 — HIP causal_conv1d | Medium | ~1.2× | ✅ Kernel done; integration deferred (Phase 2b) |
| 3a — HIP recurrent decode | Medium | ~3–5× | ✅ Kernel done; overhead > benefit at seq_len=1, disabled |
| 3b — HIP chunk prefill | High | **~5–10×** first-token | 🔲 Not started — measure prefill latency first |
| 2b — Conv1d GatedDeltaNet wiring | Medium | ~1.2× | 🔲 Deferred; lower priority than 3b |

**Next recommended step:** measure prefill latency baseline, then implement Phase 3b.

Phase 3a gives the biggest practical win because the model is most useful in interactive
decode mode (1 token at a time).  Phase 3b matters for long-context prefill.

---

## 8. File Layout

```
proofofconcept/
├── rocm/
│   ├── compressed_kernel.hip          # add ck_causal_conv1d_*, ck_gated_delta_*
│   └── compressed_kernel.h            # add declarations
└── src/
    ├── gpu_accelerated_functions.py   # add HIPCausalConv1d, HIPGatedDeltaRule wrappers
    ├── uncompressed_loader.py         # NEW: UncompressedKernelLoader
    ├── model_loader.py                # add replace_uncompressed flag + SSM injection call
    └── compressed_modules.py          # no changes needed for Phase 1

chat.py                                # add --mode uncompressed, call UncompressedKernelLoader
```

---

## 9. Expected Results

With all phases complete on MI50 / Qwen3.5-9B:

| Mode | Linear layers | SSM ops | Est. decode speed |
|---|---|---|---|
| Current uncompressed | rocBLAS | Python fallback | ~1.4 tok/s |
| Phase 1 only | our raw kernel | Python fallback | ~1.6 tok/s |
| Phase 1 + 3a | our raw kernel | HIP decode | ~5–8 tok/s |
| Phase 1 + 3a + 3b | our raw kernel | HIP decode + prefill | ~5–8 tok/s + fast prefill |
| Current compressed (baseline) | our packed kernel | bypassed | 7.7 tok/s |

Phase 3a brings uncompressed decode to parity with compressed, because the SSM bottleneck
is removed.  The remaining gap (raw matmul vs codebook lookup) is the bandwidth reduction
from 13-bit packing.

---

## 10. Notes and Risks

**Model-specific coupling**: Phase 3 patches `Qwen3_5GatedDeltaNet` by class name.  Other
SSM model families (Mamba, RWKV, Falcon-Mamba) would need separate injection points, but
the kernel math is the same — only the attribute names differ.

**Decode vs prefill kernel reuse**: the decode kernel (Phase 3a) is also a correct
(though slow) implementation of prefill — it can be used as a correctness reference and
as a temporary prefill fallback before Phase 3b is ready.

**Numerical precision**: the Python fallback promotes to float32 internally.  Our HIP
kernels should do the same (accumulate in f32, write back in bf16) to match reference.

**Conv1d in decoder generate()**: during `model.generate()`, conv states are part of the
KV cache (`cache_params.conv_states`).  `HIPCausalConv1d` must handle the two-path forward
(prefill: full conv; decode: `causal_conv1d_update`-style single-step) identically to the
original, updating `cache_params.conv_states` in place.
