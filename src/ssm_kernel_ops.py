"""
ssm_kernel_ops.py — Python wrappers for HIP SSM kernels.

Provides:
  HIPCausalConv1d          — drop-in for nn.Conv1d (depthwise, causal, SiLU)
  HIPGatedDeltaRuleDecode  — replaces torch_recurrent_gated_delta_rule (seq_len=1)
  inject_ssm_kernels(model) — walk model and patch GatedDeltaNet modules in-place

Usage:
    from ssm_kernel_ops import inject_ssm_kernels
    inject_ssm_kernels(model)   # patches all GatedDeltaNet layers

Note: prefill (seq_len > 1) still uses the PyTorch fallback.  The decode
step (seq_len = 1) is fully handled by our HIP kernel and is where the
interactive latency improvement comes from.
"""

import ctypes as _ct
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Load the shared library via the existing extension mechanism
# ---------------------------------------------------------------------------

def _get_lib():
    """Return the loaded ROCm kernel library, or None if unavailable."""
    try:
        from gpu_accelerated_functions import _load_extension
        ext = _load_extension()
        if ext is None:
            return None
        lib = ext._lib

        # Register SSM function signatures (idempotent — ctypes allows re-setting)
        _sig_3int = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                     _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_void_p]

        lib.ck_causal_conv1d_prefill_f32.restype  = _ct.c_int
        lib.ck_causal_conv1d_prefill_f32.argtypes = [
            _ct.c_void_p,  # x      [B, C, L]
            _ct.c_void_p,  # weight [C, ksz]
            _ct.c_void_p,  # out    [B, C, L]
            _ct.c_int,     # B
            _ct.c_int,     # C
            _ct.c_int,     # L
            _ct.c_int,     # ksz
            _ct.c_void_p,  # stream
        ]

        lib.ck_causal_conv1d_update_f32.restype  = _ct.c_int
        lib.ck_causal_conv1d_update_f32.argtypes = [
            _ct.c_void_p,  # x_new      [B, C]
            _ct.c_void_p,  # conv_state [B, C, ksz]
            _ct.c_void_p,  # weight     [C, ksz]
            _ct.c_void_p,  # out        [B, C]
            _ct.c_int,     # B
            _ct.c_int,     # C
            _ct.c_int,     # ksz
            _ct.c_void_p,  # stream
        ]

        lib.ck_gdr_decode_step_f32.restype  = _ct.c_int
        lib.ck_gdr_decode_step_f32.argtypes = [
            _ct.c_void_p,  # q      [B, H, KD]
            _ct.c_void_p,  # k      [B, H, KD]
            _ct.c_void_p,  # v      [B, H, VD]
            _ct.c_void_p,  # log_g  [B, H]
            _ct.c_void_p,  # beta   [B, H]
            _ct.c_void_p,  # state  [B, H, KD, VD]
            _ct.c_void_p,  # out    [B, H, VD]
            _ct.c_int,     # B
            _ct.c_int,     # H
            _ct.c_int,     # KD
            _ct.c_int,     # VD
            _ct.c_void_p,  # stream
        ]

        lib.ck_gdr_prefill_sequential_f32.restype  = _ct.c_int
        lib.ck_gdr_prefill_sequential_f32.argtypes = [
            _ct.c_void_p,  # q      [B, SL, H, KD]
            _ct.c_void_p,  # k      [B, SL, H, KD]
            _ct.c_void_p,  # v      [B, SL, H, VD]
            _ct.c_void_p,  # log_g  [B, SL, H]
            _ct.c_void_p,  # beta   [B, SL, H]
            _ct.c_void_p,  # state  [B, H, KD, VD]
            _ct.c_void_p,  # out    [B, SL, H, VD]
            _ct.c_int,     # B
            _ct.c_int,     # SL
            _ct.c_int,     # H
            _ct.c_int,     # KD
            _ct.c_int,     # VD
            _ct.c_void_p,  # stream
        ]

        return lib
    except Exception as e:
        print(f"  [ssm_kernel_ops] HIP library unavailable: {e}")
        return None


def _stream_ptr():
    try:
        return _ct.c_void_p(torch.cuda.current_stream().cuda_stream)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# L2 norm helper (matches the model's use_qk_l2norm_in_kernel=True path)
# ---------------------------------------------------------------------------

def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# HIPCausalConv1d
# ---------------------------------------------------------------------------

class HIPCausalConv1d(nn.Module):
    """
    Drop-in replacement for nn.Conv1d(groups=C, padding=ksz-1) + SiLU.

    Handles both prefill (L > 1) and decode (L == 1 with conv_state cache).
    The conv_state is a plain float32 Tensor [B, C, ksz] managed externally
    (by the model's cache_params, same as the original module).
    """

    def __init__(self, conv1d_module: nn.Conv1d):
        super().__init__()
        self._lib = _get_lib()
        C   = conv1d_module.in_channels
        ksz = conv1d_module.kernel_size[0]
        self.C   = C
        self.ksz = ksz
        # Upload weight [C, 1, ksz] → [C, ksz] float32 to GPU
        w = conv1d_module.weight.squeeze(1).float().cuda().contiguous()
        self.register_buffer('weight', w)
        # bias is None in Qwen3.5 GatedDeltaNet
        if conv1d_module.bias is not None:
            self.register_buffer('bias', conv1d_module.bias.float().cuda())
        else:
            self.bias = None

    def forward_prefill(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] → [B, C, L] with SiLU."""
        B, C, L = x.shape
        x_f32  = x.float().contiguous()
        out    = torch.empty_like(x_f32)

        if self._lib is not None:
            rc = self._lib.ck_causal_conv1d_prefill_f32(
                _ct.c_void_p(x_f32.data_ptr()),
                _ct.c_void_p(self.weight.data_ptr()),
                _ct.c_void_p(out.data_ptr()),
                B, C, L, self.ksz,
                _stream_ptr(),
            )
            if rc == 0:
                return out.to(x.dtype)

        # Fallback: standard PyTorch conv1d + SiLU
        return F.silu(F.conv1d(x, self.weight.unsqueeze(1),
                               groups=C, padding=self.ksz - 1)[:, :, :L])

    def forward_decode(self, x_new: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """x_new: [B, C], conv_state: [B, C, ksz] (updated in-place) → [B, C]."""
        B, C = x_new.shape
        x_f32     = x_new.float().contiguous()
        state_f32 = conv_state.float().contiguous()
        out       = torch.empty(B, C, dtype=torch.float32, device=x_new.device)

        if self._lib is not None:
            rc = self._lib.ck_causal_conv1d_update_f32(
                _ct.c_void_p(x_f32.data_ptr()),
                _ct.c_void_p(state_f32.data_ptr()),
                _ct.c_void_p(self.weight.data_ptr()),
                _ct.c_void_p(out.data_ptr()),
                B, C, self.ksz,
                _stream_ptr(),
            )
            if rc == 0:
                conv_state.copy_(state_f32)
                return out.to(x_new.dtype)

        # Fallback: pure PyTorch state update
        state_f32 = state_f32.roll(-1, -1)
        state_f32[:, :, -1] = x_f32
        conv_state.copy_(state_f32)
        result = (state_f32 * self.weight.unsqueeze(0)).sum(-1)
        return F.silu(result).to(x_new.dtype)

    # Note: forward() is not used directly — the GatedDeltaNet forward calls
    # self.conv1d(...) with different shapes/paths.  We expose the two helpers
    # above and patch the GatedDeltaNet forward via monkey-patching below.


# ---------------------------------------------------------------------------
# HIPGatedDeltaRuleDecode
# ---------------------------------------------------------------------------

class HIPGatedDeltaRuleDecode:
    """
    Callable that replaces torch_recurrent_gated_delta_rule for the
    seq_len == 1 decode step.  For seq_len > 1 (prefill) it falls back to
    the PyTorch implementation supplied at construction time.

    Usage:
        mod.recurrent_gated_delta_rule = HIPGatedDeltaRuleDecode(
            lib, torch_recurrent_fallback)
    """

    def __init__(self, lib, fallback_fn):
        self._lib      = lib
        self._fallback = fallback_fn

    def __call__(
        self,
        query, key, value, g, beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        **kwargs,
    ):
        B, seq_len, H, KD = query.shape
        VD = value.shape[-1]
        initial_dtype = query.dtype

        if self._lib is None:
            return self._fallback(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        # Prefill path: seq_len > 1 — single kernel call, state stays in LDS.
        if seq_len != 1:
            return self._sequential_prefill(
                query, key, value, g, beta,
                initial_state, output_final_state, use_qk_l2norm_in_kernel,
            )

        # Only accelerate decode (seq_len == 1) past this point.

        # --- Prepare inputs: L2-norm, scale, to float32, squeeze seq dim ---
        q = query.squeeze(1).float()  # [B, H, KD]
        k = key.squeeze(1).float()
        v = value.squeeze(1).float()  # [B, H, VD]

        if use_qk_l2norm_in_kernel:
            q = _l2norm(q, dim=-1)
            k = _l2norm(k, dim=-1)
        q = q / (KD ** 0.5)

        log_g = g.squeeze(1).float().contiguous()    # [B, H]
        beta_ = beta.squeeze(1).float().contiguous() # [B, H]

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if initial_state is None:
            state = torch.zeros(B, H, KD, VD, dtype=torch.float32, device=query.device)
        else:
            state = initial_state.float().contiguous()

        smem_bytes = KD * VD * 4
        if smem_bytes > 65536:
            # Too large for LDS — use fallback
            return self._fallback(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        out = torch.empty(B, H, VD, dtype=torch.float32, device=query.device)

        rc = self._lib.ck_gdr_decode_step_f32(
            _ct.c_void_p(q.data_ptr()),
            _ct.c_void_p(k.data_ptr()),
            _ct.c_void_p(v.data_ptr()),
            _ct.c_void_p(log_g.data_ptr()),
            _ct.c_void_p(beta_.data_ptr()),
            _ct.c_void_p(state.data_ptr()),
            _ct.c_void_p(out.data_ptr()),
            B, H, KD, VD,
            _stream_ptr(),
        )

        if rc != 0:
            return self._fallback(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        # out: [B, H, VD] → [B, 1, H, VD] to match expected return shape
        core_out = out.unsqueeze(1).to(initial_dtype)
        last_state = state if output_final_state else None
        return core_out, last_state

    def _sequential_prefill(
        self,
        query, key, value, g, beta,
        initial_state, output_final_state, use_qk_l2norm_in_kernel,
    ):
        """Process full input sequence in a single kernel launch (Phase 3b).

        State [KD, VD] stays in GPU shared memory (LDS) across all tokens —
        O(1) kernel launches per SSM layer regardless of sequence length.
        Falls back to PyTorch if LDS budget exceeded.
        """
        B, seq_len, H, KD = query.shape
        VD = value.shape[-1]
        initial_dtype = query.dtype

        smem_bytes = KD * VD * 4
        if smem_bytes > 65536:
            return self._fallback(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        # Prepare inputs: [B, seq_len, H, KD/VD] contiguous float32
        q = query.float().contiguous()
        k = key.float().contiguous()
        v = value.float().contiguous()

        if use_qk_l2norm_in_kernel:
            q = _l2norm(q, dim=-1)
            k = _l2norm(k, dim=-1)
        q = q / (KD ** 0.5)

        q = q.contiguous()
        k = k.contiguous()
        log_g = g.float().contiguous()    # [B, seq_len, H]
        beta_ = beta.float().contiguous() # [B, seq_len, H]

        if initial_state is None:
            state = torch.zeros(B, H, KD, VD, dtype=torch.float32, device=query.device)
        else:
            state = initial_state.float().contiguous()

        out = torch.empty(B, seq_len, H, VD, dtype=torch.float32, device=query.device)

        rc = self._lib.ck_gdr_prefill_sequential_f32(
            _ct.c_void_p(q.data_ptr()),
            _ct.c_void_p(k.data_ptr()),
            _ct.c_void_p(v.data_ptr()),
            _ct.c_void_p(log_g.data_ptr()),
            _ct.c_void_p(beta_.data_ptr()),
            _ct.c_void_p(state.data_ptr()),
            _ct.c_void_p(out.data_ptr()),
            B, seq_len, H, KD, VD,
            _stream_ptr(),
        )

        if rc != 0:
            return self._fallback(
                query, key, value, g, beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        # out: [B, seq_len, H, VD] — matches expected return shape
        last_state = state if output_final_state else None
        return out.to(initial_dtype), last_state


# ---------------------------------------------------------------------------
# Model injection
# ---------------------------------------------------------------------------

def inject_ssm_kernels(model: nn.Module) -> int:
    """Patch GatedDeltaNet layers to use HIP kernels.

    For each Qwen3_5GatedDeltaNet module replaces recurrent_gated_delta_rule
    with HIPGatedDeltaRuleDecode which dispatches:
      - seq_len == 1 (decode): ck_gdr_decode_step_f32
      - seq_len >  1 (prefill): ck_gdr_prefill_sequential_f32  (Phase 3b)
        State stays in LDS across all tokens — O(1) kernel launches per layer.

    Conv1d replacement (Phase 2b) is deferred.

    Returns the number of modules patched.
    """
    lib = _get_lib()
    if lib is None:
        print("  [ssm_kernel_ops] HIP library not available; skipping SSM injection.")
        return 0

    patched = 0
    for name, mod in model.named_modules():
        if type(mod).__name__ != 'Qwen3_5GatedDeltaNet':
            continue

        # Patch only the decode path (seq_len=1).
        # chunk_gated_delta_rule (prefill) stays as the Python fallback:
        # it uses rocBLAS for intra-chunk GEMMs which is faster than our
        # sequential kernel for large T on MI50.
        decode_fb = getattr(mod, 'recurrent_gated_delta_rule', None)
        if decode_fb is not None and not isinstance(decode_fb, HIPGatedDeltaRuleDecode):
            mod.recurrent_gated_delta_rule = HIPGatedDeltaRuleDecode(lib, decode_fb)
            patched += 1

    if patched:
        print(f"  [ssm_kernel_ops] Injected HIP GDR decode kernel into "
              f"{patched} GatedDeltaNet layers.")
    return patched
