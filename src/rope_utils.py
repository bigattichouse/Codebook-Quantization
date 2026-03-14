"""
rope_utils.py — RoPE (Rotary Position Embedding) buffer reinitialization.

Root cause of Qwen3/Llama/Mistral layer_00 divergence:
  from_config() on meta device: __init__ runs, inv_freq becomes a meta tensor.
  to_empty(device):             allocates UNINITIALIZED storage (garbage floats).
  model.to(bfloat16):           converts garbage to bfloat16 garbage.

Since inv_freq is a COMPUTED buffer (not a learned weight), it is absent from
the compression cache and _load_exact_weights() cannot restore it.  This module
recomputes inv_freq from the model config hyperparameters after model.to().

Affected models: Qwen3, Llama, Mistral, Phi, Gemma, and any HuggingFace
transformer that stores inv_freq as a nn.Module buffer.

Config locations for rope_theta (checked in order):
  config.rope_parameters.rope_theta  (Qwen3.5 multimodal)
  config.rope_scaling.rope_theta     (Qwen3 decoder-only)
  config.rope_theta                  (Llama, Mistral, Phi)
  text_config variants of all above  (multimodal wrappers)
  fallback: 10000.0                  (safe default)
"""

import torch


def resolve_rope_theta(config) -> float:
    """Extract rope_theta from a HuggingFace config object."""
    text_cfg = getattr(config, 'text_config', None)

    def _get(obj, *keys):
        for k in keys:
            v = getattr(obj, k, None)
            if v is not None:
                return v
        return None

    def _from_dict_or_obj(d):
        if isinstance(d, dict):
            return d.get('rope_theta')
        return getattr(d, 'rope_theta', None)

    rope_params = (
        _get(config, 'rope_parameters') or _get(config, 'rope_scaling') or
        _get(text_cfg, 'rope_parameters') or _get(text_cfg, 'rope_scaling')
    )
    theta = (
        (_from_dict_or_obj(rope_params) if rope_params else None) or
        _get(config, 'rope_theta') or
        _get(text_cfg, 'rope_theta') or
        10000.0
    )
    return float(theta)


def reinit_rope_buffers(model, config) -> int:
    """Recompute all inv_freq buffers in model from config hyperparameters.

    Returns the number of buffers reinitialized.

    Detection: any module with an inv_freq buffer whose dtype is not float32
    (meaning model.to() converted it away from float32), OR whose values look
    like garbage (max > 1.5, since valid inv_freq is in (0, 1]).
    """
    theta = resolve_rope_theta(config)
    reinitialized = 0

    for module_name, module in model.named_modules():
        if not hasattr(module, 'inv_freq'):
            continue
        buf = module.inv_freq
        if buf is None or buf.numel() == 0:
            continue

        buf_f32 = buf.detach().float()
        needs_reinit = (
            buf.dtype != torch.float32 or
            buf_f32.max().item() > 1.5 or
            buf_f32.min().item() < -1e-6
        )
        if not needs_reinit:
            continue

        half_dim = buf.numel()
        inv_freq = (
            1.0 / (theta ** (
                torch.arange(0, half_dim * 2, 2, dtype=torch.float32) / (half_dim * 2)
            ))
        ).to(device=buf.device)

        # register_buffer ensures float32 dtype is preserved across future model.to() calls.
        # persistent=False matches HuggingFace convention for this buffer.
        module.register_buffer('inv_freq', inv_freq, persistent=False)

        # Some NTK/YaRN implementations keep a second copy.
        if hasattr(module, 'original_inv_freq'):
            orig = module.original_inv_freq
            if orig is not None and orig.numel() == half_dim:
                module.register_buffer('original_inv_freq', inv_freq.clone(), persistent=False)

        reinitialized += 1
        print(f"  🔧 Reinitialized RoPE inv_freq in '{module_name}' "
              f"(theta={theta:.0f}, half_dim={half_dim}, was dtype={buf.dtype})")

    return reinitialized
