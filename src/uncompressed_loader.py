"""
uncompressed_loader.py — Load a stock (uncompressed) model and route all
nn.Linear / nn.Embedding layers through our HIP raw-kernel path.

This bypasses PyTorch's ROCm dispatch for every matmul, giving the same
kernel path as compressed inference without requiring a compression cache.

Usage:
    loader = UncompressedKernelLoader(device='cuda')
    model  = loader.load(model_path, config, tokenizer)

The model is returned in eval mode with all Linear/Embedding weights on GPU
inside our kernel objects.  Norms, biases, and other small parameters are
moved to the inference device via model.to(device) at the end.

Phase 2 (SSM kernel bypass for causal conv1d and gated delta rule) is a
future extension; this module handles Phase 1 only.
"""

import gc
import sys
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from gpu_accelerated_functions import GPUAcceleratedLinear, GPUAcceleratedEmbedding
from memory_utils import resolve_device

try:
    from ssm_kernel_ops import inject_ssm_kernels
    _SSM_AVAILABLE = True
except Exception:
    _SSM_AVAILABLE = False

try:
    _HIP_AVAILABLE = bool(GPUAcceleratedLinear.from_weight)
except Exception:
    _HIP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Thin nn.Module wrappers around GPUAccelerated* so they slot into
# the existing model tree without changing any other code.
# ---------------------------------------------------------------------------

class RawKernelLinear(nn.Module):
    """Drop-in for nn.Linear: weight lives on GPU, forward uses ck_linear_raw_f32."""

    def __init__(self, name: str, weight: torch.Tensor, bias=None):
        super().__init__()
        self._gpu = GPUAcceleratedLinear.from_weight(name, weight, tuple(weight.shape))
        # bias: keep as a plain tensor attribute so model.to(device) moves it
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().detach(), requires_grad=False)
        else:
            self.bias = None
        # Expose shape for anything that inspects .weight.shape
        self.in_features  = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For T > 1 (prefill / batch), fall back to rocBLAS F.linear — our raw
        # kernel launches one block per output row which is inefficient for large T.
        # For T == 1 (decode), our kernel avoids rocBLAS dispatch overhead.
        T = x.shape[0] if x.dim() == 2 else x.reshape(-1, x.shape[-1]).shape[0]
        if T > 1:
            w = self._gpu.weight_gpu
            flat = x.reshape(-1, x.shape[-1])
            out = torch.nn.functional.linear(flat, w.to(flat.dtype)).reshape(*x.shape[:-1], self.out_features)
        else:
            out = self._gpu(x).to(dtype=x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(dtype=x.dtype, device=out.device)
        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, kernel=raw_hip"


class RawKernelEmbedding(nn.Module):
    """Drop-in for nn.Embedding: weight lives on GPU, forward uses ck_embedding_raw_f32."""

    def __init__(self, name: str, weight: torch.Tensor):
        super().__init__()
        self._gpu = GPUAcceleratedEmbedding.from_weight(name, weight, tuple(weight.shape))
        self.num_embeddings = weight.shape[0]
        self.embedding_dim  = weight.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._gpu(x)

    def extra_repr(self):
        return f"vocab={self.num_embeddings}, dim={self.embedding_dim}, kernel=raw_hip"


# ---------------------------------------------------------------------------
# Module-tree walker
# ---------------------------------------------------------------------------

def _parent_and_attr(root: nn.Module, full_name: str):
    """Return (parent_module, leaf_attr_name) for a dotted parameter path."""
    parts = full_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _replace_linear_and_embedding(model: nn.Module, device: str) -> int:
    """Walk model tree, replace nn.Linear and nn.Embedding with raw-kernel variants.

    Weights are uploaded to GPU immediately during replacement so the model
    shell can be moved to the inference device afterwards without touching
    the (already-GPU) kernel weights.

    Returns the number of modules replaced.
    """
    replaced = 0
    use_gpu = (device != 'cpu')

    for name, child in list(model.named_modules()):
        if isinstance(child, (RawKernelLinear, RawKernelEmbedding)):
            continue  # already replaced

        if isinstance(child, nn.Linear):
            parent, attr = _parent_and_attr(model, name)
            w = child.weight.data
            b = child.bias.data if child.bias is not None else None
            new_mod = RawKernelLinear(name, w, b)
            setattr(parent, attr, new_mod)
            replaced += 1

        elif isinstance(child, nn.Embedding):
            parent, attr = _parent_and_attr(model, name)
            w = child.weight.data
            new_mod = RawKernelEmbedding(name, w)
            setattr(parent, attr, new_mod)
            replaced += 1

    return replaced


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class UncompressedKernelLoader:
    """Load a stock HuggingFace model and apply HIP raw-kernel bypass."""

    def __init__(self, device: str = 'cuda'):
        self.device = resolve_device(device)

    def load(self, model_path: Path, config=None, dtype=torch.bfloat16) -> nn.Module:
        """Full pipeline: from_pretrained → replace layers → move to device.

        Parameters
        ----------
        model_path : Path
            Directory containing the HuggingFace model (config.json, *.safetensors).
        config : PretrainedConfig, optional
            Pre-loaded config.  If None, loaded from model_path.
        dtype : torch.dtype
            Model dtype (default: bfloat16).

        Returns
        -------
        nn.Module
            Model in eval mode with all Linear/Embedding on GPU via our kernel.
        """
        model_path = Path(model_path)

        print(f"\n  Loading stock model to CPU (dtype={dtype})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map='cpu',
            trust_remote_code=True,
        )

        if self.device == 'cpu':
            print("  [warn] No GPU available — raw kernel requires CUDA/HIP; "
                  "running CPU fallback.")
            model.eval()
            return model

        print("  Replacing Linear/Embedding with HIP raw-kernel modules...")
        replaced = _replace_linear_and_embedding(model, self.device)
        print(f"  Replaced {replaced} modules.")

        if _SSM_AVAILABLE:
            print("  Injecting HIP SSM kernels (conv1d + gated delta rule decode)...")
            inject_ssm_kernels(model)

        print(f"  Moving remaining parameters to {self.device}...")
        model.to(device=self.device, dtype=dtype)

        gc.collect()
        model.eval()
        return model
