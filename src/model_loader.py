"""
model_loader.py — Compressed model creation and weight loading.

Responsibilities:
  1. Create a zero-memory model on meta device.
  2. Materialize it on the target device via to_empty().
  3. Load exact-mode tensors (norms, biases, SSM scalars).
  4. Replace nn.Linear / nn.Embedding with AdaptiveCodebook* modules.
  5. Re-initialize RoPE buffers wiped by the meta→device transition.

This module is intentionally free of CLI, chat, and tokenizer logic.
Callers provide a pre-built NameResolver and pre-loaded compressor.

Typical call sequence from chat.py:
    loader = CompressedModelLoader(model_path, device, compressor, codebooks, ...)
    model  = loader.create_and_load(config, model_dtype, resolver)
"""

import gc
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM

from name_resolver import NameResolver
from rope_utils import reinit_rope_buffers
from memory_utils import resolve_device

try:
    from ssm_kernel_ops import inject_ssm_kernels
    _SSM_KERNELS_AVAILABLE = True
except Exception:
    _SSM_KERNELS_AVAILABLE = False

# Note: inject_ssm_kernels patches recurrent_gated_delta_rule (seq_len=1 decode)
# only.  chunk_gated_delta_rule (prefill) is kept as the Python fallback because
# torch_chunk_gated_delta_rule already uses rocBLAS for intra-chunk GEMMs and
# is faster than our sequential HIP kernel for large T.

try:
    from compressed_modules import AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
    _COMPRESSED_MODULES_AVAILABLE = True
except ImportError:
    _COMPRESSED_MODULES_AVAILABLE = False


class CompressedModelLoader:
    """Creates and populates a compressed model from a pre-built cache."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        compressor,
        codebooks: dict,
        use_compressed_modules: bool = True,
        use_mmap: bool = False,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.compressor = compressor
        self.codebooks = codebooks
        self.use_compressed_modules = use_compressed_modules and _COMPRESSED_MODULES_AVAILABLE
        self.use_mmap = use_mmap
        self.modules_replaced = 0
        self._cpu_init = False   # set True when model shell is materialized on CPU

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_and_load(self, config, model_dtype, resolver: NameResolver):
        """Full pipeline: meta → CPU → weights → compressed modules → GPU → RoPE fix.

        Large models (>VRAM capacity) are materialized on CPU first.  Compressed
        modules upload their indices/codebooks directly to GPU during replacement,
        so the final model runs GPU-accelerated regardless of initial materialization.
        Returns the ready-to-use model in eval mode.
        """
        model = self._create_on_meta(config, model_dtype)
        model = self._materialize(model, model_dtype)
        self._load_exact_weights(model, resolver)

        if self.use_compressed_modules:
            print("\n🚀 STARTING SMART COMPRESSED LOAD")
            self._replace_modules_recursive(model, resolver, self.codebooks)
            print(f"   ✅ Layer replacement complete! ({self.modules_replaced} modules)")

        # Move non-replaced parameters (norms, biases, embeddings kept exact) to the
        # inference device.  AdaptiveCodebook* modules already hold their data on GPU
        # and are unaffected by this call.
        if self._cpu_init and self.device != 'cpu':
            print(f"\n  Moving remaining parameters to {self.device}...")
            model.to(device=self.device)

        # dtype cast must come before RoPE reinit so we can detect bfloat16 inv_freq
        model.to(model_dtype)
        reinit_rope_buffers(model, config)

        if _SSM_KERNELS_AVAILABLE:
            inject_ssm_kernels(model)

        model.eval()

        # Free preloaded tensor cache — all data has been transferred to GPU
        # modules or exact-weight parameters.  Keeping it would hold gigabytes
        # of compressed numpy arrays in CPU RAM for the lifetime of the process.
        if hasattr(self.compressor, '_loaded_weights') and self.compressor._loaded_weights:
            remaining = len(self.compressor._loaded_weights)
            if remaining:
                print(f"  Freeing {remaining} unused preloaded tensors from CPU RAM...")
            self.compressor._loaded_weights.clear()
        gc.collect()
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_on_meta(self, config, model_dtype):
        """Instantiate model with no memory footprint (meta device)."""
        print(f"\nCreating model on meta device (zero memory, dtype={model_dtype})...")
        with torch.device('meta'):
            return AutoModelForCausalLM.from_config(
                config, trust_remote_code=True
            )

    def _materialize(self, model, model_dtype):
        """Materialize meta model on CPU first.

        For compressed inference the heavy Linear/Embedding weights are replaced
        by compact GPU-side modules, so materializing on CPU avoids a transient
        VRAM allocation larger than what the compressed model actually needs.
        After _replace_modules_recursive the remaining shell is moved to the
        inference device in create_and_load().
        """
        print(f"\n📌 Materializing model shell on CPU (inference device: {self.device})...")
        model.to_empty(device='cpu')
        self._cpu_init = True
        return model

    def _load_exact_weights(self, model, resolver: NameResolver):
        """Copy exact-mode tensors (norms, biases, SSM params) into model.

        Skips direct_codebook Linear/Embedding weights — those are replaced by
        _replace_modules_recursive.  Non-replaceable direct_codebook tensors
        (e.g. Conv1d weights) are decompressed and loaded here.
        """
        loaded = 0
        skipped = 0

        # Build param→parent-module map for type checks.
        parent_map: dict = {}
        for mod_name, mod in model.named_modules():
            for pname in dict(mod.named_parameters(recurse=False)):
                full = f"{mod_name}.{pname}" if mod_name else pname
                parent_map[full] = mod

        for name, param in model.named_parameters():
            if hasattr(param, '_is_compressed') or "indices" in name or "codebook" in name:
                continue

            resolved = resolver.resolve(name)
            cache_meta = self.compressor._get_compressed_tensor_data(resolved)

            # Skip weights that _replace_modules_recursive will handle.
            if cache_meta and cache_meta.get('mode') == 'direct_codebook':
                parent = parent_map.get(name)
                if isinstance(parent, (nn.Linear, nn.Embedding)) and name.endswith('.weight'):
                    skipped += 1
                    continue
                # Biases and Conv1d weights fall through to be decompressed here.

            data = self.compressor.get_tensor(resolved)
            if data is None and "lm_head.weight" in name:
                # Tied weights: lm_head shares embed_tokens.
                data = self.compressor.get_tensor(
                    resolver.resolve("model.embed_tokens.weight")
                )

            if data is not None:
                tensor = torch.from_numpy(data).to(device=self.device, dtype=param.dtype)
                param.data.copy_(tensor.reshape(param.shape))
                del tensor, data
                loaded += 1
                if hasattr(self.compressor, '_loaded_weights'):
                    self.compressor._loaded_weights.pop(resolved, None)

        print(f"  Loaded {loaded} exact tensors "
              f"(skipped {skipped} codebook tensors — Linear/Embedding will be replaced)")

    def _replace_modules_recursive(self, module, resolver: NameResolver,
                                   global_codebooks: dict, prefix: str = ""):
        """Walk model tree and swap nn.Linear / nn.Embedding with compressed variants."""
        use_gpu = (resolve_device(self.device) != 'cpu')

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                self._try_replace_linear(module, name, full_name, child,
                                         resolver, global_codebooks, use_gpu)
            elif isinstance(child, nn.Embedding):
                self._try_replace_embedding(module, name, full_name, child,
                                            resolver, global_codebooks, use_gpu)

            self._replace_modules_recursive(child, resolver, global_codebooks, full_name)

    def _try_replace_linear(self, parent, attr_name, full_name, child,
                             resolver, global_codebooks, use_gpu):
        weight_name = f"{full_name}.weight"
        resolved = resolver.resolve(weight_name)
        data = self.compressor._get_compressed_tensor_data(resolved)

        if data is None and "lm_head" in full_name:
            data = self.compressor._get_compressed_tensor_data(
                resolver.resolve("model.embed_tokens.weight")
            )

        if not (data and data['mode'] == 'direct_codebook'):
            return

        idx_file = resolver.idx_file_for(resolved, self.model_path, self.use_mmap)
        new_layer = AdaptiveCodebookLinear.from_compressed(
            weight_name, data, global_codebooks,
            use_gpu=use_gpu, use_mmap=self.use_mmap, idx_file=idx_file
        )
        if child.bias is not None:
            new_layer.bias = child.bias.data.clone().detach()
        setattr(parent, attr_name, new_layer)
        self.modules_replaced += 1
        # Free the preloaded numpy data now that it lives in the GPU module.
        if hasattr(self.compressor, '_loaded_weights'):
            self.compressor._loaded_weights.pop(resolved, None)

    def _try_replace_embedding(self, parent, attr_name, full_name, child,
                                resolver, global_codebooks, use_gpu):
        weight_name = f"{full_name}.weight"
        resolved = resolver.resolve(weight_name)
        data = self.compressor._get_compressed_tensor_data(resolved)

        if not (data and data['mode'] == 'direct_codebook'):
            return

        idx_file = resolver.idx_file_for(resolved, self.model_path, self.use_mmap)
        new_layer = AdaptiveCodebookEmbedding.from_compressed(
            weight_name, data, global_codebooks,
            use_gpu=use_gpu, use_mmap=self.use_mmap, idx_file=idx_file
        )
        setattr(parent, attr_name, new_layer)
        self.modules_replaced += 1
        # Free the preloaded numpy data now that it lives in the GPU module.
        if hasattr(self.compressor, '_loaded_weights'):
            self.compressor._loaded_weights.pop(resolved, None)
