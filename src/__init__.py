"""
src — core compression and inference modules.

  adaptive_compressor     two-pass offline compression pipeline
  model_loader            meta-device model creation and weight loading
  name_resolver           cache↔param name mapping for multi-arch support
  rope_utils              RoPE inv_freq reinitialization after meta-device load
  memory_utils            RAM / VRAM accounting helpers
  compressed_modules      AdaptiveCodebookLinear, AdaptiveCodebookEmbedding
  gpu_accelerated_functions  CUDA kernels for compressed matmul / embedding
  fast_index_manager      vectorized CPU bitstream index unpacker
  compressed_matmul_cpu   C/OpenMP kernel wrapper with gcc JIT build
  bitpack                 N-bit stream packing utilities
  compressor              base compressor, tensor classification, k-means
"""
