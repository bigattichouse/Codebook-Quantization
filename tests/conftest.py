"""
Pytest configuration and shared fixtures for compressed model tests.
"""

import pytest
import torch
import numpy as np
import gc
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Markers & hooks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: requires a real model on disk")
    config.addinivalue_line("markers", "memory: tests that check memory usage")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run slow tests")
    parser.addoption("--model-path", type=str, default=None,
                     help="Path to test model for integration tests")
    parser.addoption("--model", type=str, default=None,
                     help="Path to model directory for layer-correctness tests")


def pytest_collection_modifyitems(config, items):
    skip_slow = not config.getoption("--run-slow")
    for item in items:
        if "slow" in item.keywords and skip_slow:
            item.add_marker(pytest.mark.skip(reason="need --run-slow option"))
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="integration test — needs real model"))


# ---------------------------------------------------------------------------
# Auto-use fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Deterministic seeds for all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture(autouse=True)
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_weight_matrix():
    """A random (256, 128) float32 matrix — typical small weight shape."""
    return np.random.randn(256, 128).astype(np.float32) * 0.02


@pytest.fixture
def sample_codebook():
    """A sorted float32 codebook of 256 entries spanning a typical weight range."""
    return np.sort(np.random.randn(256).astype(np.float32) * 0.02)


@pytest.fixture
def sample_indices():
    """uint8 indices into a 256-entry codebook."""
    return np.random.randint(0, 256, size=256 * 128, dtype=np.uint8)


@pytest.fixture
def synthetic_model_dir(tmp_path):
    """Create a minimal fake safetensors model directory for testing.

    The directory contains:
      - config.json (minimal Llama-like config)
      - tokenizer_config.json (minimal)
      - model.safetensors (two small tensors: embed_tokens.weight and a linear weight)
    """
    import json
    import struct

    model_dir = tmp_path / "test_model"
    model_dir.mkdir()

    # config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "model_type": "llama",
        "dtype": "float32",
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    # Build a minimal safetensors file with two BF16 tensors
    tensors = {
        "model.embed_tokens.weight": {"shape": [256, 64], "dtype": "BF16"},
        "model.layers.0.self_attn.q_proj.weight": {"shape": [64, 64], "dtype": "BF16"},
    }
    # Generate random BF16 data (stored as uint16)
    data_blobs = {}
    offset = 0
    header = {}
    for name, info in tensors.items():
        n_elements = 1
        for d in info["shape"]:
            n_elements *= d
        blob = np.random.randn(n_elements).astype(np.float32)
        # Convert to BF16 (truncate lower 16 bits)
        bf16 = (blob.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        data_blobs[name] = bf16
        header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + len(bf16)],
        }
        offset += len(bf16)

    header_bytes = json.dumps(header).encode("utf-8")
    st_path = model_dir / "model.safetensors"
    with open(st_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for name in tensors:
            f.write(data_blobs[name])

    return model_dir
