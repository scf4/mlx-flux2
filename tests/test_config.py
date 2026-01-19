"""Tests for configuration loading."""

import json
import tempfile
from pathlib import Path

import pytest

from flux2_mlx.config import (
    Flux2Config,
    Qwen3Config,
    VAEConfig,
    load_flux2_config,
    load_qwen3_config,
    load_vae_config,
)


def test_flux2_config_dataclass():
    """Test Flux2Config dataclass instantiation."""
    config = Flux2Config(
        in_channels=64,
        context_in_dim=1024,
        hidden_size=2048,
        num_heads=16,
        depth=12,
        depth_single_blocks=6,
        axes_dim=[16, 48, 64],
        theta=10000.0,
        mlp_ratio=4.0,
        use_guidance_embed=True,
    )
    assert config.in_channels == 64
    assert config.theta == 10000.0
    assert isinstance(config.theta, float)


def test_load_flux2_config():
    """Test loading Flux2Config from JSON."""
    data = {
        "in_channels": 64,
        "joint_attention_dim": 1024,
        "num_attention_heads": 16,
        "attention_head_dim": 128,
        "num_layers": 12,
        "num_single_layers": 6,
        "axes_dims_rope": [16, 48, 64],
        "rope_theta": 10000.0,
        "mlp_ratio": 4.0,
        "guidance_embeds": True,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        config = load_flux2_config(Path(f.name))
    assert config.in_channels == 64
    assert config.hidden_size == 16 * 128
    assert isinstance(config.theta, float)


def test_vae_config_dataclass():
    """Test VAEConfig dataclass instantiation."""
    config = VAEConfig(
        resolution=512,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        norm_num_groups=32,
        bn_eps=1e-5,
        bn_momentum=0.1,
        ps=(2, 2),
        force_upcast=True,
    )
    assert config.resolution == 512
    assert config.ps == (2, 2)


def test_load_vae_config():
    """Test loading VAEConfig from JSON."""
    data = {
        "sample_size": 512,
        "in_channels": 3,
        "block_out_channels": [128, 256, 512, 512],
        "out_channels": 3,
        "layers_per_block": 2,
        "latent_channels": 16,
        "norm_num_groups": 32,
        "batch_norm_eps": 1e-5,
        "batch_norm_momentum": 0.1,
        "patch_size": [2, 2],
        "force_upcast": True,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        config = load_vae_config(Path(f.name))
    assert config.resolution == 512
    assert config.ch_mult == [1, 2, 4, 4]


def test_qwen3_config_dataclass():
    """Test Qwen3Config dataclass instantiation."""
    config = Qwen3Config(
        model_type="qwen3",
        hidden_size=2048,
        num_hidden_layers=24,
        intermediate_size=8192,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        rms_norm_eps=1e-6,
        vocab_size=32000,
        max_position_embeddings=8192,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        rope_scaling=None,
    )
    assert config.hidden_size == 2048
    assert config.rope_theta == 1000000.0


def test_load_qwen3_config():
    """Test loading Qwen3Config from JSON."""
    data = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "intermediate_size": 8192,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32000,
        "max_position_embeddings": 8192,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        config = load_qwen3_config(Path(f.name))
    assert config.hidden_size == 2048
    assert config.head_dim == 128  # computed from hidden_size / num_attention_heads
