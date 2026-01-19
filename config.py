from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Flux2Config:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    mlp_ratio: float
    use_guidance_embed: bool


@dataclass
class VAEConfig:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    norm_num_groups: int
    bn_eps: float
    bn_momentum: float
    ps: tuple[int, int]
    force_upcast: bool


@dataclass
class Qwen3Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    tie_word_embeddings: bool
    rope_scaling: dict | None


def load_flux2_config(path: Path) -> Flux2Config:
    data = json.loads(path.read_text())
    return Flux2Config(
        in_channels=int(data["in_channels"]),
        context_in_dim=int(data["joint_attention_dim"]),
        hidden_size=int(data["num_attention_heads"] * data["attention_head_dim"]),
        num_heads=int(data["num_attention_heads"]),
        depth=int(data["num_layers"]),
        depth_single_blocks=int(data["num_single_layers"]),
        axes_dim=list(data["axes_dims_rope"]),
        theta=int(data["rope_theta"]),
        mlp_ratio=float(data["mlp_ratio"]),
        use_guidance_embed=bool(data.get("guidance_embeds", False)),
    )


def load_vae_config(path: Path) -> VAEConfig:
    data = json.loads(path.read_text())
    return VAEConfig(
        resolution=int(data["sample_size"]),
        in_channels=int(data["in_channels"]),
        ch=int(data["block_out_channels"][0]),
        out_ch=int(data["out_channels"]),
        ch_mult=[int(x) // int(data["block_out_channels"][0]) for x in data["block_out_channels"]],
        num_res_blocks=int(data["layers_per_block"]),
        z_channels=int(data["latent_channels"]),
        norm_num_groups=int(data["norm_num_groups"]),
        bn_eps=float(data["batch_norm_eps"]),
        bn_momentum=float(data["batch_norm_momentum"]),
        ps=(int(data["patch_size"][0]), int(data["patch_size"][1])),
        force_upcast=bool(data.get("force_upcast", False)),
    )


def load_qwen3_config(path: Path) -> Qwen3Config:
    data = json.loads(path.read_text())
    head_dim = int(data.get("head_dim", int(data["hidden_size"]) // int(data["num_attention_heads"])))
    return Qwen3Config(
        model_type=str(data["model_type"]),
        hidden_size=int(data["hidden_size"]),
        num_hidden_layers=int(data["num_hidden_layers"]),
        intermediate_size=int(data["intermediate_size"]),
        num_attention_heads=int(data["num_attention_heads"]),
        num_key_value_heads=int(data["num_key_value_heads"]),
        head_dim=head_dim,
        rms_norm_eps=float(data["rms_norm_eps"]),
        vocab_size=int(data["vocab_size"]),
        max_position_embeddings=int(data["max_position_embeddings"]),
        rope_theta=float(data["rope_theta"]),
        tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
        rope_scaling=data.get("rope_scaling"),
    )
