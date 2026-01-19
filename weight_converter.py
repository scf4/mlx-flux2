from __future__ import annotations

from typing import Dict

import mlx.core as mx

from .config import Flux2Config


def convert_flux2_diffusers_weights(weights: Dict[str, mx.array], cfg: Flux2Config) -> Dict[str, mx.array]:
    out: Dict[str, mx.array] = {}

    rename = {
        "x_embedder.weight": "img_in.weight",
        "context_embedder.weight": "txt_in.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
        "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
        "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
        "proj_out.weight": "final_layer.linear.weight",
    }
    for src, dst in rename.items():
        if src in weights:
            out[dst] = weights[src]

    for i in range(cfg.depth):
        base = f"transformer_blocks.{i}.attn"
        q = weights[f"{base}.to_q.weight"]
        k = weights[f"{base}.to_k.weight"]
        v = weights[f"{base}.to_v.weight"]
        out[f"double_blocks.{i}.img_attn.qkv.weight"] = mx.concatenate([q, k, v], axis=0)
        aq = weights[f"{base}.add_q_proj.weight"]
        ak = weights[f"{base}.add_k_proj.weight"]
        av = weights[f"{base}.add_v_proj.weight"]
        out[f"double_blocks.{i}.txt_attn.qkv.weight"] = mx.concatenate([aq, ak, av], axis=0)

        out[f"double_blocks.{i}.img_attn.proj.weight"] = weights[f"{base}.to_out.0.weight"]
        out[f"double_blocks.{i}.txt_attn.proj.weight"] = weights[f"{base}.to_add_out.weight"]

        out[f"double_blocks.{i}.img_attn.norm.query_norm.scale"] = weights[f"{base}.norm_q.weight"]
        out[f"double_blocks.{i}.img_attn.norm.key_norm.scale"] = weights[f"{base}.norm_k.weight"]
        out[f"double_blocks.{i}.txt_attn.norm.query_norm.scale"] = weights[f"{base}.norm_added_q.weight"]
        out[f"double_blocks.{i}.txt_attn.norm.key_norm.scale"] = weights[f"{base}.norm_added_k.weight"]

        out[f"double_blocks.{i}.img_mlp.0.weight"] = weights[f"transformer_blocks.{i}.ff.linear_in.weight"]
        out[f"double_blocks.{i}.img_mlp.2.weight"] = weights[f"transformer_blocks.{i}.ff.linear_out.weight"]
        out[f"double_blocks.{i}.txt_mlp.0.weight"] = weights[f"transformer_blocks.{i}.ff_context.linear_in.weight"]
        out[f"double_blocks.{i}.txt_mlp.2.weight"] = weights[f"transformer_blocks.{i}.ff_context.linear_out.weight"]

    for i in range(cfg.depth_single_blocks):
        base = f"single_transformer_blocks.{i}.attn"
        out[f"single_blocks.{i}.linear1.weight"] = weights[f"{base}.to_qkv_mlp_proj.weight"]
        out[f"single_blocks.{i}.linear2.weight"] = weights[f"{base}.to_out.weight"]
        out[f"single_blocks.{i}.norm.query_norm.scale"] = weights[f"{base}.norm_q.weight"]
        out[f"single_blocks.{i}.norm.key_norm.scale"] = weights[f"{base}.norm_k.weight"]

    return out


def convert_vae_diffusers_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    out: Dict[str, mx.array] = {}

    def rename(src: str, dst: str):
        if src in weights:
            out[dst] = weights[src]

    rename("bn.running_mean", "bn.running_mean")
    rename("bn.running_var", "bn.running_var")

    rename("encoder.conv_in.weight", "encoder.conv_in.weight")
    rename("encoder.conv_in.bias", "encoder.conv_in.bias")
    rename("encoder.conv_norm_out.weight", "encoder.norm_out.weight")
    rename("encoder.conv_norm_out.bias", "encoder.norm_out.bias")
    rename("encoder.conv_out.weight", "encoder.conv_out.weight")
    rename("encoder.conv_out.bias", "encoder.conv_out.bias")

    rename("decoder.conv_in.weight", "decoder.conv_in.weight")
    rename("decoder.conv_in.bias", "decoder.conv_in.bias")
    rename("decoder.conv_norm_out.weight", "decoder.norm_out.weight")
    rename("decoder.conv_norm_out.bias", "decoder.norm_out.bias")
    rename("decoder.conv_out.weight", "decoder.conv_out.weight")
    rename("decoder.conv_out.bias", "decoder.conv_out.bias")

    rename("quant_conv.weight", "encoder.quant_conv.weight")
    rename("quant_conv.bias", "encoder.quant_conv.bias")
    rename("post_quant_conv.weight", "decoder.post_quant_conv.weight")
    rename("post_quant_conv.bias", "decoder.post_quant_conv.bias")

    for i in range(4):
        for j in range(2):
            prefix = f"encoder.down_blocks.{i}.resnets.{j}"
            dst = f"encoder.down.{i}.block.{j}"
            rename(f"{prefix}.conv1.weight", f"{dst}.conv1.weight")
            rename(f"{prefix}.conv1.bias", f"{dst}.conv1.bias")
            rename(f"{prefix}.conv2.weight", f"{dst}.conv2.weight")
            rename(f"{prefix}.conv2.bias", f"{dst}.conv2.bias")
            rename(f"{prefix}.norm1.weight", f"{dst}.norm1.weight")
            rename(f"{prefix}.norm1.bias", f"{dst}.norm1.bias")
            rename(f"{prefix}.norm2.weight", f"{dst}.norm2.weight")
            rename(f"{prefix}.norm2.bias", f"{dst}.norm2.bias")
            rename(f"{prefix}.conv_shortcut.weight", f"{dst}.nin_shortcut.weight")
            rename(f"{prefix}.conv_shortcut.bias", f"{dst}.nin_shortcut.bias")
        if i != 3:
            rename(
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight",
                f"encoder.down.{i}.downsample.conv.weight",
            )
            rename(
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias",
                f"encoder.down.{i}.downsample.conv.bias",
            )

    for j in range(2):
        src = f"encoder.mid_block.resnets.{j}"
        dst = "encoder.mid.block_1" if j == 0 else "encoder.mid.block_2"
        rename(f"{src}.conv1.weight", f"{dst}.conv1.weight")
        rename(f"{src}.conv1.bias", f"{dst}.conv1.bias")
        rename(f"{src}.conv2.weight", f"{dst}.conv2.weight")
        rename(f"{src}.conv2.bias", f"{dst}.conv2.bias")
        rename(f"{src}.norm1.weight", f"{dst}.norm1.weight")
        rename(f"{src}.norm1.bias", f"{dst}.norm1.bias")
        rename(f"{src}.norm2.weight", f"{dst}.norm2.weight")
        rename(f"{src}.norm2.bias", f"{dst}.norm2.bias")
        rename(f"{src}.conv_shortcut.weight", f"{dst}.nin_shortcut.weight")
        rename(f"{src}.conv_shortcut.bias", f"{dst}.nin_shortcut.bias")

    attn = "encoder.mid_block.attentions.0"
    dst = "encoder.mid.attn_1"
    rename(f"{attn}.group_norm.weight", f"{dst}.norm.weight")
    rename(f"{attn}.group_norm.bias", f"{dst}.norm.bias")
    rename(f"{attn}.to_q.weight", f"{dst}.q.weight")
    rename(f"{attn}.to_q.bias", f"{dst}.q.bias")
    rename(f"{attn}.to_k.weight", f"{dst}.k.weight")
    rename(f"{attn}.to_k.bias", f"{dst}.k.bias")
    rename(f"{attn}.to_v.weight", f"{dst}.v.weight")
    rename(f"{attn}.to_v.bias", f"{dst}.v.bias")
    rename(f"{attn}.to_out.0.weight", f"{dst}.proj_out.weight")
    rename(f"{attn}.to_out.0.bias", f"{dst}.proj_out.bias")

    num_res = 4
    for i in range(num_res):
        dst_i = num_res - 1 - i
        for j in range(3):
            src = f"decoder.up_blocks.{i}.resnets.{j}"
            dst = f"decoder.up.{dst_i}.block.{j}"
            rename(f"{src}.conv1.weight", f"{dst}.conv1.weight")
            rename(f"{src}.conv1.bias", f"{dst}.conv1.bias")
            rename(f"{src}.conv2.weight", f"{dst}.conv2.weight")
            rename(f"{src}.conv2.bias", f"{dst}.conv2.bias")
            rename(f"{src}.norm1.weight", f"{dst}.norm1.weight")
            rename(f"{src}.norm1.bias", f"{dst}.norm1.bias")
            rename(f"{src}.norm2.weight", f"{dst}.norm2.weight")
            rename(f"{src}.norm2.bias", f"{dst}.norm2.bias")
            rename(f"{src}.conv_shortcut.weight", f"{dst}.nin_shortcut.weight")
            rename(f"{src}.conv_shortcut.bias", f"{dst}.nin_shortcut.bias")
        if i != num_res - 1:
            rename(
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight",
                f"decoder.up.{dst_i}.upsample.conv.weight",
            )
            rename(
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias",
                f"decoder.up.{dst_i}.upsample.conv.bias",
            )

    for j in range(2):
        src = f"decoder.mid_block.resnets.{j}"
        dst = "decoder.mid.block_1" if j == 0 else "decoder.mid.block_2"
        rename(f"{src}.conv1.weight", f"{dst}.conv1.weight")
        rename(f"{src}.conv1.bias", f"{dst}.conv1.bias")
        rename(f"{src}.conv2.weight", f"{dst}.conv2.weight")
        rename(f"{src}.conv2.bias", f"{dst}.conv2.bias")
        rename(f"{src}.norm1.weight", f"{dst}.norm1.weight")
        rename(f"{src}.norm1.bias", f"{dst}.norm1.bias")
        rename(f"{src}.norm2.weight", f"{dst}.norm2.weight")
        rename(f"{src}.norm2.bias", f"{dst}.norm2.bias")
        rename(f"{src}.conv_shortcut.weight", f"{dst}.nin_shortcut.weight")
        rename(f"{src}.conv_shortcut.bias", f"{dst}.nin_shortcut.bias")

    attn = "decoder.mid_block.attentions.0"
    dst = "decoder.mid.attn_1"
    rename(f"{attn}.group_norm.weight", f"{dst}.norm.weight")
    rename(f"{attn}.group_norm.bias", f"{dst}.norm.bias")
    rename(f"{attn}.to_q.weight", f"{dst}.q.weight")
    rename(f"{attn}.to_q.bias", f"{dst}.q.bias")
    rename(f"{attn}.to_k.weight", f"{dst}.k.weight")
    rename(f"{attn}.to_k.bias", f"{dst}.k.bias")
    rename(f"{attn}.to_v.weight", f"{dst}.v.weight")
    rename(f"{attn}.to_v.bias", f"{dst}.v.bias")
    rename(f"{attn}.to_out.0.weight", f"{dst}.proj_out.weight")
    rename(f"{attn}.to_out.0.bias", f"{dst}.proj_out.bias")

    return out
