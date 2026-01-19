from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen3Config
from .defaults import TEXT_ENCODER_MAX_LENGTH, TEXT_ENCODER_OUTPUT_LAYERS
from .tokenizer import Qwen3Tokenizer

OUTPUT_LAYERS_QWEN3 = TEXT_ENCODER_OUTPUT_LAYERS
_MAX_OUTPUT_LAYER = max(OUTPUT_LAYERS_QWEN3)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, safe_attn: bool = False):
        super().__init__()
        self.safe_attn = safe_attn
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.n_rep = self.n_heads // self.n_kv_heads

        dim = cfg.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rope_theta = cfg.rope_theta

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        b, l, _ = x.shape
        orig_dtype = x.dtype

        q = self.q_proj(x).reshape(b, l, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)

        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=1)
            v = mx.repeat(v, self.n_rep, axis=1)

        if self.safe_attn:
            qf = q.astype(mx.float32)
            kf = k.astype(mx.float32)
            vf = v.astype(mx.float32)
            mf = None if mask is None else mask.astype(mx.float32)
            out = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=self.scale, mask=mf)
            out = out.astype(orig_dtype)
        else:
            if mask is not None and mask.dtype != q.dtype:
                mask = mask.astype(q.dtype)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, cfg: Qwen3Config, safe_attn: bool = False):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg, safe_attn=safe_attn)
        self.mlp = Qwen3MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3Backbone(nn.Module):
    def __init__(self, cfg: Qwen3Config, safe_attn: bool = False):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Qwen3Block(cfg, safe_attn=safe_attn) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        h = self.embed_tokens(input_ids)
        mask = None
        if attention_mask is not None:
            mask = build_causal_padding_mask(attention_mask, dtype=h.dtype)
        output_layers_set = set(OUTPUT_LAYERS_QWEN3)
        selected_states: List[mx.array] = []
        for i, layer in enumerate(self.layers):
            h = layer(h, mask)
            if (i + 1) in output_layers_set:
                selected_states.append(h)
            if (i + 1) >= _MAX_OUTPUT_LAYER:
                break
        if selected_states:
            return mx.concatenate(selected_states, axis=-1)
        h = self.norm(h)
        return h


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config, safe_attn: bool = False):
        super().__init__()
        self.model = Qwen3Backbone(cfg, safe_attn=safe_attn)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self.model(input_ids, attention_mask)


class Qwen3Embedder:
    def __init__(self, cfg: Qwen3Config, tokenizer: Qwen3Tokenizer, safe_attn: bool = False):
        self.cfg = cfg
        self.model = Qwen3Model(cfg, safe_attn=safe_attn)
        self.tokenizer = tokenizer
        self.max_length = TEXT_ENCODER_MAX_LENGTH

    def tokenize(self, prompts: Iterable[str]):
        ids, mask = self.tokenizer.encode_batch(prompts, max_length=self.max_length)
        return ids, mask

    def __call__(self, prompts: List[str]) -> mx.array:
        input_ids, attention_mask = self.tokenize(prompts)
        return self.model(input_ids, attention_mask)


def build_causal_padding_mask(attention_mask: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    b, l = attention_mask.shape
    idx = mx.arange(l)
    causal = idx[:, None] < idx[None, :]
    key_mask = attention_mask.astype(mx.bool_)
    invalid = causal[None, :, :] | (~key_mask[:, None, :])
    mask = invalid.astype(dtype) * mx.finfo(dtype).min
    return mask[:, None, :, :]
