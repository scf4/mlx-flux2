from __future__ import annotations

from typing import Iterable, List

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen3Config
from .defaults import TEXT_ENCODER_MAX_LENGTH, TEXT_ENCODER_OUTPUT_LAYERS
from .tokenizer import Qwen3Tokenizer

OUTPUT_LAYERS_QWEN3 = TEXT_ENCODER_OUTPUT_LAYERS
_OUTPUT_LAYERS_SET = frozenset(OUTPUT_LAYERS_QWEN3)
_MAX_OUTPUT_LAYER = max(OUTPUT_LAYERS_QWEN3)

# Cache for causal mask triangles by sequence length
_CAUSAL_MASK_CACHE: dict[int, mx.array] = {}
# Cache for final causal-only masks (no padding) by (length, dtype)
_CAUSAL_ONLY_MASK_CACHE: dict[tuple[int, mx.Dtype], mx.array] = {}
# Cache for -inf values by dtype
_NEG_INF_CACHE: dict[mx.Dtype, mx.array] = {}


class FastRMSNorm(nn.Module):
    """RMSNorm using mx.fast.rms_norm for better performance."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


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
        # Fused Q/K/V projection for efficiency (single matmul instead of 3)
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(dim, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = FastRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = FastRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rope_theta = cfg.rope_theta

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        b, l, _ = x.shape
        orig_dtype = x.dtype

        # Single fused projection then split
        qkv = self.qkv_proj(x)
        q = qkv[..., :self.q_dim].reshape(b, l, self.n_heads, self.head_dim)
        k = qkv[..., self.q_dim:self.q_dim + self.kv_dim].reshape(b, l, self.n_kv_heads, self.head_dim)
        v = qkv[..., self.q_dim + self.kv_dim:].reshape(b, l, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0)

        if self.n_rep > 1:
            # Use broadcast instead of repeat to avoid copying KV tensors
            # k,v: (b, n_kv, l, d) -> (b, n_heads, l, d) where n_heads = n_kv * n_rep
            k = mx.broadcast_to(
                k[:, :, None, :, :],
                (b, self.n_kv_heads, self.n_rep, l, self.head_dim)
            ).reshape(b, self.n_heads, l, self.head_dim)
            v = mx.broadcast_to(
                v[:, :, None, :, :],
                (b, self.n_kv_heads, self.n_rep, l, self.head_dim)
            ).reshape(b, self.n_heads, l, self.head_dim)

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
        self.input_layernorm = FastRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = FastRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

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
        self.norm = FastRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        h = self.embed_tokens(input_ids)
        mask = None
        if attention_mask is not None:
            mask = build_causal_padding_mask(attention_mask, dtype=h.dtype)
        selected_states: List[mx.array] = []
        for i, layer in enumerate(self.layers):
            h = layer(h, mask)
            if (i + 1) in _OUTPUT_LAYERS_SET:
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


def _get_neg_inf(dtype: mx.Dtype) -> mx.array:
    """Get cached -inf value for dtype."""
    if dtype not in _NEG_INF_CACHE:
        _NEG_INF_CACHE[dtype] = mx.array(mx.finfo(dtype).min, dtype=dtype)
    return _NEG_INF_CACHE[dtype]


def _get_causal_triangle(l: int) -> mx.array:
    """Get cached causal triangle mask (True = invalid/future position)."""
    if l not in _CAUSAL_MASK_CACHE:
        idx = mx.arange(l)
        _CAUSAL_MASK_CACHE[l] = idx[:, None] < idx[None, :]
    return _CAUSAL_MASK_CACHE[l]


def build_causal_only_mask(l: int, dtype: mx.Dtype) -> mx.array:
    """Build causal-only mask (no padding). Cached by (length, dtype)."""
    cache_key = (l, dtype)
    if cache_key not in _CAUSAL_ONLY_MASK_CACHE:
        causal = _get_causal_triangle(l)
        neg_inf = _get_neg_inf(dtype)
        zero = mx.array(0, dtype=dtype)
        mask = mx.where(causal, neg_inf, zero)
        _CAUSAL_ONLY_MASK_CACHE[cache_key] = mask[None, None, :, :]
    return _CAUSAL_ONLY_MASK_CACHE[cache_key]


def build_causal_padding_mask(attention_mask: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Build causal mask with padding support.

    Fast path: if attention_mask is all ones, use cached causal-only mask.
    """
    _, l = attention_mask.shape

    # Fast path: check if all positions are valid (no padding)
    # This avoids building per-batch masks when unnecessary
    all_valid = mx.all(attention_mask == 1)  # type: ignore[arg-type]
    mx.eval(all_valid)
    if all_valid.item():
        return build_causal_only_mask(l, dtype)

    # Full path: combine causal mask with padding mask
    causal = _get_causal_triangle(l)
    key_mask = attention_mask.astype(mx.bool_)
    invalid = causal[None, :, :] | (~key_mask[:, None, :])

    # Use where instead of astype * finfo.min for cleaner compilation
    neg_inf = _get_neg_inf(dtype)
    zero = mx.array(0, dtype=dtype)
    mask = mx.where(invalid, neg_inf, zero)
    return mask[:, None, :, :]
