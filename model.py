from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import Flux2Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.scale, self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=eps)
        self.key_norm = RMSNorm(dim, eps=eps)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.astype(v.dtype), k.astype(v.dtype)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        return nn.silu(x1) * x2


_TIMESTEP_FREQS_CACHE: dict[int, mx.array] = {}


def _get_timestep_freqs(dim: int, max_period: int = 10000) -> mx.array:
    key = (dim, max_period)
    if key not in _TIMESTEP_FREQS_CACHE:
        half = dim // 2
        _TIMESTEP_FREQS_CACHE[key] = mx.exp(
            -math.log(max_period)
            * mx.arange(0, half, dtype=mx.float32)
            / half
        )
    return _TIMESTEP_FREQS_CACHE[key]


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000, time_factor: float = 1000.0):
    t = t * time_factor
    freqs = _get_timestep_freqs(dim, max_period)
    args = t[:, None].astype(mx.float32) * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb.astype(t.dtype)


def compute_rope_frequencies(dim: int, theta: int) -> mx.array:
    """Precompute RoPE frequency basis (cached per dim/theta pair)."""
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even")
    scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    return 1.0 / (theta ** scale)


def rope(pos: mx.array, dim: int, theta: int, omega: mx.array | None = None) -> mx.array:
    if omega is None:
        omega = compute_rope_frequencies(dim, theta)
    out = pos.astype(mx.float32)[..., None] * omega
    cos_out = mx.cos(out)
    sin_out = mx.sin(out)
    stacked = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
    return stacked.reshape((*stacked.shape[:-1], 2, 2))


def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array) -> Tuple[mx.array, mx.array]:
    b, h, l, d = xq.shape
    xq_ = xq.reshape(b, h, l, d // 2, 1, 2)
    xk_ = xk.reshape(b, h, l, d // 2, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(b, h, l, d), xk_out.reshape(b, h, l, d)


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    pe: mx.array,
    scale: float,
    safe_attn: bool = False,
) -> mx.array:
    if safe_attn:
        qf = q.astype(mx.float32)
        kf = k.astype(mx.float32)
        vf = v.astype(mx.float32)
        qf, kf = apply_rope(qf, kf, pe)
        out = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=scale)
        out = out.astype(q.dtype)
    else:
        q, k = apply_rope(q, k, pe)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    out = out.transpose(0, 2, 1, 3)
    return out.reshape(out.shape[0], out.shape[1], -1)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self._omega_cache = [compute_rope_frequencies(d, theta) for d in axes_dim]

    def __call__(self, ids: mx.array) -> mx.array:
        parts = [
            rope(ids[..., i], self.axes_dim[i], self.theta, self._omega_cache[i])
            for i in range(len(self.axes_dim))
        ]
        emb = mx.concatenate(parts, axis=-3)
        return mx.expand_dims(emb, axis=1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(nn.silu(self.in_layer(x)))


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def __call__(self, vec: mx.array):
        out = self.lin(nn.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        chunks = mx.split(out, self.multiplier, axis=-1)
        first = (chunks[0], chunks[1], chunks[2])
        if self.is_double:
            second = (chunks[3], chunks[4], chunks[5])
            return first, second
        return first, None


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.mlp_act = SiLUActivation()

    def __call__(self, x: mx.array, pe: mx.array, mod, safe_attn: bool = False):
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = mx.split(
            self.linear1(x_mod),
            [3 * self.hidden_size],
            axis=-1,
        )
        b, l, _ = qkv.shape
        qkv = qkv.reshape(b, l, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe, self.scale, safe_attn=safe_attn)
        out = self.linear2(mx.concatenate([attn, self.mlp_act(mlp)], axis=-1))
        return x + mod_gate * out


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_mult_factor = 2
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5

        self.img_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.img_attn = SelfAttention(hidden_size, num_heads=num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.img_mlp = [
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        ]

        self.txt_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.txt_attn = SelfAttention(hidden_size, num_heads=num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.txt_mlp = [
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        ]

    def __call__(
        self,
        img: mx.array,
        txt: mx.array,
        pe: mx.array,
        pe_ctx: mx.array,
        mod_img,
        mod_txt,
        safe_attn: bool = False,
    ):
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        img_modulated = (1 + img_mod1_scale) * self.img_norm1(img) + img_mod1_shift
        img_qkv = self.img_attn.qkv(img_modulated)
        b, l, _ = img_qkv.shape
        img_qkv = img_qkv.reshape(b, l, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = (1 + txt_mod1_scale) * self.txt_norm1(txt) + txt_mod1_shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        b, l, _ = txt_qkv.shape
        txt_qkv = txt_qkv.reshape(b, l, 3, self.num_heads, -1).transpose(2, 0, 3, 1, 4)
        txt_q, txt_k, txt_v = txt_qkv[0], txt_qkv[1], txt_qkv[2]
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = mx.concatenate([txt_q, img_q], axis=2)
        k = mx.concatenate([txt_k, img_k], axis=2)
        v = mx.concatenate([txt_v, img_v], axis=2)
        pe_all = mx.concatenate([pe_ctx, pe], axis=2)

        attn = attention(q, k, v, pe_all, self.scale, safe_attn=safe_attn)
        txt_attn = attn[:, : txt_q.shape[2]]
        img_attn = attn[:, txt_q.shape[2] :]

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img_mlp_in = (1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift
        img = img + img_mod2_gate * self.img_mlp[2](self.img_mlp[1](self.img_mlp[0](img_mlp_in)))

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt_mlp_in = (1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift
        txt = txt + txt_mod2_gate * self.txt_mlp[2](self.txt_mlp[1](self.txt_mlp[0](txt_mlp_in)))
        return img, txt


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
        ]

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        mod = self.adaLN_modulation[1](self.adaLN_modulation[0](vec))
        shift, scale = mx.split(mod, 2, axis=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        return self.linear(x)


class Flux2(nn.Module):
    def __init__(self, params: Flux2Config):
        super().__init__()
        if params.hidden_size % params.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError("axes_dim must sum to head_dim")

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, disable_bias=True)

        self.double_blocks = [
            DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
            for _ in range(params.depth)
        ]
        self.single_blocks = [
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
            for _ in range(params.depth_single_blocks)
        ]

        self.double_stream_modulation_img = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.double_stream_modulation_txt = Modulation(self.hidden_size, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_size, double=False, disable_bias=True)

        self.final_layer = LastLayer(self.hidden_size, self.out_channels)
        self.safe_attn = False

    def embed_txt(self, ctx: mx.array) -> mx.array:
        """Pre-embed text context. Call once and reuse across denoising steps."""
        return self.txt_in(ctx)

    def embed_guidance(self, guidance: mx.array) -> mx.array:
        """Pre-embed guidance. Call once and reuse across denoising steps."""
        guidance_emb = timestep_embedding(guidance, 256)
        return self.guidance_in(guidance_emb)

    def __call__(
        self,
        x: mx.array,
        x_ids: mx.array,
        timesteps: mx.array,
        ctx: mx.array,
        ctx_ids: mx.array,
        guidance: mx.array | None,
        pe_x: mx.array | None = None,
        pe_ctx: mx.array | None = None,
        txt_embedded: mx.array | None = None,
        guidance_embedded: mx.array | None = None,
    ) -> mx.array:
        num_txt_tokens = ctx.shape[1]
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if self.use_guidance_embed and guidance is not None:
            if guidance_embedded is not None:
                vec = vec + guidance_embedded
            else:
                guidance_emb = timestep_embedding(guidance, 256)
                vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = txt_embedded if txt_embedded is not None else self.txt_in(ctx)

        if pe_x is None:
            pe_x = self.pe_embedder(x_ids)
        if pe_ctx is None:
            pe_ctx = self.pe_embedder(ctx_ids)

        for block in self.double_blocks:
            img, txt = block(
                img,
                txt,
                pe_x,
                pe_ctx,
                double_block_mod_img,
                double_block_mod_txt,
                safe_attn=self.safe_attn,
            )

        img = mx.concatenate([txt, img], axis=1)
        pe = mx.concatenate([pe_ctx, pe_x], axis=2)

        for block in self.single_blocks:
            img = block(img, pe, single_block_mod, safe_attn=self.safe_attn)

        img = img[:, num_txt_tokens:, :]
        img = self.final_layer(img, vec)
        return img
