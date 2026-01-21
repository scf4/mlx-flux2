from __future__ import annotations

import math
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import VAEConfig


def swish(x: mx.array) -> mx.array:
    return nn.silu(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None, norm_groups: int):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.use_shortcut = in_channels != out_channels

        self.norm1 = nn.GroupNorm(norm_groups, in_channels, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, out_channels, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.use_shortcut:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.use_shortcut:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.scale = in_channels ** -0.5  # Cache attention scale
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True, pytorch_compatible=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, hgt, wdt, c = q.shape
        q = q.reshape(b, hgt * wdt, c)[:, None, :, :]
        k = k.reshape(b, hgt * wdt, c)[:, None, :, :]
        v = v.reshape(b, hgt * wdt, c)[:, None, :, :]
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.reshape(b, hgt, wdt, c)
        return x + self.proj_out(attn)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = mx.pad(x, pad, mode="constant")
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        x = x.reshape(b, h, 1, w, 1, c)
        x = mx.broadcast_to(x, (b, h, 2, w, 2, c))
        x = x.reshape(b, h * 2, w * 2, c)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
        norm_groups: int,
    ):
        super().__init__()
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, kernel_size=1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = []
        block_in = ch
        curr_res = resolution
        for i_level in range(self.num_resolutions):
            block = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, norm_groups))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, norm_groups)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, norm_groups)

        self.norm_out = nn.GroupNorm(norm_groups, block_in, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        norm_groups: int,
    ):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, norm_groups)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, norm_groups)

        self.up = []
        for i_level in reversed(range(self.num_resolutions)):
            block = []
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, norm_groups))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.append(up)
        # Reverse to match weight indexing: up[0]=finest, up[n-1]=coarsest
        self.up = self.up[::-1]

        self.norm_out = nn.GroupNorm(norm_groups, block_in, eps=1e-6, affine=True, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: VAEConfig):
        super().__init__()
        self.params = params
        self.force_upcast = params.force_upcast
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            norm_groups=params.norm_num_groups,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            norm_groups=params.norm_num_groups,
        )
        self.bn_eps = params.bn_eps
        self.bn_momentum = params.bn_momentum
        self.ps = params.ps
        self.bn = nn.BatchNorm(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.bn.train(mode=False)  # VAE is inference-only
        self._inv_norm_scale: mx.array | None = None
        self._inv_norm_mean: mx.array | None = None

    def normalize(self, z: mx.array) -> mx.array:
        return self.bn(z)

    def inv_normalize(self, z: mx.array) -> mx.array:
        if self._inv_norm_scale is None:
            self._inv_norm_scale = mx.sqrt(self.bn.running_var.reshape(1, 1, 1, -1) + self.bn_eps)
            self._inv_norm_mean = self.bn.running_mean.reshape(1, 1, 1, -1)
        return z * self._inv_norm_scale + self._inv_norm_mean

    def encode(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        if self.force_upcast and x.dtype != mx.float32:
            x = x.astype(mx.float32)
        moments = self.encoder(x)
        mean = moments[..., :self.params.z_channels]
        b, h, w, c = mean.shape
        pi, pj = self.ps
        mean = mean.reshape(b, h // pi, pi, w // pj, pj, c)
        mean = mean.transpose(0, 1, 3, 5, 2, 4)
        z = mean.reshape(b, h // pi, w // pj, c * pi * pj)
        z = self.normalize(z)
        if self.force_upcast and z.dtype != orig_dtype:
            z = z.astype(orig_dtype)
        return z

    def decode(self, z: mx.array) -> mx.array:
        orig_dtype = z.dtype
        if self.force_upcast and z.dtype != mx.float32:
            z = z.astype(mx.float32)
        z = self.inv_normalize(z)
        b, h, w, cp = z.shape
        pi, pj = self.ps
        c = cp // (pi * pj)
        z = z.reshape(b, h, w, c, pi, pj)
        z = z.transpose(0, 1, 4, 2, 5, 3)
        z = z.reshape(b, h * pi, w * pj, c)
        out = self.decoder(z)
        if self.force_upcast and out.dtype != orig_dtype:
            out = out.astype(orig_dtype)
        return out
