from __future__ import annotations

import math
import time
from typing import Callable, List, Tuple

import mlx.core as mx
from PIL import Image

from .model import Flux2
from .image import default_prep


def generalized_time_snr_shift(t: mx.array, mu: float, sigma: float) -> mx.array:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def get_schedule(num_steps: int, image_seq_len: int) -> List[float]:
    mu = compute_empirical_mu(image_seq_len, num_steps)
    timesteps = mx.linspace(1.0, 0.0, num_steps + 1)
    timesteps = generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps.tolist()


def prc_txt(x: mx.array, t_coord: mx.array | None = None) -> Tuple[mx.array, mx.array]:
    l = x.shape[0]
    t = mx.arange(1) if t_coord is None else t_coord
    h = mx.arange(1)
    w = mx.arange(1)
    l_coords = mx.arange(l)
    tt, hh, ww, ll = mx.meshgrid(t, h, w, l_coords, indexing="ij")
    ids = mx.stack([tt, hh, ww, ll], axis=-1).reshape(-1, 4)
    return x, ids


def prc_img(x: mx.array, t_coord: mx.array | None = None) -> Tuple[mx.array, mx.array]:
    c, h, w = x.shape
    t = mx.arange(1) if t_coord is None else t_coord
    h_coords = mx.arange(h)
    w_coords = mx.arange(w)
    l = mx.arange(1)
    tt, hh, ww, ll = mx.meshgrid(t, h_coords, w_coords, l, indexing="ij")
    ids = mx.stack([tt, hh, ww, ll], axis=-1).reshape(-1, 4)
    tokens = x.transpose(1, 2, 0).reshape(h * w, c)
    return tokens, ids


def batched_prc_img(x: mx.array, t_coord: mx.array | None = None) -> Tuple[mx.array, mx.array]:
    toks = []
    ids = []
    for i in range(x.shape[0]):
        t_i = t_coord[i] if t_coord is not None else None
        tok, idx = prc_img(x[i], t_i)
        toks.append(tok)
        ids.append(idx)
    return mx.stack(toks), mx.stack(ids)


def batched_prc_txt(x: mx.array, t_coord: mx.array | None = None) -> Tuple[mx.array, mx.array]:
    toks = []
    ids = []
    for i in range(x.shape[0]):
        t_i = t_coord[i] if t_coord is not None else None
        tok, idx = prc_txt(x[i], t_i)
        toks.append(tok)
        ids.append(idx)
    return mx.stack(toks), mx.stack(ids)


def listed_prc_img(x_list: List[mx.array], t_coord: List[mx.array] | None = None):
    toks = []
    ids = []
    for i, x in enumerate(x_list):
        t_i = t_coord[i] if t_coord is not None else None
        tok, idx = prc_img(x, t_i)
        toks.append(tok)
        ids.append(idx)
    return toks, ids


def encode_image_refs(ae, img_ctx: List[Image.Image]):
    scale = 10
    if len(img_ctx) > 1:
        limit_pixels = 1024**2
    elif len(img_ctx) == 1:
        limit_pixels = 2024**2
    else:
        limit_pixels = None
    if not img_ctx:
        return None, None
    img_ctx_prep = [default_prep(img, limit_pixels=limit_pixels) for img in img_ctx]
    encoded_refs = []
    for img in img_ctx_prep:
        latent = ae.encode(mx.expand_dims(img, axis=0))
        latent = latent.transpose(0, 3, 1, 2)[0]
        encoded_refs.append(latent)
    t_off = [mx.array([scale * (1 + i)], dtype=mx.int32) for i in range(len(encoded_refs))]
    ref_tokens, ref_ids = listed_prc_img(encoded_refs, t_coord=t_off)
    ref_tokens = mx.concatenate(ref_tokens, axis=0)
    ref_ids = mx.concatenate(ref_ids, axis=0)
    ref_tokens = mx.expand_dims(ref_tokens, axis=0)
    ref_ids = mx.expand_dims(ref_ids, axis=0)
    return ref_tokens, ref_ids


def scatter_ids(x: mx.array, x_ids: mx.array) -> List[mx.array]:
    out_list = []
    for data, pos in zip(x, x_ids):
        t_ids = pos[:, 0].astype(mx.int32)
        h_ids = pos[:, 1].astype(mx.int32)
        w_ids = pos[:, 2].astype(mx.int32)
        t_ids_cmpr = compress_time(t_ids)
        t = int(mx.max(t_ids_cmpr).item()) + 1
        h = int(mx.max(h_ids).item()) + 1
        w = int(mx.max(w_ids).item()) + 1
        flat_ids = t_ids_cmpr * (h * w) + h_ids * w + w_ids
        c = data.shape[1]
        out = mx.zeros((t * h * w, c), dtype=data.dtype)
        indices = mx.broadcast_to(flat_ids[:, None], (flat_ids.shape[0], c))
        out = mx.put_along_axis(out, indices, data, axis=0)
        out = out.reshape(t, h, w, c).transpose(3, 0, 1, 2)
        out_list.append(mx.expand_dims(out, axis=0))
    return out_list


def compress_time(t_ids: mx.array) -> mx.array:
    t_min = mx.min(t_ids)
    t_max = mx.max(t_ids)
    mx.eval(t_min, t_max)
    if int(t_min.item()) == int(t_max.item()):
        return mx.zeros_like(t_ids)
    t_list = [int(v) for v in t_ids.tolist()]
    uniq = sorted(set(t_list))
    remap = {val: i for i, val in enumerate(uniq)}
    mapped = [remap[v] for v in t_list]
    return mx.array(mapped, dtype=t_ids.dtype)


def denoise(
    model: Flux2,
    img: mx.array,
    img_ids: mx.array,
    txt: mx.array,
    txt_ids: mx.array,
    timesteps: List[float],
    guidance: float,
    img_cond_seq: mx.array | None = None,
    img_cond_seq_ids: mx.array | None = None,
    log_fn: Callable[[int, float, float, mx.array, mx.array], None] | None = None,
    pe_x: mx.array | None = None,
    pe_ctx: mx.array | None = None,
    model_fn=None,
    step_times: List[float] | None = None,
    txt_embedded: mx.array | None = None,
    guidance_embedded: mx.array | None = None,
    eval_freq: int = 1,
) -> mx.array:
    if model_fn is None:
        model_fn = model
    guidance_vec = mx.full((img.shape[0],), guidance, dtype=img.dtype)
    if img_cond_seq is not None:
        img_input_ids = mx.concatenate([img_ids, img_cond_seq_ids], axis=1)
    else:
        img_input_ids = img_ids
    if pe_x is None:
        pe_x = model.pe_embedder(img_input_ids)
    if pe_ctx is None:
        pe_ctx = model.pe_embedder(txt_ids)
    if txt_embedded is None:
        txt_embedded = model.embed_txt(txt)
    if guidance_embedded is None and model.use_guidance_embed:
        guidance_embedded = model.embed_guidance(guidance_vec)
    num_steps = len(timesteps) - 1
    for step, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        step_start = time.perf_counter()
        t_vec = mx.full((img.shape[0],), t_curr, dtype=img.dtype)
        img_input = img
        if img_cond_seq is not None:
            img_input = mx.concatenate([img_input, img_cond_seq], axis=1)
        pred = model_fn(
            img_input,
            img_input_ids,
            t_vec,
            txt,
            txt_ids,
            guidance_vec,
            pe_x,
            pe_ctx,
            txt_embedded,
            guidance_embedded,
        )
        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]
        img = img + (t_prev - t_curr) * pred
        if eval_freq <= 1 or (step + 1) % eval_freq == 0 or step == num_steps - 1:
            mx.eval(img)
        step_time = time.perf_counter() - step_start
        if step_times is not None:
            step_times.append(step_time)
        if log_fn is not None:
            log_fn(step, t_curr, t_prev, img, pred)
    return img


def denoise_cfg(
    model: Flux2,
    img: mx.array,
    img_ids: mx.array,
    txt: mx.array,
    txt_ids: mx.array,
    timesteps: List[float],
    guidance: float,
    img_cond_seq: mx.array | None = None,
    img_cond_seq_ids: mx.array | None = None,
    log_fn: Callable[[int, float, float, mx.array, mx.array], None] | None = None,
    pe_x: mx.array | None = None,
    pe_ctx: mx.array | None = None,
    model_fn=None,
    step_times: List[float] | None = None,
    txt_embedded: mx.array | None = None,
    eval_freq: int = 1,
) -> mx.array:
    if model_fn is None:
        model_fn = model
    img = mx.concatenate([img, img], axis=0)
    img_ids = mx.concatenate([img_ids, img_ids], axis=0)
    if img_cond_seq is not None:
        img_cond_seq = mx.concatenate([img_cond_seq, img_cond_seq], axis=0)
        img_cond_seq_ids = mx.concatenate([img_cond_seq_ids, img_cond_seq_ids], axis=0)

    if img_cond_seq is not None:
        img_input_ids = mx.concatenate([img_ids, img_cond_seq_ids], axis=1)
    else:
        img_input_ids = img_ids
    if pe_x is None:
        pe_x = model.pe_embedder(img_input_ids)
    if pe_ctx is None:
        pe_ctx = model.pe_embedder(txt_ids)
    if txt_embedded is None:
        txt_embedded = model.embed_txt(txt)

    num_steps = len(timesteps) - 1
    for step, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        step_start = time.perf_counter()
        t_vec = mx.full((img.shape[0],), t_curr, dtype=img.dtype)
        img_input = img
        if img_cond_seq is not None:
            img_input = mx.concatenate([img_input, img_cond_seq], axis=1)
        call_fn = model_fn
        guidance_arg = None
        if model.use_guidance_embed:
            call_fn = model
        else:
            guidance_arg = mx.zeros((img.shape[0],), dtype=img.dtype)
        pred = call_fn(
            img_input,
            img_input_ids,
            t_vec,
            txt,
            txt_ids,
            guidance_arg,
            pe_x,
            pe_ctx,
            txt_embedded,
            None,
        )
        if img_cond_seq is not None:
            pred = pred[:, : img.shape[1]]
        pred_uncond, pred_cond = mx.split(pred, 2, axis=0)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        pred = mx.concatenate([pred, pred], axis=0)
        img = img + (t_prev - t_curr) * pred
        if eval_freq <= 1 or (step + 1) % eval_freq == 0 or step == num_steps - 1:
            mx.eval(img)
        step_time = time.perf_counter() - step_start
        if step_times is not None:
            step_times.append(step_time)
        if log_fn is not None:
            log_fn(step, t_curr, t_prev, img, pred)

    return mx.split(img, 2, axis=0)[0]
