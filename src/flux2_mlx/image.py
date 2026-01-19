from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


def to_rgb(img: Image.Image | List[Image.Image]):
    if isinstance(img, list):
        return [to_rgb(x) for x in img]
    return img.convert("RGB")


def cap_pixels(img: Image.Image, k: int) -> Image.Image:
    w, h = img.size
    if w * h <= k:
        return img
    scale = math.sqrt(k / (w * h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def cap_min_pixels(img: Image.Image, max_ar: float = 8.0, min_sidelength: int = 64) -> Image.Image:
    w, h = img.size
    if w < min_sidelength or h < min_sidelength:
        raise ValueError(f"Image too small: {w}x{h}")
    if w / h > max_ar or h / w > max_ar:
        raise ValueError(f"Image aspect ratio too extreme: {w}x{h}")
    return img


def center_crop_to_multiple(img: Image.Image, mult: int) -> Image.Image:
    w, h = img.size
    new_w = (w // mult) * mult
    new_h = (h // mult) * mult
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def pil_to_array(img: Image.Image) -> mx.array:
    img = img.convert("RGB")
    arr = mx.array(np.array(img, dtype=np.float32))
    return arr / 127.5 - 1.0


def array_to_pil(arr: mx.array) -> Image.Image:
    arr = mx.clip(arr, -1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).astype(mx.uint8)
    return Image.fromarray(np.array(arr), mode="RGB")


def default_prep(img: Image.Image, limit_pixels: int | None, ensure_multiple: int = 16) -> mx.array:
    img = ImageOps.exif_transpose(img)
    img = to_rgb(img)
    img = cap_min_pixels(img)
    if limit_pixels is not None:
        img = cap_pixels(img, limit_pixels)
    img = center_crop_to_multiple(img, ensure_multiple)
    return pil_to_array(img)


def load_images(paths: Iterable[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
    return images
