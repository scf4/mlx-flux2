"""Tests for image preprocessing functions."""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from flux2_mlx.image import (
    array_to_pil,
    cap_min_pixels,
    cap_pixels,
    center_crop_to_multiple,
    default_prep,
    load_images,
    pil_to_array,
    to_rgb,
)


def test_to_rgb_single():
    """Test RGB conversion for single image."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    result = to_rgb(img)
    assert result.mode == "RGB"


def test_to_rgb_list():
    """Test RGB conversion for list of images."""
    imgs = [
        Image.new("L", (100, 100)),
        Image.new("RGBA", (100, 100)),
    ]
    results = to_rgb(imgs)
    assert all(img.mode == "RGB" for img in results)


def test_cap_pixels_no_resize():
    """Test cap_pixels does nothing when under limit."""
    img = Image.new("RGB", (100, 100))
    result = cap_pixels(img, 20000)
    assert result.size == (100, 100)


def test_cap_pixels_resize():
    """Test cap_pixels resizes when over limit."""
    img = Image.new("RGB", (200, 200))
    result = cap_pixels(img, 10000)
    assert result.size[0] * result.size[1] <= 10000


def test_cap_min_pixels_valid():
    """Test cap_min_pixels passes for valid image."""
    img = Image.new("RGB", (100, 100))
    result = cap_min_pixels(img)
    assert result.size == (100, 100)


def test_cap_min_pixels_too_small():
    """Test cap_min_pixels raises for small image."""
    img = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="too small"):
        cap_min_pixels(img)


def test_cap_min_pixels_extreme_aspect():
    """Test cap_min_pixels raises for extreme aspect ratio."""
    img = Image.new("RGB", (1000, 100))
    with pytest.raises(ValueError, match="aspect ratio"):
        cap_min_pixels(img)


def test_center_crop_to_multiple():
    """Test center crop produces correct dimensions."""
    img = Image.new("RGB", (100, 100))
    result = center_crop_to_multiple(img, 16)
    assert result.size[0] % 16 == 0
    assert result.size[1] % 16 == 0


def test_pil_to_array():
    """Test PIL to MLX array conversion."""
    img = Image.new("RGB", (64, 64), color=(127, 127, 127))
    arr = pil_to_array(img)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == mx.float32
    assert mx.abs(arr).max() <= 1.0


def test_array_to_pil():
    """Test MLX array to PIL conversion."""
    arr = mx.zeros((64, 64, 3), dtype=mx.float32)
    img = array_to_pil(arr)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_default_prep():
    """Test default preprocessing pipeline."""
    img = Image.new("RGB", (256, 256))
    arr = default_prep(img, limit_pixels=None)
    assert arr.shape[0] % 16 == 0
    assert arr.shape[1] % 16 == 0
    assert arr.dtype == mx.float32


def test_load_images():
    """Test load_images properly handles file handles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.png"
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(path)

        loaded = load_images([path])
        assert len(loaded) == 1
        assert loaded[0].mode == "RGB"
        assert loaded[0].size == (64, 64)
