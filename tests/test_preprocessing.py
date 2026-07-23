"""Tests for image decoding and preprocessing."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from brain_disease_detection.data.preprocessing import load_image, preprocess_image
from brain_disease_detection.exceptions import ImageProcessingError


def test_preprocess_shape_dtype_and_range(config, png_bytes):
    tensor = preprocess_image(png_bytes, config.image)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_preprocess_resizes_non_square(config, make_png):
    tensor = preprocess_image(make_png(size=(320, 200)), config.image)
    assert tensor.shape == (1, 224, 224, 3)


def test_rescale_is_applied(config):
    # A fully white image should map to 1.0 after the 1/255 rescale.
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (255, 255, 255)).save(buffer, format="PNG")
    tensor = preprocess_image(buffer.getvalue(), config.image)
    assert np.allclose(tensor, 1.0, atol=1e-4)


def test_load_image_converts_to_rgb(make_png):
    image = load_image(make_png())
    assert image.mode == "RGB"


def test_corrupt_bytes_raise(config):
    with pytest.raises(ImageProcessingError):
        preprocess_image(b"definitely not an image", config.image)


def test_accepts_stream_and_bytes_equivalently(config, png_bytes):
    from_bytes = preprocess_image(png_bytes, config.image)
    from_stream = preprocess_image(io.BytesIO(png_bytes), config.image)
    assert np.array_equal(from_bytes, from_stream)
