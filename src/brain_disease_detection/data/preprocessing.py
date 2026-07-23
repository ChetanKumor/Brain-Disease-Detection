"""Image preprocessing.

Deliberately depends only on Pillow and NumPy — not TensorFlow — so it can be
imported and unit-tested in a lightweight environment, and reused by both the
training pipeline and the online inference path.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..config import ImageConfig
from ..exceptions import ImageProcessingError
from ..logger import get_logger

logger = get_logger(__name__)

# Anything that can be turned into a decoded image.
ImageSource = str | Path | bytes | bytearray | BinaryIO


def load_image(source: ImageSource) -> Image.Image:
    """Decode ``source`` into an RGB :class:`PIL.Image.Image`.

    Accepts a filesystem path, raw ``bytes``, or an open binary stream. The
    image is fully decoded eagerly so that malformed data raises here rather
    than lazily somewhere downstream.

    Raises:
        ImageProcessingError: if the bytes are not a valid, decodable image.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(bytes(source))
        image = Image.open(source)
        image.load()  # force full decode now, surfacing truncated files here
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Rejected undecodable image: %s", exc)
        raise ImageProcessingError(
            "The uploaded file could not be read as an image. Please upload a "
            "valid PNG, JPEG, or similar image file."
        ) from exc


def preprocess_image(source: ImageSource, image_config: ImageConfig) -> np.ndarray:
    """Decode, resize, and normalise an image into a model-ready batch tensor.

    Args:
        source: The raw image (path, bytes, or stream).
        image_config: Target geometry and rescale factor.

    Returns:
        A ``float32`` array of shape ``(1, height, width, channels)`` with pixel
        values scaled by ``image_config.rescale`` (``1/255`` by default).

    Raises:
        ImageProcessingError: if the image cannot be decoded or preprocessed.
    """
    image = load_image(source)

    # Pillow's resize takes (width, height); our config stores (height, width).
    pil_size = (image_config.width, image_config.height)
    if image.size != pil_size:
        image = image.resize(pil_size, Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) * np.float32(image_config.rescale)

    if array.shape != image_config.input_shape:
        raise ImageProcessingError(
            f"Preprocessed image has shape {array.shape}, expected {image_config.input_shape}."
        )

    return np.expand_dims(array, axis=0)
