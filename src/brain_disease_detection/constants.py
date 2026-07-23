"""Project-wide constants and enumerations.

These are compile-time defaults that rarely change. Values that an operator may
want to tune (paths, hyper-parameters, class labels) live in
``configs/config.yaml`` and are surfaced through
:mod:`brain_disease_detection.config` instead.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# Repository root, resolved relative to this file:
#   src/brain_disease_detection/constants.py -> parents[2] == repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

# Image tensors fed to the CNNs default to 224x224 RGB — the input size used by
# the ImageNet backbones this project fine-tunes (MobileNetV2 / EfficientNet).
DEFAULT_IMAGE_HEIGHT: int = 224
DEFAULT_IMAGE_WIDTH: int = 224
DEFAULT_IMAGE_CHANNELS: int = 3

# File extensions the web layer is willing to accept for upload.
ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)


class Disease(str, Enum):
    """Canonical identifiers for the supported diagnostic tasks.

    Inheriting from :class:`str` means members compare equal to their value,
    which keeps YAML keys, form values, and API payloads interoperable while
    still giving us a typed, typo-proof handle in Python code.
    """

    ALZHEIMERS = "alzheimers"
    BRAIN_STROKE = "brain_stroke"
    BRAIN_TUMOR = "brain_tumor"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value
