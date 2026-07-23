"""Model architecture and the runtime model registry.

TensorFlow is imported lazily inside the functions/methods that need it so that
importing this sub-package (e.g. from the web layer or a test) does not pull in
TensorFlow unless a model is actually built or loaded.
"""

from __future__ import annotations

from .registry import ModelRegistry

__all__ = ["ModelRegistry"]
