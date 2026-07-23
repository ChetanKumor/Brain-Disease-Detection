"""Runtime model registry.

Loading a Keras model is expensive, so the registry loads each model at most
once and caches it for the process lifetime. It is safe to share across the
threads of a WSGI server: loads are guarded by a lock. Instances are callable,
satisfying the :class:`~brain_disease_detection.inference.predictor.ModelProvider`
protocol, so a registry can be handed straight to a :class:`Predictor`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Dict

from ..config import Config
from ..exceptions import ModelLoadError, ModelNotFoundError
from ..logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..inference.predictor import Model

logger = get_logger(__name__)


class ModelRegistry:
    """Lazily loads and caches trained models by disease key."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._cache: Dict[str, "Model"] = {}
        self._lock = threading.Lock()

    def __call__(self, disease_key: str) -> "Model":
        """Alias for :meth:`get`, so the registry is a valid ``ModelProvider``."""
        return self.get(disease_key)

    def get(self, disease_key: str) -> "Model":
        """Return the model for ``disease_key``, loading it on first use.

        Raises:
            ConfigurationError: if ``disease_key`` is not configured.
            ModelNotFoundError: if the model file does not exist.
            ModelLoadError: if the file exists but cannot be loaded.
        """
        self._config.disease(disease_key)  # validates the key early

        cached = self._cache.get(disease_key)
        if cached is not None:
            return cached

        with self._lock:
            # Re-check inside the lock in case another thread just loaded it.
            cached = self._cache.get(disease_key)
            if cached is not None:
                return cached
            model = self._load(disease_key)
            self._cache[disease_key] = model
            return model

    def _load(self, disease_key: str) -> "Model":
        path = self._config.model_path(disease_key)
        if not path.exists():
            raise ModelNotFoundError(
                f"No trained model for '{disease_key}' at {path}. Train one with "
                f"`python scripts/train.py --disease {disease_key}` or place the "
                f"'.keras' file there."
            )
        logger.info("Loading model for '%s' from %s", disease_key, path)
        try:
            import tensorflow as tf  # noqa: PLC0415 - deferred heavy import

            model = tf.keras.models.load_model(path)
        except Exception as exc:  # noqa: BLE001 - re-wrapped as a domain error
            raise ModelLoadError(f"Failed to load model '{disease_key}' from {path}: {exc}") from exc
        logger.info("Model '%s' loaded successfully.", disease_key)
        return model

    def availability(self) -> Dict[str, bool]:
        """Map each configured disease to whether its model file exists on disk."""
        return {
            key: self._config.model_path(key).exists() for key in self._config.diseases
        }

    def warmup(self) -> None:
        """Eagerly load every available model (useful at server start-up)."""
        for key, present in self.availability().items():
            if present:
                try:
                    self.get(key)
                except (ModelNotFoundError, ModelLoadError) as exc:
                    logger.warning("Skipping warmup for '%s': %s", key, exc)
