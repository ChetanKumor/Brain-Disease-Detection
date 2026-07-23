"""The prediction service.

:class:`Predictor` orchestrates preprocessing, model lookup, and post-processing
into a single call. It depends on an abstract :class:`ModelProvider` rather than
TensorFlow directly, which keeps the class fast to import and trivial to unit
test with an in-memory fake model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import Config
from ..data.preprocessing import ImageSource, preprocess_image
from ..exceptions import PredictionError
from ..logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Model(Protocol):
    """The minimal surface the predictor needs from a model object.

    Any Keras model satisfies this, but so does a hand-rolled stub in a test.
    """

    def predict(self, x: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray: ...


class ModelProvider(Protocol):
    """Resolves a disease key to a ready-to-use :class:`Model`.

    Implemented by :class:`brain_disease_detection.models.registry.ModelRegistry`
    in production and by a simple lambda/dict in tests.
    """

    def __call__(self, disease_key: str) -> Model: ...


@dataclass(frozen=True)
class ClassProbability:
    """A single class and the probability the model assigned to it."""

    label: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    """The structured outcome of a single prediction."""

    disease: str
    display_name: str
    predicted_label: str
    confidence: float
    probabilities: tuple[ClassProbability, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dictionary (for the API response)."""
        return {
            "disease": self.disease,
            "display_name": self.display_name,
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "probabilities": [asdict(p) for p in self.probabilities],
        }


def _to_probability_vector(raw: np.ndarray, num_classes: int) -> np.ndarray:
    """Coerce a raw model output into a valid probability distribution.

    Handles both models that already end in a softmax (output sums to 1) and
    models that emit raw logits, applying a numerically-stable softmax only when
    needed so we never softmax an already-normalised vector.

    Raises:
        PredictionError: if the output length does not match ``num_classes``.
    """
    vector = np.asarray(raw, dtype=np.float64).ravel()

    if vector.size != num_classes:
        raise PredictionError(
            f"Model produced {vector.size} outputs but the task has "
            f"{num_classes} classes. The model and configuration are out of sync."
        )

    is_distribution = bool((vector >= 0).all()) and bool(
        np.isclose(vector.sum(), 1.0, atol=1e-3)
    )
    if is_distribution:
        return vector

    # Treat the output as logits and apply a stable softmax.
    shifted = vector - vector.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


class Predictor:
    """Runs the full image-to-label inference pipeline for one request."""

    def __init__(self, config: Config, model_provider: ModelProvider) -> None:
        self._config = config
        self._model_provider = model_provider

    def predict(self, disease_key: str, image_source: ImageSource) -> PredictionResult:
        """Classify ``image_source`` for the given ``disease_key``.

        Args:
            disease_key: One of the configured disease identifiers.
            image_source: Raw image bytes, a path, or a binary stream.

        Returns:
            A :class:`PredictionResult` with the winning label, its confidence,
            and the full per-class probability breakdown.

        Raises:
            ConfigurationError: if ``disease_key`` is not configured.
            ImageProcessingError: if the image cannot be preprocessed.
            ModelNotFoundError / ModelLoadError: if the model is unavailable.
            PredictionError: if inference itself fails.
        """
        disease = self._config.disease(disease_key)  # validates the key
        tensor = preprocess_image(image_source, self._config.image)
        model = self._model_provider(disease_key)

        try:
            raw_output = model.predict(tensor, verbose=0)  # type: ignore[call-arg]
        except TypeError:
            # Fakes / non-Keras models may not accept a ``verbose`` kwarg.
            raw_output = model.predict(tensor)
        except Exception as exc:  # noqa: BLE001 - re-wrapped as a domain error
            raise PredictionError(f"Inference failed for '{disease_key}': {exc}") from exc

        probabilities = _to_probability_vector(raw_output, disease.num_classes)
        winner_index = int(np.argmax(probabilities))

        per_class = tuple(
            ClassProbability(label=label, probability=float(probabilities[i]))
            for i, label in enumerate(disease.class_labels)
        )
        result = PredictionResult(
            disease=disease.key,
            display_name=disease.display_name,
            predicted_label=disease.class_labels[winner_index],
            confidence=float(probabilities[winner_index]),
            probabilities=per_class,
        )
        logger.info(
            "Predicted %s=%s (confidence=%.3f)",
            disease.key,
            result.predicted_label,
            result.confidence,
        )
        return result
