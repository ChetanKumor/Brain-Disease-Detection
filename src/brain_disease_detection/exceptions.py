"""Custom exception hierarchy for the project.

Using a small, well-defined hierarchy lets callers (the web layer, the CLI,
tests) distinguish *expected* domain errors — a bad upload, a missing model —
from genuinely unexpected failures, and lets the web layer map each to the
right HTTP status code instead of leaking stack traces to users.
"""

from __future__ import annotations


class BrainDiseaseDetectionError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(BrainDiseaseDetectionError):
    """Raised when configuration is missing, malformed, or inconsistent."""


class ImageProcessingError(BrainDiseaseDetectionError):
    """Raised when an uploaded image cannot be decoded or preprocessed.

    This is a *client* error: the input is at fault, not the server.
    """


class ModelNotFoundError(BrainDiseaseDetectionError):
    """Raised when a requested model file does not exist on disk."""


class ModelLoadError(BrainDiseaseDetectionError):
    """Raised when a model file exists but cannot be loaded."""


class UnknownDiseaseError(BrainDiseaseDetectionError):
    """Raised when a caller requests a disease that is not configured."""


class PredictionError(BrainDiseaseDetectionError):
    """Raised when inference fails after the image has been preprocessed."""
