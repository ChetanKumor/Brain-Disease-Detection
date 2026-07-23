"""Online inference: turning an uploaded image into a labelled prediction."""

from __future__ import annotations

from .predictor import ClassProbability, Predictor, PredictionResult

__all__ = ["ClassProbability", "Predictor", "PredictionResult"]
