"""Online inference: turning an uploaded image into a labelled prediction."""

from __future__ import annotations

from .predictor import ClassProbability, PredictionResult, Predictor

__all__ = ["ClassProbability", "Predictor", "PredictionResult"]
