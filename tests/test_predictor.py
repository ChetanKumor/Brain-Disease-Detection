"""Tests for the prediction service, including output post-processing."""

from __future__ import annotations

import numpy as np
import pytest

from brain_disease_detection.exceptions import PredictionError
from brain_disease_detection.inference.predictor import Predictor


class _FixedModel:
    def __init__(self, output: np.ndarray) -> None:
        self._output = output

    def predict(self, x, **kwargs):
        return self._output


def _predictor(config, output: np.ndarray) -> Predictor:
    return Predictor(config, lambda key: _FixedModel(output))


def test_softmax_output_is_passed_through(config, png_bytes):
    output = np.array([[0.1, 0.2, 0.65, 0.05]])  # already a distribution
    result = _predictor(config, output).predict("brain_tumor", png_bytes)
    assert result.predicted_label == "NoTumor"
    assert result.confidence == pytest.approx(0.65, abs=1e-6)


def test_logits_are_softmaxed(config, png_bytes):
    output = np.array([[2.0, 1.0, 0.1, 0.0]])  # logits -> not a distribution
    result = _predictor(config, output).predict("brain_tumor", png_bytes)
    total = sum(p.probability for p in result.probabilities)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert result.predicted_label == "Glioma"  # highest logit


def test_probabilities_align_with_labels(config, png_bytes):
    output = np.array([[0.7, 0.1, 0.1, 0.1]])
    result = _predictor(config, output).predict("brain_tumor", png_bytes)
    labels = [p.label for p in result.probabilities]
    assert labels == list(config.disease("brain_tumor").class_labels)


def test_class_count_mismatch_raises(config, png_bytes):
    output = np.array([[0.5, 0.5]])  # 2 outputs for a 4-class task
    with pytest.raises(PredictionError):
        _predictor(config, output).predict("brain_tumor", png_bytes)


def test_to_dict_is_json_friendly(config, png_bytes):
    output = np.array([[0.1, 0.2, 0.65, 0.05]])
    payload = _predictor(config, output).predict("brain_tumor", png_bytes).to_dict()
    assert payload["disease"] == "brain_tumor"
    assert payload["predicted_label"] == "NoTumor"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert len(payload["probabilities"]) == 4
    assert set(payload["probabilities"][0]) == {"label", "probability"}


def test_model_without_verbose_kwarg_still_works(config, png_bytes):
    class NoVerbose:
        def predict(self, x):  # note: no **kwargs
            return np.array([[0.25, 0.25, 0.25, 0.25]])

    result = Predictor(config, lambda k: NoVerbose()).predict("brain_tumor", png_bytes)
    assert result.confidence == pytest.approx(0.25, abs=1e-6)
