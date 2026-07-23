"""Tests for the model registry (availability, errors, caching)."""

from __future__ import annotations

import pytest

from brain_disease_detection.exceptions import ConfigurationError, ModelNotFoundError
from brain_disease_detection.models.registry import ModelRegistry


def test_availability_reports_all_configured_diseases(config):
    availability = ModelRegistry(config).availability()
    assert set(availability) == set(config.diseases)
    # No real weights are committed, so nothing should be reported as present.
    assert all(present is False for present in availability.values())


def test_get_missing_model_raises(config):
    with pytest.raises(ModelNotFoundError):
        ModelRegistry(config).get("brain_tumor")


def test_get_unknown_disease_raises(config):
    with pytest.raises(ConfigurationError):
        ModelRegistry(config).get("not_a_disease")


def test_callable_delegates_to_get(config):
    registry = ModelRegistry(config)
    with pytest.raises(ModelNotFoundError):
        registry("brain_tumor")


def test_models_are_cached_after_first_load(config):
    registry = ModelRegistry(config)
    calls: list[str] = []

    sentinel = object()

    def fake_load(disease_key: str):
        calls.append(disease_key)
        return sentinel

    registry._load = fake_load  # type: ignore[assignment]

    first = registry.get("brain_tumor")
    second = registry.get("brain_tumor")
    assert first is sentinel and second is sentinel
    assert calls == ["brain_tumor"]  # loaded exactly once
