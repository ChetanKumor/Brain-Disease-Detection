"""Tests for configuration loading, validation, and environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain_disease_detection.config import DiseaseConfig, load_config
from brain_disease_detection.exceptions import ConfigurationError


def test_loads_expected_diseases(config):
    assert set(config.diseases) == {"alzheimers", "brain_stroke", "brain_tumor"}


def test_image_input_shape(config):
    assert config.image.input_shape == (224, 224, 3)
    assert config.image.target_size == (224, 224)


def test_brain_tumor_labels_are_alphabetical(config):
    # Regression guard for the corrected label ordering.
    assert config.disease("brain_tumor").class_labels == (
        "Glioma",
        "Meningioma",
        "NoTumor",
        "Pituitary",
    )


def test_unknown_disease_raises(config):
    with pytest.raises(ConfigurationError):
        config.disease("does_not_exist")


def test_model_path_resolves_under_models_dir(config):
    path = config.model_path("brain_tumor")
    assert path.name == "brain_tumor_model.keras"
    assert path.parent == config.models_dir


def test_duplicate_labels_rejected():
    with pytest.raises(ConfigurationError):
        DiseaseConfig(
            key="x",
            display_name="X",
            model_file="x.keras",
            class_labels=("A", "A"),
        )


def test_empty_labels_rejected():
    with pytest.raises(ConfigurationError):
        DiseaseConfig(key="x", display_name="X", model_file="x.keras", class_labels=())


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("diseases: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(bad)


def test_environment_overrides_runtime_settings(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
image: {height: 224, width: 224, channels: 3}
diseases:
  demo:
    display_name: Demo
    model_file: demo.keras
    class_labels: [A, B]
app: {port: 8000, max_upload_mb: 10}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("APP_ENV", "production")

    cfg = load_config(cfg_file)
    assert cfg.app.port == 9999
    assert cfg.app.max_upload_mb == 25
    assert cfg.app.max_upload_bytes == 25 * 1024 * 1024
    assert cfg.app.is_production is True


def test_invalid_integer_env_raises(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
diseases:
  demo: {display_name: Demo, model_file: demo.keras, class_labels: [A, B]}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PORT", "not-a-number")
    with pytest.raises(ConfigurationError):
        load_config(cfg_file)
