"""Typed, layered configuration.

Configuration is resolved in two layers:

1. **Defaults from YAML** (``configs/config.yaml``) — model definitions, class
   labels, image geometry, and training hyper-parameters. These are
   version-controlled and reviewed like code.
2. **Environment overrides** — deployment-specific and secret values (host,
   port, secret key, upload limits) read from the environment, so the same
   image can run in development and production without code changes.

Everything is exposed as frozen dataclasses, giving editors and ``mypy`` full
knowledge of the shape of the configuration and preventing accidental mutation
at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    PROJECT_ROOT,
)
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class ImageConfig:
    """Geometry and normalisation applied to every input image."""

    height: int = DEFAULT_IMAGE_HEIGHT
    width: int = DEFAULT_IMAGE_WIDTH
    channels: int = DEFAULT_IMAGE_CHANNELS
    rescale: float = 1.0 / 255.0

    @property
    def target_size(self) -> tuple[int, int]:
        """``(height, width)`` tuple used when resizing with Pillow."""
        return self.height, self.width

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """``(height, width, channels)`` tensor shape expected by the model."""
        return self.height, self.width, self.channels


@dataclass(frozen=True)
class DiseaseConfig:
    """Everything the system needs to know about one diagnostic task."""

    key: str
    display_name: str
    model_file: str
    class_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.class_labels:
            raise ConfigurationError(
                f"Disease '{self.key}' must define at least one class label."
            )
        if len(set(self.class_labels)) != len(self.class_labels):
            raise ConfigurationError(
                f"Disease '{self.key}' has duplicate class labels: "
                f"{self.class_labels}"
            )

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters for the transfer-learning training pipeline."""

    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5
    fine_tune_at: int = 100
    validation_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    seed: int = 42
    backbone: str = "MobileNetV2"


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the web application (mostly environment-driven)."""

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: str = "change-me"
    max_upload_mb: int = 10
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@dataclass(frozen=True)
class Config:
    """The fully-resolved application configuration."""

    image: ImageConfig
    training: TrainingConfig
    app: AppConfig
    diseases: Mapping[str, DiseaseConfig]
    models_dir: Path
    data_dir: Path
    _label_index: dict[str, DiseaseConfig] = field(default_factory=dict, repr=False)

    def disease(self, key: str) -> DiseaseConfig:
        """Return the :class:`DiseaseConfig` for ``key``.

        Raises:
            ConfigurationError: if the disease is not configured.
        """
        try:
            return self.diseases[key]
        except KeyError as exc:
            valid = ", ".join(sorted(self.diseases)) or "(none configured)"
            raise ConfigurationError(
                f"Unknown disease '{key}'. Configured diseases: {valid}."
            ) from exc

    def model_path(self, key: str) -> Path:
        """Absolute path to the trained model file for ``key``."""
        return self.models_dir / self.disease(key).model_file


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Expected an integer for '{name}', got {value!r}.") from exc


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative path against the project root."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _parse_diseases(raw: Mapping[str, Any]) -> dict[str, DiseaseConfig]:
    if not raw:
        raise ConfigurationError("Configuration must define at least one disease.")

    diseases: dict[str, DiseaseConfig] = {}
    for key, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ConfigurationError(f"Disease '{key}' must be a mapping, got {type(spec).__name__}.")
        try:
            diseases[key] = DiseaseConfig(
                key=key,
                display_name=str(spec["display_name"]),
                model_file=str(spec["model_file"]),
                class_labels=tuple(str(label) for label in spec["class_labels"]),
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"Disease '{key}' is missing required field {exc}."
            ) from exc
    return diseases


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML configuration {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"Configuration root must be a mapping, got {type(data).__name__}.")
    return dict(data)


def _build_app_config(raw: Mapping[str, Any]) -> AppConfig:
    """Merge YAML ``app`` defaults with environment overrides."""
    defaults = AppConfig()
    return AppConfig(
        env=os.getenv("APP_ENV", raw.get("env", defaults.env)),
        host=os.getenv("HOST", raw.get("host", defaults.host)),
        port=_as_int(os.getenv("PORT", raw.get("port", defaults.port)), "PORT"),
        secret_key=os.getenv("SECRET_KEY", raw.get("secret_key", defaults.secret_key)),
        max_upload_mb=_as_int(
            os.getenv("MAX_UPLOAD_MB", raw.get("max_upload_mb", defaults.max_upload_mb)),
            "MAX_UPLOAD_MB",
        ),
        log_level=os.getenv("LOG_LEVEL", raw.get("log_level", defaults.log_level)),
    )


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate the application configuration.

    Args:
        path: Path to a YAML config file. Defaults to the ``CONFIG_PATH``
            environment variable, then ``configs/config.yaml``.

    Returns:
        A fully-resolved, immutable :class:`Config`.

    Raises:
        ConfigurationError: if the file is missing, malformed, or inconsistent.
    """
    config_path = Path(path) if path is not None else Path(
        os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    )
    raw = _load_yaml(config_path)

    image_raw = raw.get("image", {})
    image = ImageConfig(
        height=_as_int(image_raw.get("height", DEFAULT_IMAGE_HEIGHT), "image.height"),
        width=_as_int(image_raw.get("width", DEFAULT_IMAGE_WIDTH), "image.width"),
        channels=_as_int(image_raw.get("channels", DEFAULT_IMAGE_CHANNELS), "image.channels"),
        rescale=float(image_raw.get("rescale", 1.0 / 255.0)),
    )

    training_raw = raw.get("training", {})
    training_defaults = TrainingConfig()
    training = TrainingConfig(
        batch_size=_as_int(training_raw.get("batch_size", training_defaults.batch_size), "batch_size"),
        epochs=_as_int(training_raw.get("epochs", training_defaults.epochs), "epochs"),
        learning_rate=float(training_raw.get("learning_rate", training_defaults.learning_rate)),
        fine_tune_learning_rate=float(
            training_raw.get("fine_tune_learning_rate", training_defaults.fine_tune_learning_rate)
        ),
        fine_tune_at=_as_int(training_raw.get("fine_tune_at", training_defaults.fine_tune_at), "fine_tune_at"),
        validation_split=float(training_raw.get("validation_split", training_defaults.validation_split)),
        test_split=float(training_raw.get("test_split", training_defaults.test_split)),
        early_stopping_patience=_as_int(
            training_raw.get("early_stopping_patience", training_defaults.early_stopping_patience),
            "early_stopping_patience",
        ),
        reduce_lr_patience=_as_int(
            training_raw.get("reduce_lr_patience", training_defaults.reduce_lr_patience),
            "reduce_lr_patience",
        ),
        reduce_lr_factor=float(training_raw.get("reduce_lr_factor", training_defaults.reduce_lr_factor)),
        seed=_as_int(training_raw.get("seed", training_defaults.seed), "seed"),
        backbone=str(training_raw.get("backbone", training_defaults.backbone)),
    )

    diseases = _parse_diseases(raw.get("diseases", {}))
    app = _build_app_config(raw.get("app", {}))

    paths_raw = raw.get("paths", {})
    models_dir = _resolve_path(os.getenv("MODELS_DIR", paths_raw.get("models_dir", "models")))
    data_dir = _resolve_path(os.getenv("DATA_DIR", paths_raw.get("data_dir", "data")))

    return Config(
        image=image,
        training=training,
        app=app,
        diseases=diseases,
        models_dir=models_dir,
        data_dir=data_dir,
        _label_index={key: cfg for key, cfg in diseases.items()},
    )
