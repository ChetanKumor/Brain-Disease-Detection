"""The training pipeline.

Two-phase transfer learning:

1. **Feature extraction** — the backbone is frozen and only the new head is
   trained, quickly adapting the classifier to the domain.
2. **Fine-tuning** — the top backbone layers are unfrozen and training continues
   at a much lower learning rate, letting the network specialise without
   destroying the pretrained features.

Callbacks provide overfitting protection (early stopping), learning-rate
scheduling (reduce-on-plateau), and best-weight checkpointing. Metrics are
computed on a held-out test set and written to disk; nothing is fabricated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config
from ..exceptions import ConfigurationError
from ..logger import get_logger
from ..models.architecture import build_model, enable_fine_tuning
from .data_loader import build_datasets

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tensorflow as tf

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Summary of a completed training run (persisted alongside the model)."""

    disease: str
    backbone: str
    class_names: tuple[str, ...]
    epochs_trained: int
    test_accuracy: float
    test_loss: float
    model_path: str
    history: dict[str, list[float]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _build_callbacks(config: Config, checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    import tensorflow as tf

    t = config.training
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=t.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=t.reduce_lr_factor,
            patience=t.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def _merge_history(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def train(
    disease_key: str,
    config: Config,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    epochs: int | None = None,
) -> TrainingResult:
    """Train, fine-tune, evaluate, and persist a classifier for one disease.

    Args:
        disease_key: The disease to train (must be configured).
        config: The application configuration.
        data_dir: Dataset root. Defaults to ``<data_dir>/<disease_key>``.
        output_dir: Where to write the model and metrics. Defaults to
            ``config.models_dir``.
        epochs: Overrides ``config.training.epochs`` for each phase if given.

    Returns:
        A :class:`TrainingResult` describing the run.
    """
    import tensorflow as tf

    disease = config.disease(disease_key)
    training = config.training

    tf.keras.utils.set_random_seed(training.seed)

    data_dir = Path(data_dir) if data_dir else config.data_dir / disease_key
    output_dir = Path(output_dir) if output_dir else config.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = epochs or training.epochs

    logger.info("=== Training '%s' from %s ===", disease_key, data_dir)
    datasets = build_datasets(
        data_dir, config.image, training, expected_labels=disease.class_labels
    )
    if datasets.class_names != disease.class_labels:  # defensive; already checked
        raise ConfigurationError("Class label mismatch after dataset construction.")

    model, backbone = build_model(
        num_classes=disease.num_classes,
        input_shape=config.image.input_shape,
        backbone_name=training.backbone,
    )

    checkpoint_path = output_dir / f"{disease_key}_best.keras"
    callbacks = _build_callbacks(config, checkpoint_path)

    # Phase 1 — train the head with the backbone frozen.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Phase 1/2: feature extraction (backbone frozen).")
    history_head = model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=total_epochs,
        callbacks=callbacks,
    )

    # Phase 2 — unfreeze the top of the backbone and fine-tune slowly.
    enable_fine_tuning(backbone, training.fine_tune_at)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training.fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Phase 2/2: fine-tuning (top backbone layers unfrozen).")
    history_fine = model.fit(
        datasets.train,
        validation_data=datasets.validation,
        epochs=total_epochs,
        callbacks=callbacks,
    )

    logger.info("Evaluating on the held-out test set.")
    test_metrics = model.evaluate(datasets.test, return_dict=True, verbose=0)

    model_path = config.model_path(disease_key)
    model.save(model_path)
    logger.info("Saved trained model to %s", model_path)

    result = TrainingResult(
        disease=disease_key,
        backbone=training.backbone,
        class_names=datasets.class_names,
        epochs_trained=len(history_head.epoch) + len(history_fine.epoch),
        test_accuracy=float(test_metrics.get("accuracy", 0.0)),
        test_loss=float(test_metrics.get("loss", 0.0)),
        model_path=str(model_path),
        history=_merge_history(history_head, history_fine),
    )

    metrics_path = output_dir / f"{disease_key}_metrics.json"
    metrics_path.write_text(result.to_json(), encoding="utf-8")
    logger.info(
        "Done. Test accuracy=%.4f, loss=%.4f. Metrics -> %s",
        result.test_accuracy,
        result.test_loss,
        metrics_path,
    )
    return result
