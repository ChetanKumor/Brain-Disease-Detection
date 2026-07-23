"""Dataset construction for training.

Builds reproducible train / validation / test splits from a directory of images
laid out as ``<data_dir>/<class_name>/<image>``. Keras only offers a two-way
split, so the held-out portion is further divided into validation and test sets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import ImageConfig, TrainingConfig
from ..exceptions import ConfigurationError
from ..logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tensorflow as tf

logger = get_logger(__name__)


@dataclass
class DatasetBundle:
    """The three dataset splits plus the resolved class names."""

    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    class_names: tuple[str, ...]


def build_datasets(
    data_dir: str | Path,
    image_config: ImageConfig,
    training_config: TrainingConfig,
    expected_labels: Sequence[str] | None = None,
) -> DatasetBundle:
    """Load and split an image-classification dataset from ``data_dir``.

    Args:
        data_dir: Directory of per-class image sub-folders.
        image_config: Target image geometry and rescale factor.
        training_config: Supplies batch size, seed, and split ratios.
        expected_labels: If given, the discovered class order must match this
            exactly — a guard against a silent train/inference label mismatch.

    Returns:
        A :class:`DatasetBundle` with prefetched, rescaled datasets.

    Raises:
        ConfigurationError: if the directory is missing or labels disagree.
    """
    import tensorflow as tf

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ConfigurationError(f"Dataset directory not found: {data_dir}")

    holdout_fraction = training_config.validation_split + training_config.test_split
    if not 0.0 < holdout_fraction < 1.0:
        raise ConfigurationError(
            f"validation_split + test_split must be between 0 and 1, got {holdout_fraction}."
        )

    common = {
        "directory": str(data_dir),
        "validation_split": holdout_fraction,
        "seed": training_config.seed,
        "image_size": image_config.target_size,
        "batch_size": training_config.batch_size,
        "label_mode": "int",
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    holdout_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)

    class_names = tuple(train_ds.class_names)
    logger.info("Discovered %d classes: %s", len(class_names), class_names)
    if expected_labels is not None and class_names != tuple(expected_labels):
        raise ConfigurationError(
            "Class order discovered on disk does not match the configured labels.\n"
            f"  on disk:     {class_names}\n"
            f"  configured:  {tuple(expected_labels)}\n"
            "Fix the `class_labels` in configs/config.yaml so predictions map to "
            "the correct class."
        )

    # Divide the held-out data into validation and test by whole batches.
    holdout_batches = int(holdout_ds.cardinality().numpy())
    test_fraction = training_config.test_split / holdout_fraction
    test_batches = max(1, int(round(holdout_batches * test_fraction)))
    test_batches = min(test_batches, max(1, holdout_batches - 1))
    test_ds = holdout_ds.take(test_batches)
    val_ds = holdout_ds.skip(test_batches)

    rescale = tf.constant(image_config.rescale, dtype=tf.float32)

    def _normalise(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.cast(images, tf.float32) * rescale, labels

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(_normalise, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(_normalise, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(_normalise, num_parallel_calls=autotune).prefetch(autotune)

    logger.info(
        "Split into %d train / %d val / %d test batches.",
        int(train_ds.cardinality().numpy()),
        int(val_ds.cardinality().numpy()),
        test_batches,
    )
    return DatasetBundle(train_ds, val_ds, test_ds, class_names)
