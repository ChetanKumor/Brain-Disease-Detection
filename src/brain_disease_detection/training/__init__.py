"""The offline training pipeline (dataset loading, training, evaluation).

TensorFlow is imported lazily so importing this package stays cheap.
"""

from __future__ import annotations

from .data_loader import DatasetBundle, build_datasets
from .trainer import TrainingResult, train

__all__ = ["DatasetBundle", "TrainingResult", "build_datasets", "train"]
