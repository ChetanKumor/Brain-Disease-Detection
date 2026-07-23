#!/usr/bin/env python
"""Command-line entry point for training the disease classifiers.

Examples
--------
Train a single model::

    python scripts/train.py --disease brain_tumor --data-dir data/brain_tumor

Train every configured model, overriding the epoch count::

    python scripts/train.py --disease all --epochs 40

The dataset directory must contain one sub-folder per class, e.g.::

    data/brain_tumor/
        Glioma/       *.jpg
        Meningioma/   *.jpg
        NoTumor/      *.jpg
        Pituitary/    *.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when the script is run directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_disease_detection.config import load_config  # noqa: E402
from brain_disease_detection.exceptions import BrainDiseaseDetectionError  # noqa: E402
from brain_disease_detection.logger import configure_logging, get_logger  # noqa: E402

logger = get_logger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train brain-disease MRI/CT classifiers via transfer learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--disease",
        required=True,
        help="Disease key to train (e.g. 'brain_tumor'), or 'all' for every model.",
    )
    parser.add_argument("--config", default=None, help="Path to the YAML config file.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root. Defaults to <data_dir>/<disease> from the config.",
    )
    parser.add_argument(
        "--output", default=None, help="Directory to write the trained model(s)."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the configured epoch count."
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    # Import lazily so `--help` works without TensorFlow installed.
    from brain_disease_detection.training import train

    try:
        config = load_config(args.config)
        targets = list(config.diseases) if args.disease == "all" else [args.disease]
        config.disease(targets[0])  # validate early for a clean error message

        for disease_key in targets:
            result = train(
                disease_key=disease_key,
                config=config,
                data_dir=args.data_dir,
                output_dir=args.output,
                epochs=args.epochs,
            )
            logger.info(
                "[%s] test_accuracy=%.4f over %d epochs -> %s",
                disease_key,
                result.test_accuracy,
                result.epochs_trained,
                result.model_path,
            )
    except BrainDiseaseDetectionError as exc:
        logger.error("Training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
