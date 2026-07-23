"""Console entry point for training.

Exposed as the ``brain-disease-train`` command (see pyproject.toml) and reused by
``scripts/train.py`` for running directly from a checkout.
"""

from __future__ import annotations

import argparse

from ..config import load_config
from ..exceptions import BrainDiseaseDetectionError
from ..logger import configure_logging, get_logger

logger = get_logger("brain_disease_detection.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="brain-disease-train",
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
    parser.add_argument("--output", default=None, help="Directory to write trained model(s).")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epochs.")
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    # Imported lazily so `--help` works without TensorFlow installed.
    from . import train

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
