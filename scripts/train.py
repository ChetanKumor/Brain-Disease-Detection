#!/usr/bin/env python
"""Convenience wrapper to run training from a repository checkout.

Equivalent to the installed ``brain-disease-train`` console command, but works
without ``pip install`` by putting ``src`` on the path first.

Examples
--------
Train one model::

    python scripts/train.py --disease brain_tumor --data-dir data/brain_tumor

Train every configured model::

    python scripts/train.py --disease all --epochs 40

The dataset directory must contain one sub-folder per class (see data/README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src`` importable when run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_disease_detection.training.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
