"""Shared pytest fixtures.

The ``sys.path`` insertion lets the suite run from a bare checkout (``pytest``)
without first installing the package; the same is configured in pyproject.toml
for the standard workflow.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image  # noqa: E402

from brain_disease_detection.config import Config, load_config  # noqa: E402
from brain_disease_detection.inference.predictor import Predictor  # noqa: E402
from brain_disease_detection.web import create_app  # noqa: E402


@pytest.fixture(scope="session")
def config() -> Config:
    """The real application configuration loaded from configs/config.yaml."""
    return load_config()


def _png_bytes(size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    """A small, valid PNG image as raw bytes."""
    return _png_bytes()


@pytest.fixture
def make_png():
    """Factory returning PNG bytes of a requested size/colour."""
    return _png_bytes


class StubModel:
    """A deterministic stand-in for a Keras model.

    Returns a fixed distribution over ``num_classes`` so predictions are
    predictable in tests. Accepts (and ignores) any extra kwargs.
    """

    def __init__(self, num_classes: int, winner: int = 0) -> None:
        probs = np.full(num_classes, 0.1 / max(num_classes - 1, 1), dtype=np.float64)
        probs[winner] = 0.9
        self._probs = (probs / probs.sum()).reshape(1, -1)

    def predict(self, x, **kwargs):  # noqa: ANN001, D102
        return self._probs


@pytest.fixture
def stub_predictor(config: Config) -> Predictor:
    """A predictor backed by stub models (no real weights required)."""

    def provider(disease_key: str) -> StubModel:
        return StubModel(config.disease(disease_key).num_classes, winner=0)

    return Predictor(config, provider)


@pytest.fixture
def app(config: Config, stub_predictor: Predictor):
    """A Flask app wired to the stub predictor."""
    application = create_app(config, predictor=stub_predictor)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    """A Flask test client."""
    return app.test_client()
