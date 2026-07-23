"""Flask application factory.

The factory pattern keeps global state out of import scope, makes the app
trivially testable (each test builds its own app, optionally with a stubbed
predictor), and lets configuration flow in explicitly.
"""

from __future__ import annotations

from typing import Optional

from flask import Flask
from werkzeug.exceptions import RequestEntityTooLarge

from ..config import Config, load_config
from ..exceptions import (
    ConfigurationError,
    ImageProcessingError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
    UnknownDiseaseError,
)
from ..inference.predictor import Predictor
from ..logger import configure_logging, get_logger
from ..models.registry import ModelRegistry
from .routes import build_blueprint

logger = get_logger(__name__)


def _register_error_handlers(app: Flask) -> None:
    """Map domain exceptions to sensible HTTP responses via the routes helper."""
    from .routes import handle_domain_error

    # Client-side / input errors -> 4xx
    app.register_error_handler(ImageProcessingError, lambda e: handle_domain_error(e, 400))
    app.register_error_handler(UnknownDiseaseError, lambda e: handle_domain_error(e, 400))
    app.register_error_handler(ConfigurationError, lambda e: handle_domain_error(e, 400))
    app.register_error_handler(
        RequestEntityTooLarge,
        lambda e: handle_domain_error(
            ImageProcessingError("The uploaded file is too large."), 413
        ),
    )
    # Server / availability errors -> 5xx
    app.register_error_handler(ModelNotFoundError, lambda e: handle_domain_error(e, 503))
    app.register_error_handler(ModelLoadError, lambda e: handle_domain_error(e, 500))
    app.register_error_handler(PredictionError, lambda e: handle_domain_error(e, 500))


def create_app(
    config: Optional[Config] = None,
    predictor: Optional[Predictor] = None,
) -> Flask:
    """Build and configure a Flask application.

    Args:
        config: Application configuration. Loaded from disk if omitted.
        predictor: An inference service. A real one backed by
            :class:`ModelRegistry` is created if omitted; tests can inject a
            stub so the web layer can be exercised without any model files.

    Returns:
        A configured :class:`flask.Flask` instance.
    """
    config = config or load_config()
    configure_logging(config.app.log_level)

    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.app.secret_key
    app.config["MAX_CONTENT_LENGTH"] = config.app.max_upload_bytes
    app.config["APP_CONFIG"] = config

    registry = ModelRegistry(config)
    app.config["MODEL_REGISTRY"] = registry
    app.config["PREDICTOR"] = predictor or Predictor(config, registry)

    _register_error_handlers(app)
    app.register_blueprint(build_blueprint())

    logger.info(
        "Application initialised (env=%s, models available=%s).",
        config.app.env,
        registry.availability(),
    )
    return app
