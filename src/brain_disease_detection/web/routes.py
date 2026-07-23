"""HTTP routes: a server-rendered form and a JSON prediction API.

Both entry points funnel into the same validation and prediction helpers so the
browser UI and API clients behave identically.
"""

from __future__ import annotations

import os
from typing import Any

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
)
from werkzeug.datastructures import FileStorage

from ..config import Config
from ..constants import ALLOWED_IMAGE_EXTENSIONS
from ..exceptions import ImageProcessingError, UnknownDiseaseError
from ..inference.predictor import PredictionResult, Predictor
from ..logger import get_logger

logger = get_logger(__name__)


def _config() -> Config:
    return current_app.config["APP_CONFIG"]


def _predictor() -> Predictor:
    return current_app.config["PREDICTOR"]


def _disease_catalogue() -> list[dict[str, Any]]:
    """Configured diseases plus whether each model is currently available."""
    config = _config()
    availability = current_app.config["MODEL_REGISTRY"].availability()
    return [
        {
            "key": key,
            "display_name": disease.display_name,
            "class_labels": list(disease.class_labels),
            "available": availability.get(key, False),
        }
        for key, disease in config.diseases.items()
    ]


def _validate_upload(disease: str | None, image: FileStorage | None) -> None:
    """Validate the request inputs, raising domain errors on failure."""
    if not disease:
        raise UnknownDiseaseError("Please choose a disease type.")
    if disease not in _config().diseases:
        valid = ", ".join(_config().diseases)
        raise UnknownDiseaseError(f"Unknown disease '{disease}'. Choose one of: {valid}.")
    if image is None or not image.filename:
        raise ImageProcessingError("Please attach an MRI/CT scan image.")

    extension = os.path.splitext(image.filename)[1].lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ImageProcessingError(f"Unsupported file type '{extension}'. Allowed: {allowed}.")


def _run_prediction(disease: str | None, image: FileStorage | None) -> PredictionResult:
    _validate_upload(disease, image)
    assert disease is not None and image is not None  # narrowed by validation
    image_bytes = image.read()
    if not image_bytes:
        raise ImageProcessingError("The uploaded file is empty.")
    return _predictor().predict(disease, image_bytes)


def handle_domain_error(error: Exception, status_code: int) -> Response:
    """Render a domain error as JSON for API paths and as HTML otherwise."""
    message = str(error)
    logger.info("Request error (%s): %s", status_code, message)
    if request.path.startswith("/api/") or request.is_json:
        response = jsonify({"error": message, "type": type(error).__name__})
        response.status_code = status_code
        return response
    body = render_template(
        "index.html",
        diseases=_disease_catalogue(),
        error=message,
        result=None,
        selected_disease=request.form.get("disease"),
    )
    return Response(body, status=status_code)


def build_blueprint() -> Blueprint:
    """Construct the application's route blueprint."""
    bp = Blueprint("main", __name__)

    @bp.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            diseases=_disease_catalogue(),
            result=None,
            error=None,
            selected_disease=None,
        )

    @bp.post("/")
    def predict_form() -> str:
        # Domain errors bubble up to the registered error handlers.
        result = _run_prediction(request.form.get("disease"), request.files.get("image"))
        return render_template(
            "index.html",
            diseases=_disease_catalogue(),
            result=result,
            error=None,
            selected_disease=result.disease,
        )

    @bp.post("/api/predict")
    def predict_api() -> Response:
        result = _run_prediction(request.form.get("disease"), request.files.get("image"))
        return jsonify(result.to_dict())

    @bp.get("/api/diseases")
    def list_diseases() -> Response:
        return jsonify({"diseases": _disease_catalogue()})

    @bp.get("/api/health")
    def health() -> Response:
        availability = current_app.config["MODEL_REGISTRY"].availability()
        return jsonify(
            {
                "status": "ok",
                "models_available": availability,
                "any_model_ready": any(availability.values()),
            }
        )

    return bp
