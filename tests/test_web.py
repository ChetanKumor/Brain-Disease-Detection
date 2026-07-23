"""Integration tests for the Flask routes and error handling."""

from __future__ import annotations

import io


def _upload(make_png, filename: str = "scan.png"):
    return {"disease": "brain_tumor", "image": (io.BytesIO(make_png()), filename)}


def test_index_renders_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"predict-form" in response.data
    assert b"Brain Disease Detection" in response.data


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert set(body["models_available"]) == {"alzheimers", "brain_stroke", "brain_tumor"}


def test_diseases_endpoint(client):
    body = client.get("/api/diseases").get_json()
    assert len(body["diseases"]) == 3
    assert {d["key"] for d in body["diseases"]} == {
        "alzheimers",
        "brain_stroke",
        "brain_tumor",
    }


def test_api_predict_success(client, make_png):
    response = client.post(
        "/api/predict", data=_upload(make_png), content_type="multipart/form-data"
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["disease"] == "brain_tumor"
    assert body["predicted_label"] in {"Glioma", "Meningioma", "NoTumor", "Pituitary"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 4


def test_html_form_renders_result(client, make_png):
    response = client.post(
        "/", data=_upload(make_png), content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert b"result-panel" in response.data
    assert b"Prediction:" in response.data


def test_missing_disease_returns_400_json(client, make_png):
    response = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(make_png()), "scan.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["type"] == "UnknownDiseaseError"


def test_unknown_disease_returns_400(client, make_png):
    data = {"disease": "nope", "image": (io.BytesIO(make_png()), "scan.png")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_bad_extension_rejected(client):
    data = {"disease": "brain_tumor", "image": (io.BytesIO(b"hi"), "notes.txt")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_corrupt_image_rejected(client):
    data = {"disease": "brain_tumor", "image": (io.BytesIO(b"not-an-image"), "scan.png")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["type"] == "ImageProcessingError"


def test_empty_file_rejected(client):
    data = {"disease": "brain_tumor", "image": (io.BytesIO(b""), "scan.png")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_html_error_path_renders_page_not_json(client, make_png):
    data = {"disease": "", "image": (io.BytesIO(make_png()), "scan.png")}
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"error-panel" in response.data


def test_unknown_route_returns_404(client):
    assert client.get("/does-not-exist").status_code == 404
