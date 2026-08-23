"""Health + model status endpoints."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "threatcast-backend"
    assert body["database"] == "ok"
    assert isinstance(body["model"], dict)


def test_health_reports_model_loaded(client):
    body = client.get("/api/v1/health").json()
    assert body["model"]["loaded"] is True
    assert body["model"]["device"] == "cpu"


def test_health_without_model(tmp_path):
    """Missing artifacts -> app still boots, health reports unavailable."""
    import os

    os.environ["ML_ARTIFACTS_DIR"] = str(tmp_path / "empty")
    from app.core.config import get_settings

    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as c:
            body = c.get("/api/v1/health").json()
            assert body["status"] == "ok"          # service alive...
            assert body["model"]["loaded"] is False  # ...but honest about the model
            assert body["model"]["reason"] in ("artifact_missing", "load_failed", "not_loaded_yet")
    finally:
        get_settings.cache_clear()


def test_models_endpoint(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["active"]["status"] == "loaded"
    assert body["active"]["name"] == "threatcast-world-model"
    assert body["active"]["prediction_horizon"] >= 1


def test_openapi_docs_served(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"])
    assert {
        "/api/v1/health", "/api/v1/models", "/api/v1/ingestion/upload",
        "/api/v1/ingestion/jobs/{job_id}", "/api/v1/predict",
        "/api/v1/predictions/{prediction_id}",
        "/api/v1/predictions/{prediction_id}/timeline",
        "/api/v1/states/{state_id}",
    } <= paths
