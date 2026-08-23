"""Global error envelope conformance for every failure path."""

import io


def _err(r):
    body = r.json()
    assert "error" in body, body
    e = body["error"]
    assert set(e) == {"code", "message", "details", "request_id"}
    assert e["request_id"].startswith("req_")
    return e


def test_unknown_route_envelope(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404
    assert _err(r)["code"] == "NOT_FOUND"


def test_validation_error_envelope(client):
    r = client.post("/api/v1/predict", json={"nonsense": True})
    assert r.status_code == 422
    e = _err(r)
    assert e["code"] == "INVALID_INPUT"


def test_upload_missing_file_envelope(client):
    r = client.post("/api/v1/ingestion/upload")
    assert r.status_code == 422
    assert _err(r)["code"] == "INVALID_INPUT"


def test_unsupported_format_envelope(client):
    r = client.post("/api/v1/ingestion/upload",
                    files={"file": ("x.tar", io.BytesIO(b"junk"))})
    assert r.status_code == 422
    assert _err(r)["code"] == "UNSUPPORTED_FORMAT"


def test_model_not_loaded_envelope(client):
    from app.services.model_service import ModelService

    old = client.app.state.model_service
    client.app.state.model_service = ModelService("missing/dir", device="cpu")
    try:
        r = client.post("/api/v1/predict", json={"sequence_id": "s"})
        assert r.status_code in (422, 503)  # validation may fire first; both enveloped
        assert _err(r)["code"] in ("INVALID_INPUT", "MODEL_NOT_LOADED")
    finally:
        client.app.state.model_service = old


def test_request_id_header_present(client):
    r = client.get("/api/v1/health")
    assert r.headers.get("X-Request-ID", "").startswith("req_")
