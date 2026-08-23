"""Prediction endpoint: valid, invalid sequence, model unavailable, persistence."""

from datetime import datetime, timedelta, timezone

import numpy as np

from tests.conftest import FEATURES, L


def make_valid_sequence(seq_id: str = "seq_api_test"):
    rng = np.random.default_rng(21)
    t0 = datetime.now(timezone.utc)
    states = []
    for i in range(L):
        start = t0 + timedelta(seconds=10 * i)
        states.append({
            "state_id": f"state_{seq_id}_{i:06d}",
            "timestamp_start": start,
            "timestamp_end": start + timedelta(seconds=10),
            "window_seconds": 10.0,
            "features": {n: float(rng.normal()) for n in FEATURES},
            "label": None,
            "label_source": None,
        })
    return {
        "sequence_id": seq_id,
        "states": states,
        "sequence_length": L,
        "window_seconds": 10.0,
        "target_state": None,
    }


def test_predict_success_and_persistence(client):
    payload = make_valid_sequence()
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction_id"].startswith("pred_")
    for k in ("risk_score", "malicious_probability", "confidence"):
        assert 0.0 <= body[k] <= 1.0
    assert [fs["step"] for fs in body["future_states"]] == [1, 2, 3]
    assert body["predicted_stage"]["source"] is None  # never fabricated

    # persisted and retrievable with identical values
    got = client.get(f"/api/v1/predictions/{body['prediction_id']}")
    assert got.status_code == 200
    again = got.json()
    assert abs(again["risk_score"] - body["risk_score"]) < 1e-6
    assert len(again["future_states"]) == 3


def test_predict_wrong_length_rejected(client):
    payload = make_valid_sequence()
    payload["states"] = payload["states"][:3]  # too few for L=5
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "INVALID_INPUT"
    assert "INVALID_SEQUENCE" in err["message"]


def test_predict_empty_states_rejected(client):
    payload = make_valid_sequence()
    payload["states"] = []
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_INPUT"


def test_prediction_not_found_envelope(client):
    r = client.get("/api/v1/predictions/pred_does_not_exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_timeline_shape(client):
    pred = client.post("/api/v1/predict", json=make_valid_sequence("tl")).json()
    tl = client.get(f"/api/v1/predictions/{pred['prediction_id']}/timeline").json()
    assert tl["prediction_id"] == pred["prediction_id"]
    steps = tl["risk_timeline"]
    assert steps[0]["step"] == 0 and all(s["risk_score"] >= 0 for s in steps)
    timestamps = [s["timestamp"] for s in steps if s.get("timestamp")]
    assert len(timestamps) >= 2


def test_model_unavailable_returns_503(client):
    """Swap in an unloaded ModelService -> predict must fail gracefully."""
    from app.services.model_service import ModelService

    old = client.app.state.model_service
    client.app.state.model_service = ModelService(
        artifacts_dir="definitely/missing", device="cpu"
    )
    try:
        r = client.post("/api/v1/predict", json=make_valid_sequence())
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "MODEL_NOT_LOADED"

        h = client.get("/api/v1/health").json()
        assert h["model"]["loaded"] is False
    finally:
        client.app.state.model_service = old
