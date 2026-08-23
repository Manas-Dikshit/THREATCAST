"""Backend smoke tests: health endpoint + contract schemas round-trip."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import (
    ErrorEnvelope,
    NetworkState,
    NetworkStateSequence,
    PredictionResult,
)
from datetime import datetime, timezone


def test_health():
    client = TestClient(create_app())
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_unknown_endpoint_uses_error_envelope():
    client = TestClient(create_app())
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    err = ErrorEnvelope.model_validate(r.json())
    assert err.error.code == "NOT_FOUND"


def test_network_state_round_trip():
    state = NetworkState(
        state_id="state_000001",
        timestamp_start=datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        timestamp_end=datetime(2026, 1, 1, 10, 0, 10, tzinfo=timezone.utc),
        window_seconds=10.0,
        features={"flow_count": 120, "syn_ratio": 0.42},
    )
    parsed = NetworkState.model_validate_json(state.model_dump_json())
    assert parsed.features["syn_ratio"] == 0.42
    # open feature dict: extension without schema change
    parsed.features["new_future_feature"] = 1.5


def test_sequence_defaults_match_contract():
    seq = NetworkStateSequence(sequence_id="seq_000001")
    assert seq.sequence_length == 5
    assert seq.window_seconds == 10.0
    assert seq.states == []


def test_prediction_result_bounds_and_model_info():
    now = datetime.now(timezone.utc)
    res = PredictionResult(
        prediction_id="pred_123", timestamp=now, risk_score=0.87,
        malicious_probability=0.91, confidence=0.84,
    )
    assert res.model.name == "threatcast-world-model"
    try:
        PredictionResult(prediction_id="x", timestamp=now, risk_score=1.5,
                         malicious_probability=0.0, confidence=0.0)
        raised = False
    except Exception:
        raised = True
    assert raised, "risk_score must be bounded to [0,1]"
