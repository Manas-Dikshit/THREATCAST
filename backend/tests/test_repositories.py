"""Repository layer + prediction persistence details (explanations, futures)."""

from datetime import datetime, timedelta, timezone

import numpy as np

from app.models.tables import NetworkStateRow
from app.repositories.repositories import (
    DatasetRepository, IngestionJobRepository, ModelRepository,
    PredictionRepository, StateRepository,
)
from backend.tests.conftest import FEATURES, L


def _seq_payload(seq_id):
    rng = np.random.default_rng(31)
    t0 = datetime.now(timezone.utc)
    states = [{
        "state_id": f"state_{i}", "timestamp_start": (t0 + timedelta(seconds=10 * i)).isoformat(),
        "timestamp_end": (t0 + timedelta(seconds=10 * i + 10)).isoformat(),
        "window_seconds": 10.0,
        "features": {n: float(rng.normal()) for n in FEATURES},
        "label": None, "label_source": None,
    } for i in range(L)]
    return {"sequence_id": seq_id, "states": states,
            "sequence_length": L, "window_seconds": 10.0}


def test_dataset_job_state_roundtrip(db_session):
    ds = DatasetRepository(db_session).create(
        filename="a.csv", file_format="csv", size_bytes=123)
    job_repo = IngestionJobRepository(db_session)
    job = job_repo.create("a.csv", "csv")
    job_repo.set_status(job, "COMPLETED")
    assert job.finished_at is not None and job.status == "COMPLETED"

    now = datetime.now(timezone.utc)
    StateRepository(db_session).add(NetworkStateRow(
        state_id="s1", dataset_id=ds.id,
        timestamp_start=now, timestamp_end=now + timedelta(seconds=10),
        window_seconds=10.0, features={"flow_count": 3.0}))

    got = StateRepository(db_session).get_by_state_id("s1")
    assert got is not None and got.dataset_id == ds.id
    assert len(StateRepository(db_session).for_dataset(ds.id)) == 1


def test_prediction_persistence_full_graph(client, db_session):
    """POST /predict then verify predictions+future+explanations rows exist."""
    pred = client.post("/api/v1/predict", json=_seq_payload("persist")).json()
    repo = PredictionRepository(db_session)
    row = repo.get_by_prediction_id(pred["prediction_id"])
    assert row is not None
    assert row.model_version
    assert len(row.future_states) == 3
    assert {fs.step for fs in row.future_states} == {1, 2, 3}
    if pred["feature_contributions"]:
        assert len(row.explanations) >= 3

    rebuilt = repo.to_result(row)
    assert rebuilt.prediction_id == pred["prediction_id"]
    assert abs(rebuilt.risk_score - pred["risk_score"]) < 1e-6


def test_model_registry_upsert(db_session):
    repo = ModelRepository(db_session)
    rec = repo.upsert_active(name="threatcast-world-model", version="9.9.9",
                             artifact_path="/x", device="cpu", status="loaded",
                             sequence_length=5, prediction_horizon=3)
    assert rec.is_active and rec.loaded_at is not None
    again = repo.upsert_active(name="threatcast-world-model", version="9.9.9",
                               artifact_path="/x", device="cuda", status="unavailable",
                               sequence_length=5, prediction_horizon=3)
    assert again.id == rec.id and again.device == "cuda"
    assert ModelRepository(db_session).active().id == rec.id
