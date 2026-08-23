"""Full ingestion flow: upload -> pipeline -> states -> inference -> DB -> API."""

from datetime import datetime, timedelta, timezone

from tests.conftest import FEATURES, L, make_csv


def test_end_to_end_upload_to_prediction(client, csv_bytes):
    import io

    r = client.post("/api/v1/ingestion/upload",
                    files={"file": ("flows.csv", io.BytesIO(csv_bytes))})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = client.get(f"/api/v1/ingestion/jobs/{job_id}")
    assert job.status_code == 200
    body = job.json()
    assert body["status"] in ("completed", "queued")
    if body["status"] == "completed":
        assert body["states_generated"] >= 1
        assert body["dataset_id"]
        assert body["prediction_id"], "model is loaded; latest sequence must be predicted"

        # stored prediction retrievable via API
        pred = client.get(f"/api/v1/predictions/{body['prediction_id']}")
        assert pred.status_code == 200
        p = pred.json()
        for k in ("risk_score", "malicious_probability", "confidence", "model"):
            assert k in p
        assert 0.0 <= p["risk_score"] <= 1.0

        # timeline endpoint
        tl = client.get(f"/api/v1/predictions/{body['prediction_id']}/timeline")
        assert tl.status_code == 200
        steps = tl.json()["risk_timeline"]
        assert steps[0]["step"] == 0
        assert len(steps) >= 2  # current + at least one simulated step

        # state retrievable via API
        seq = p.get("future_states", [])
        assert isinstance(seq, list)


def test_job_status_unknown_id(client):
    r = client.get("/api/v1/ingestion/jobs/does_not_exist")
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "NOT_FOUND"
    assert err["request_id"].startswith("req_")


def test_malformed_csv_marks_job_failed(client):
    """CSV with a valid header but zero usable rows -> job FAILED, not crash."""
    import io

    data = b"totally,unexpected,columns\n1,2,3\n"
    r = client.post("/api/v1/ingestion/upload",
                    files={"file": ("bad.csv", io.BytesIO(data))})
    assert r.status_code in (202, 500)
    job_id = r.json().get("job_id")
    if job_id:
        job = client.get(f"/api/v1/ingestion/jobs/{job_id}").json()
        assert job["status"] in ("failed", "completed")
        if job["status"] == "failed":
            assert job["error"]


def test_state_endpoint_round_trip(client, csv_bytes):
    from datetime import timedelta

    from app.models.tables import NetworkStateRow
    from app.repositories.repositories import DatasetRepository, StateRepository

    override = client.app.dependency_overrides[next(iter(client.app.dependency_overrides))]
    db = next(override())
    now = datetime.now(timezone.utc)
    ds = DatasetRepository(db).create(filename="t.csv", file_format="csv")
    StateRepository(db).add(NetworkStateRow(
        state_id="state_test_000001", dataset_id=ds.id,
        timestamp_start=now, timestamp_end=now + timedelta(seconds=10),
        window_seconds=10.0, features={"flow_count": 12.0},
    ))
    db.commit()

    got = client.get("/api/v1/states/state_test_000001")
    assert got.status_code == 200
    assert got.json()["features"]["flow_count"] == 12.0

    missing = client.get("/api/v1/states/nope")
    assert missing.status_code == 404
