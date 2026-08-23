# THREATCAST API Reference (planned)

Base URL: `http://localhost:8000` · Prefix: `/api/v1`
Phase 1 implements only `/health`; all other endpoints are contracts for later backend phases.

## Conventions

- All timestamps: UTC ISO-8601.
- All scores: float `[0, 1]`.
- **Every error** returns the global envelope:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Invalid network state sequence",
    "details": {},
    "request_id": "req_123"
  }
}
```

Error codes: `INVALID_INPUT` (422), `NOT_FOUND` (404), `PAYLOAD_TOO_LARGE` (413), `JOB_FAILED` (500), `MODEL_NOT_LOADED` (503), `INTERNAL_ERROR` (500).

---

## GET /api/v1/health

Liveness/readiness probe.

- **Request**: none
- **Response** 200:
```json
{ "status": "ok", "service": "threatcast-backend" }
```
- **Errors**: none expected.

---

## POST /api/v1/ingestion/upload

Upload telemetry (PCAP / NetFlow-IPFIX / CSV flows / auth logs) and start an ingestion job.

- **Request**: `multipart/form-data` with `file` (≤ `MAX_UPLOAD_SIZE_MB`) + optional `source_type` field (`pcap|netflow|csv|authlog`, default auto-detected).
- **Response** 202:
```json
{ "job_id": "job_000123", "status": "queued" }
```
- **Errors**: `INVALID_INPUT` (unsupported type), `PAYLOAD_TOO_LARGE`.
- Example:
```powershell
curl.exe -F "file=@sample.csv" -F "source_type=csv" http://localhost:8000/api/v1/ingestion/upload
```

## GET /api/v1/ingestion/jobs/{job_id}

Ingestion job status/result.

- **Response** 200:
```json
{
  "job_id": "job_000123",
  "status": "completed",
  "dataset_id": "ds_000001",
  "states_generated": 864,
  "error": null
}
```
- **Errors**: `NOT_FOUND`.

## POST /api/v1/predict

Run world-model prediction on a network-state sequence.

- **Request** body (`application/json`, schema = `NetworkStateSequence`, CONTRACT.md §6):
```json
{
  "sequence_id": "seq_000001",
  "states": [ { "...": "NetworkState objects" } ],
  "sequence_length": 5,
  "window_seconds": 10,
  "target_state": null
}
```
- **Response** 200: `PredictionResult` (CONTRACT.md §7):
```json
{
  "prediction_id": "pred_123",
  "timestamp": "2026-08-23T10:30:00Z",
  "risk_score": 0.87,
  "malicious_probability": 0.91,
  "confidence": 0.84,
  "predicted_stage": { "id": null, "name": null, "confidence": null, "source": null },
  "future_states": [],
  "feature_contributions": [],
  "model": { "name": "threatcast-world-model", "version": "0.1.0" }
}
```
- **Errors**: `INVALID_INPUT` (wrong sequence length), `MODEL_NOT_LOADED`.

## GET /api/v1/predictions/{prediction_id}/timeline

Prediction plus its forward-simulated future states over time.

- **Response** 200:
```json
{
  "prediction_id": "pred_123",
  "risk_timeline": [
    { "step": 0, "timestamp": "2026-08-23T10:30:00Z", "risk_score": 0.87 },
    { "step": 1, "timestamp": "2026-08-23T10:30:10Z", "risk_score": 0.89 }
  ]
}
```
- **Errors**: `NOT_FOUND`.

## GET /api/v1/states/{state_id}

Fetch a stored canonical network state.

- **Response** 200: `NetworkState` object (CONTRACT.md §5).
- **Errors**: `NOT_FOUND`.

---

Interactive OpenAPI docs are served by FastAPI at `/docs` once the backend phase lands.
