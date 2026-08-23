# THREATCAST API Reference

Base URL: `http://localhost:8000` · Prefix: `/api/v1`
Interactive OpenAPI docs: `/docs` (Swagger UI), `/redoc`.

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

Error codes: `INVALID_INPUT` (422), `NOT_FOUND` (404), `PAYLOAD_TOO_LARGE` (413),
`JOB_FAILED` (500), `MODEL_NOT_LOADED` (503), `DATABASE_ERROR` (500),
`INTERNAL_ERROR` (500).
Every response carries an `X-Request-ID` header.

---

## GET /api/v1/health

Liveness/readiness probe.

- **Response** 200:
```json
{
  "status": "ok",
  "service": "threatcast-backend",
  "version": "0.4.0",
  "database": "ok",
  "model": {
    "loaded": true,
    "status": "loaded",
    "model_version": "1.0.0",
    "device": "cpu",
    "sequence_length": 5,
    "prediction_horizon": 3,
    "error": null
  }
}
```

## GET /api/v1/models

Active world-model metadata as recorded at load time.

- **Response** 200:
```json
{
  "active": {
    "status": "loaded", "name": "threatcast-world-model", "version": "1.0.0",
    "artifact_path": ".../world_model.pt", "device": "cpu",
    "sequence_length": 5, "prediction_horizon": 3, "loaded_at": "..."
  },
  "history": []
}
```

## POST /api/v1/ingestion/upload

Upload telemetry, run the Phase 2 pipeline **and** the world model synchronously
(job completes before the response returns).

- **Request**: `multipart/form-data` with `file` (`csv|pcap|pcapng`, ≤ `MAX_UPLOAD_SIZE_MB`) + optional `source_type`.
- **Response** 202: `{ "job_id": "job_ab12cd34", "status": "queued" }`
- **Errors**: `INVALID_INPUT` (unsupported type/extension/malformed file), `PAYLOAD_TOO_LARGE`, `JOB_FAILED` (pipeline failure).
```powershell
curl.exe -F "file=@sample.csv" http://localhost:8000/api/v1/ingestion/upload
```

## GET /api/v1/ingestion/jobs/{job_id}

Ingestion job status/result. `prediction_id` links the auto-generated forecast.

- **Response** 200:
```json
{
  "job_id": "job_ab12cd34", "filename": "upload_x.csv", "format": "csv",
  "status": "completed", "dataset_id": "ds_...", "states_generated": 10,
  "sequences_generated": 6, "prediction_id": "pred_...",
  "error": null, "created_at": "...", "started_at": "...", "finished_at": "..."
}
```
Statuses: `queued → processing → completed | failed`.
- **Errors**: `NOT_FOUND`.

## POST /api/v1/predict

Run the world model on a state sequence and persist the full result.

- **Request** body = `NetworkStateSequence` (CONTRACT.md §6):
```json
{
  "sequence_id": "seq_000001",
  "states": [ { "...": "NetworkState objects" } ],
  "sequence_length": 5,
  "window_seconds": 10,
  "target_state": null
}
```
- **Response** 200: full `PredictionResult` (CONTRACT.md §7) incl.
`risk_score`, `malicious_probability`, `confidence`, `future_states[]`,
`feature_contributions[]`, `model{}`.
- **Errors**: `INVALID_INPUT`, `MODEL_NOT_LOADED`.

## GET /api/v1/predictions/{prediction_id}

Fetch a stored prediction (same shape as POST /predict).

- **Errors**: `NOT_FOUND`.

## GET /api/v1/predictions/{prediction_id}/timeline

Prediction plus forward-simulated steps for charting.

- **Response** 200:
```json
{
  "prediction_id": "pred_123", "generated_at": "...",
  "risk_timeline": [
    { "step": 0, "timestamp": "...", "risk_score": 0.87,
      "malicious_probability": 0.91, "confidence": 0.84 },
    { "step": 1, "timestamp": "...", "risk_score": 0.87,
      "malicious_probability": null, "confidence": 0.80 }
  ]
}
```
- **Errors**: `NOT_FOUND`.

## GET /api/v1/states/{state_id}

Fetch a stored canonical network state (CONTRACT.md §5 shape; `flow_summary`
aggregates are inside `features`).

- **Errors**: `NOT_FOUND`.
