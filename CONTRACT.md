# THREATCAST GLOBAL CONTRACT

> **DO NOT MODIFY GLOBAL CONTRACTS WITHOUT FIRST CHECKING EXISTING ARCHITECTURE.**
> This document is the single source of truth for cross-module interfaces.
> Later phases consume these contracts; changing them breaks every dependent module.

## 1. Technology versions

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| FastAPI | >= 0.110 |
| Pydantic | 2.x (v2 API only) |
| SQLAlchemy | 2.x |
| Alembic | 1.x |
| PostgreSQL | 16 |
| PyTorch | >= 2.1 (CUDA when available) |
| scikit-learn | >= 1.4 |
| NumPy / Pandas | latest stable |
| Scapy | >= 2.5 |
| Node.js | 20 LTS |
| React / TypeScript / Vite | React 18, TS 5.x, Vite 5.x |
| Recharts | 2.x |
| pytest / Vitest | pytest >= 8, Vitest >= 1 |

## 2. Ports and API prefix

| Service  | Port |
|----------|------|
| Frontend | **5173** |
| Backend  | **8000** |
| PostgreSQL | **5432** |

API prefix: `/api/v1`. No alternate default ports are permitted.

## 3. Environment variables

Defined in `.env.example`; loaded via `pydantic-settings` in backend, plain parsing elsewhere.

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_ENV` | `development` | runtime environment |
| `BACKEND_HOST` | `0.0.0.0` | backend bind host |
| `BACKEND_PORT` | `8000` | backend port |
| `FRONTEND_PORT` | `5173` | Vite dev server / mapped port |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `DATABASE_URL` | `postgresql://threatcast:threatcast@localhost:5432/threatcast` | DB DSN |
| `ML_MODEL_PATH` | `./ml/artifacts/world_model.pt` | trained model artifact |
| `ML_METADATA_PATH` | `./ml/artifacts/model_metadata.json` | training metadata |
| `ML_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` |
| `ML_SEQUENCE_LENGTH` | `5` | states per input sequence |
| `ML_PREDICTION_HORIZON` | `3` | K forward-simulated steps |
| `TIME_WINDOW_SECONDS` | `10` | network-state window size |
| `MAX_UPLOAD_SIZE_MB` | `500` | upload limit |
| `LOG_LEVEL` | `INFO` | log verbosity |

No real secrets ever enter the repository.

## 4. Repository ownership

| Module | Owner path | Responsibility |
|--------|-----------|----------------|
| M1 — Ingestion + feature extraction + time windows | `data_pipeline/` | raw telemetry → canonical features → windowed states |
| M2 — ML / world model / training / evaluation | `ml/` | state sequences → learned dynamics → predictions |
| M3 — Backend / database / REST API / serving | `backend/` | persistence, REST API, model serving facade |
| M4 — Frontend dashboard | `frontend/` | defender UI only |
| M5 — Integration / security / testing / devops | `integration/`, `security/`, `tests/`, `docker/`, `docs/` | cross-cutting concerns |

Rules: no module silently duplicates another; cross-module data flows exclusively through the schemas below.

## 5. NetworkState schema (canonical)

```json
{
  "state_id": "state_000001",
  "timestamp_start": "2026-01-01T10:00:00Z",
  "timestamp_end": "2026-01-01T10:00:10Z",
  "window_seconds": 10,
  "features": {
    "flow_count": 120,
    "packet_count": 1820,
    "byte_count": 98231,
    "syn_ratio": 0.42,
    "ack_ratio": 0.31,
    "rst_ratio": 0.04,
    "mean_iat": 0.031,
    "iat_variance": 0.012,
    "ttl_mean": 61.2,
    "ttl_variance": 4.1,
    "tcp_window_mean": 64210,
    "retransmission_count": 12,
    "unique_src_ports": 41,
    "unique_dst_ports": 17,
    "port_scan_score": 0.21
  },
  "flow_summary": {
    "unique_source_hosts": 8,
    "unique_destination_hosts": 12
  },
  "label": null,
  "label_source": null
}
```

Rules:
- `features` is an **open dict** (`Dict[str, float]`) — the exact feature set is NOT hard-coded and may be extended by later phases without breaking the contract.
- Canonical implementation: `backend/app/schemas/network_state.py::NetworkState` (Pydantic v2). JSON Schema export is the reference for other languages/modules.
- `timestamp_*` are UTC ISO-8601.
- `label` is ground truth ONLY when `label_source` names a verified source; otherwise both stay `null`.

## 6. NetworkStateSequence schema

```json
{
  "sequence_id": "seq_000001",
  "states": [],
  "sequence_length": 5,
  "window_seconds": 10,
  "target_state": null
}
```

Defaults (all configurable via env): `ML_SEQUENCE_LENGTH=5`, `ML_PREDICTION_HORIZON=3`, `TIME_WINDOW_SECONDS=10`.
Implementation: `backend/app/schemas/network_state.py::NetworkStateSequence`.

## 7. PredictionResult schema

```json
{
  "prediction_id": "pred_123",
  "timestamp": "2026-08-23T10:30:00Z",
  "risk_score": 0.87,
  "malicious_probability": 0.91,
  "confidence": 0.84,
  "predicted_stage": {
    "id": null,
    "name": null,
    "confidence": null,
    "source": null
  },
  "future_states": [],
  "feature_contributions": [],
  "model": {
    "name": "threatcast-world-model",
    "version": "0.1.0"
  }
}
```

All scores are floats in `[0, 1]`. Implementation: `backend/app/schemas/prediction.py::PredictionResult`.

## 8. ML contract

The world model must learn `P(S[t+1] | S[t])` and support iterative rollout to `S[t+K]` — it is **not** a static binary classifier. Conceptual separation (each its own module):

1. network state representation
2. temporal sequence modelling
3. future state prediction
4. malicious probability
5. attack progression
6. explainability

Preferred architecture: lightweight Temporal Transformer sized for RTX 3050 Laptop 6 GB VRAM. LSTM permitted as baseline.

**Stable interface** (implemented in a later ML phase):

```python
def predict(sequence: NetworkStateSequence) -> PredictionResult
```

## 9. Attack-stage contract

Candidate stages: Reconnaissance, Initial Access, Lateral Movement, Command and Control, Exfiltration.

- **GROUND TRUTH** = labels verifiably present in a dataset (e.g. CIC-IDS2018 attack categories).
- **DERIVED STAGE** = stage inferred by THREATCAST models or mapping heuristics.
- Never claim a derived stage is provided by CIC-IDS2018 unless verified against the actual downloaded files. No ground truth may be fabricated.

## 10. API contract

Full specifications with examples: [docs/API.md](docs/API.md).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/health` | liveness/readiness probe |
| POST | `/api/v1/ingestion/upload` | upload telemetry dataset |
| GET | `/api/v1/ingestion/jobs/{job_id}` | ingestion job status |
| POST | `/api/v1/predict` | run world-model prediction on a state sequence |
| GET | `/api/v1/predictions/{prediction_id}/timeline` | prediction timeline |
| GET | `/api/v1/states/{state_id}` | fetch stored network state |

### Global error format

Every error response:

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

Implementation: `backend/app/schemas/errors.py::ErrorEnvelope`.

## 11. Database contract

PostgreSQL tables (full specification: [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)):

`datasets`, `ingestion_jobs`, `network_states`, `predictions`, `future_predictions`, `explanations`, `models`

Conventions: snake_case names, UUID or text primary keys, `created_at` timestamps, JSONB for flexible payloads (features, explanations), FKs `network_states.dataset_id -> datasets.id`, `predictions.sequence/state references`, indexed lookup columns (`timestamp_end`, `job_id`, `prediction_id`). Schema is applied by Alembic in the backend phase — not before.

## 12. Explanation contract

```json
{
  "prediction_id": "pred_123",
  "method": "attention|shap|integrated_gradients",
  "feature_contributions": [
    {"feature": "syn_ratio", "contribution": 0.31}
  ],
  "generated_at": "2026-08-23T10:30:05Z"
}
```

Contributions sum need not equal 1; sign indicates direction of risk influence.

## 13. Naming conventions

- Python files/functions: `snake_case`; classes: `PascalCase`
- TypeScript variables/functions: `camelCase`; components/classes/types: `PascalCase`
- API paths: `/api/v1/resource-name`
- Database: `snake_case`
- Env vars: `UPPER_SNAKE_CASE`
- Artifacts: descriptive versioned names — `world_model.pt`, `model_metadata.json`, `feature_schema.json`

## 14. Logging conventions

Structured single-line logs containing: timestamp, level, component, message, request/job ID where applicable, exception details where appropriate.

Component names (fixed): `DATA_PIPELINE`, `ML`, `BACKEND`, `FRONTEND`, `INTEGRATION`, `SECURITY`.

Never log secrets, credentials, or full packet payloads.

## 15. Artifact conventions

- Model weights: `ml/artifacts/world_model.pt` (+ versioned copies `world_model_v<semver>.pt`)
- Metadata: `model_metadata.json` (training config, git SHA, metrics)
- Feature schema: `feature_schema.json` (feature name → dtype/stats)
- Label mappings: `label_mappings.json`
- All artifacts reproducible; no manual edits.

## 16. Integration rules

1. Modules communicate only through the schemas in this contract.
2. New features extend `NetworkState.features` additively; never rename existing keys.
3. Ports, env var names, API prefix, and error format are frozen.
4. Contract changes require updating CONTRACT.md, affected docs, and validation tests in the same change.
5. Every phase ends with `pytest tests/smoke -v` green.
