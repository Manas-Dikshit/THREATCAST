# THREATCAST Data Schema

Covers the canonical state/sequence contracts and the planned PostgreSQL schema.

## 1. Canonical NetworkState (CONTRACT.md §5)

Implemented in `backend/app/schemas/network_state.py` (Pydantic v2).

| Field | Type | Notes |
|-------|------|-------|
| `state_id` | str | e.g. `state_000001` |
| `timestamp_start` / `timestamp_end` | datetime UTC | window bounds |
| `window_seconds` | float | from `TIME_WINDOW_SECONDS` |
| `features` | `Dict[str, float]` | **open dict** — additive extension only, count never hard-coded |
| `flow_summary` | object (extra allowed) | host counts etc. |
| `label` | str \| null | ground truth only when verified |
| `label_source` | str \| null | names the verified source, else null |

Seed feature set (indicative, not exhaustive): `flow_count`, `packet_count`, `byte_count`, `syn_ratio`, `ack_ratio`, `rst_ratio`, `mean_iat`, `iat_variance`, `ttl_mean`, `ttl_variance`, `tcp_window_mean`, `retransmission_count`, `unique_src_ports`, `unique_dst_ports`, `port_scan_score`.

## 2. NetworkStateSequence (§6)

`{sequence_id, states[], sequence_length=5, window_seconds=10, target_state}`. Defaults configurable via env.

## 3. Planned PostgreSQL schema (implemented in backend phase via Alembic)

Conventions: snake_case, UUID text PKs, `created_at TIMESTAMPTZ DEFAULT now()`, JSONB for flexible payloads.

```
datasets
  id TEXT PK
  name TEXT NOT NULL
  source_type TEXT CHECK (source_type IN ('pcap','netflow','csv','authlog'))
  file_hash TEXT UNIQUE          -- dedupe / reproducibility
  meta JSONB                     -- profiling results, column mapping
  created_at TIMESTAMPTZ

ingestion_jobs
  id TEXT PK
  dataset_id TEXT -> datasets.id
  status TEXT CHECK (status IN ('queued','running','completed','failed'))
  states_generated INT
  error TEXT NULL
  started_at TIMESTAMPTZ
  finished_at TIMESTAMPTZ NULL
  INDEX (status), INDEX (dataset_id)

network_states
  id TEXT PK                     -- state_id
  dataset_id TEXT -> datasets.id
  job_id TEXT -> ingestion_jobs.id
  timestamp_start TIMESTAMPTZ NOT NULL
  timestamp_end TIMESTAMPTZ NOT NULL
  window_seconds REAL
  features JSONB NOT NULL        -- open dict preserved as-is
  flow_summary JSONB
  label TEXT NULL                -- ground truth only (verified source)
  label_source TEXT NULL
  UNIQUE (dataset_id, timestamp_start)
  INDEX (timestamp_end)

predictions
  id TEXT PK                     -- prediction_id
  model_id TEXT -> models.id
  input_sequence JSONB           -- or FK list; final choice in backend phase
  timestamp TIMESTAMPTZ
  risk_score REAL CHECK (risk_score BETWEEN 0 AND 1)
  malicious_probability REAL CHECK (malicious_probability BETWEEN 0 AND 1)
  confidence REAL CHECK (confidence BETWEEN 0 AND 1)
  predicted_stage JSONB
  created_at TIMESTAMPTZ
  INDEX (timestamp)

future_predictions               -- K-step rollout rows of a prediction
  id TEXT PK
  prediction_id TEXT -> predictions.id ON DELETE CASCADE
  step INT                       -- 1..K
  timestamp TIMESTAMPTZ
  features JSONB
  confidence REAL NULL
  UNIQUE (prediction_id, step)

explanations
  id TEXT PK
  prediction_id TEXT -> predictions.id ON DELETE CASCADE
  method TEXT                    -- attention | shap | integrated_gradients
  feature_contributions JSONB    -- [{feature, contribution}]
  created_at TIMESTAMPTZ

models
  id TEXT PK
  name TEXT                      -- threatcast-world-model
  version TEXT
  artifact_path TEXT             -- ml/artifacts/world_model.pt
  metadata_path TEXT
  trained_at TIMESTAMPTZ NULL
  metrics JSONB NULL
  UNIQUE (name, version)
```

Relationships: dataset 1—N jobs, dataset/job 1—N states, model 1—N predictions, prediction 1—N future_predictions/explanations.

This document is the fixed DB specification for the backend phase; deviations require a CONTRACT.md change first.
