# THREATCAST Data Schema

Covers the canonical state/sequence contracts, the Phase 2 pipeline schemas, and the planned PostgreSQL schema.

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

## 2b. Pipeline schemas (Phase 2, implemented in `data_pipeline/src/schemas/`)

### FlowRecord (normalized intermediate record)

Every field except `timestamp` is optional; `None` = "source did not provide it" (never fabricated).

| Field | Type | Notes |
|-------|------|-------|
| `timestamp` | datetime UTC | flow start |
| `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol` | str/int | protocol numbers normalized (`6→"tcp"`) |
| `total_bytes`, `total_packets`, `duration_s` | float | CIC fwd+bwd components summed when totals absent |
| `flags` | str | TCP flag letters `"FSRPAU"` subset, e.g. `"SA"` |
| `iat_mean_s`, `iat_var_s`, `iat_max_s` | float | per-flow IAT stats when source provides them (CIC µs converted to s) |
| `fwd_bytes`, `bwd_bytes`, `fwd_packets`, `bwd_packets` | float | directional counters |
| `ttl_mean`, `ttl_var`, `tcp_window_mean`, `retransmission_count`, `fragmentation_count`, `payload_mean`, `payload_max` | float/int | PCAP-only packet-level features |
| `label`, `label_kind` | str / LabelKind | verbatim dataset label; kind ∈ {ground_truth, derived, unknown} |

### FeatureSchema / PreprocessingMetadata / DatasetProfile

- **FeatureSchema**: ordered `feature_names` — the authoritative tensor column order; versioned.
- **PreprocessingMetadata**: `preprocessing_version`, feature names, per-feature mean/std (fit on train only), fit time window, source dataset, window/sequence/horizon config, label mappings.
- **DatasetProfile**: dynamic inspection of actual files — row/column counts, duplicate rows, per-column dtype/nulls/infs/stats/samples, canonical column mapping (`mapped_columns`), unmapped headers, timestamp range, raw label distribution, warnings.

### Window features

Computed from what records carry; unavailable features are omitted, never zero-filled. Seed set: `flow_count`, `packet_count`, `byte_count`, `{syn,ack,fin,rst,psh,urg}_ratio`, `mean_iat`, `iat_variance`, `iat_max` (fallback: inter-flow gaps), `fwd_byte_ratio`, `unique_src_ports`, `unique_dst_ports`, `port_scan_score` (derived heuristic), plus PCAP-only `ttl_mean`, `ttl_variance`, `tcp_window_mean`, `retransmission_count`, `fragmentation_count`, `payload_size_mean`, `payload_size_max`.

### Sequence generation & leakage rules

1. States bucket into fixed `TIME_WINDOW_SECONDS` windows anchored at the epoch grid; empty windows are skipped.
2. Sequences slide over contiguous runs only — a gap breaks the run.
3. `target_state = S(t+K)` within the run where available, else `None`.
4. Splits are chronological (`split_time_aware`); normalization stats are fitted on train data only.
5. Labels attach to states only with a verified `label_source`.

### ML-ready dataset output (`export_ml_dataset`)

| File | Content |
|------|---------|
| `states.parquet` | flattened state table |
| `sequences.jsonl` | contract JSON per line |
| `tensors.npz` | `X [N,L,F]`, `Y [N,F]`, `target_mask [N]`, `label_ids [N]` (float32/int64) |
| `feature_schema.json` / `preprocessing_metadata.json` / `label_mappings.json` / `dataset_profile.json` | metadata |

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
