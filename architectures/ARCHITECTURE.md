# THREATCAST — Detailed Architecture

Version: 0.1.0 (Phase 1) · Contracts live in [../CONTRACT.md](../CONTRACT.md)

## 1. Problem

Network defenders need *foresight*. Signature/ML IDS tools label traffic that already traversed the network. THREATCAST treats the network as a dynamical system: telemetry is compressed into time-windowed **network states** `S[t]`, and a learned world model estimates `P(S[t+1] | S[t])`. Iterating this transition enables K-step simulation of how an intrusion may unfold, plus risk, attack-progression and MITRE ATT&CK stage outputs for defenders.

## 2. Objectives

1. Predict future network states and malicious activity probability before compromise completes.
2. Explain predictions at feature level for defender trust.
3. Run fully offline on a laptop-class GPU (RTX 3050, 6 GB).
4. Remain extensible: open feature schema, pluggable predictors, streaming-ready ingestion.

## 3. High-level architecture

```mermaid
flowchart TD
    subgraph INPUT
        A1[PCAP]
        A2[NetFlow / IPFIX]
        A3[CSV flow records]
        A4[Auth / security logs]
    end
    subgraph data_pipeline
        B[Ingestion] --> C[Feature extraction<br/>flow + packet level]
        C --> D[Time window engine<br/>TIME_WINDOW_SECONDS]
    end
    D --> E["Canonical NetworkState S[t]<br/>(open feature dict)"]
    subgraph ml
        F[World model<br/>Temporal Transformer / LSTM baseline]
        G[Forward simulator S[t+1..K]]
        H[Attack predictor heads]
    end
    E --> F --> G
    F --> H
    subgraph security
        I[MITRE ATT&CK mapper]
        J[Explainability engine]
    end
    H --> I
    H --> J
    G --> I
    subgraph backend
        K[FastAPI /api/v1]
        L[(PostgreSQL)]
    end
    I --> K
    J --> K
    H --> K
    E --> K
    K --> L
    K --> M[Defender dashboard<br/>frontend React + Recharts]
```

## 4. Data flow

1. Telemetry files arrive via upload (`POST /api/v1/ingestion/upload`) or direct pipeline invocation.
2. Parsers normalise to flow records (5-tuple + timestamps + counters).
3. Feature extraction computes flow-level (counts, flag ratios, IAT stats, port diversity) and packet-level features (TTL/window stats, retransmissions).
4. The window engine buckets flows into `TIME_WINDOW_SECONDS` windows → one `NetworkState` per window.
5. States roll up into `NetworkStateSequence` (length `ML_SEQUENCE_LENGTH`).
6. The world model consumes sequences, predicts next-state distribution; the forward simulator unrolls K=`ML_PREDICTION_HORIZON` steps.
7. Predictor heads emit risk/malicious probability/stage; MITRE mapper and explainability enrich the `PredictionResult`.
8. Backend persists states/predictions/explanations; dashboard visualises timelines.

## 5. Module boundaries

| Module | Path | Owns | Must not do |
|--------|------|------|-------------|
| M1 | `data_pipeline/` | ingestion, parsers, features, windows, preprocessing | no ML, no API |
| M2 | `ml/` | state models, world model, training, inference, evaluation | no parsing of raw PCAP |
| M3 | `backend/` | REST API, DB, serving facade | no model math |
| M4 | `frontend/` | dashboard UI | no business logic |
| M5 | `integration/ security/ tests/ docker/ docs/` | cross-cutting | — |

Data crosses boundaries only as canonical schemas (CONTRACT.md §5–§7).

## 6. ML architecture

- **Input**: `NetworkStateSequence` (default length 5), features projected to a fixed vector via saved `feature_schema.json`.
- **Backbone**: Temporal Transformer encoder (small: ~4 layers, d_model ~128, 4 heads) + causal next-state head; LSTM baseline for comparison.
- **Heads**: future-state regression/distribution, malicious probability, risk score, optional stage logits.
- **Rollout**: autoregressive K-step simulation with uncertainty accumulation.
- **Constraints**: ≤ 6 GB VRAM → mixed precision (AMP), gradient accumulation, configurable batch size, checkpointing.
- Details: [docs/MODEL.md](../docs/MODEL.md), [ml/MODEL.md](../ml/MODEL.md).

## 7. Backend architecture

FastAPI layered app:

```
app/
├── api/          routers (/api/v1)
├── core/         settings (pydantic-settings), logging
├── models/       SQLAlchemy ORM (later phase)
├── schemas/      Pydantic contracts (canonical schema home)
├── services/     orchestration (ingestion, prediction facade)
└── repositories/ DB access (later phase)
```

Alembic owns migrations. Serving = thin adapter calling `ml`'s `predict(sequence) -> PredictionResult`.

## 8. Frontend architecture

Vite + React + TypeScript. Pages: Overview/risk timeline, Prediction detail, Ingestion jobs. `services/` wraps REST calls to `/api/v1`; `types/` mirrors backend Pydantic schemas as TS interfaces; Recharts renders risk timelines and predicted-state trajectories.

## 9. Database architecture

PostgreSQL 16; contract in [docs/DATA_SCHEMA.md](../docs/DATA_SCHEMA.md). Seven tables (`datasets`, `ingestion_jobs`, `network_states`, `predictions`, `future_predictions`, `explanations`, `models`) with JSONB for flexible payloads, indexed on lookup columns.

## 10. Explainability architecture

Feature-attribution layer over the world model: attention inspection first, SHAP/integrated-gradients where feasible. Output conforms to the explanation contract (CONTRACT.md §12): ordered `feature_contributions` attached to each prediction. Lives in `security/explainability/` (implementation later phase); never blocks the prediction path — computed async or on demand.

## 11. MITRE mapping architecture

Maps predicted stages/signal patterns to ATT&CK Enterprise techniques. Design:

1. Stage predictor emits derived-stage probabilities (Recon → Exfiltration taxonomy).
2. Deterministic mapping table (YAML in `security/attack_mapping/`) translates stage + top contributing features → technique IDs.
3. Every mapping records `source: "derived"` — ground truth only from verified dataset labels.

See [docs/MITRE_MAPPING.md](../docs/MITRE_MAPPING.md).

## 12. Deployment architecture

Docker Compose: `postgres` (16-alpine, healthcheck), `backend` (uvicorn :8000, healthcheck `/api/v1/health`), `frontend` (:5173). ML runs host-side (GPU passthrough to containers is out of scope until needed). Dockerfiles in `docker/`.

## 13. Offline architecture

No internet at runtime: datasets supplied manually into `data/` (gitignored), training/inference local, pip/npm dependencies vendored at install time, no cloud APIs anywhere.

## 14. Future streaming architecture

Phase 10+ concern. Ingestion interface will accept a "source" abstraction so batch file readers can be replaced/augmented by tailing consumers (e.g. Zeek/Suricata logs, Kafka if justified then). Nothing in Phase 1 depends on streaming; the window engine already treats input as timestamped record streams, which is the seam streaming will plug into.

## 15. Security boundaries

- Defensive-scope software only (see CONTRACT.md §Security in root README).
- Trust boundary: uploads — size-limited (`MAX_UPLOAD_SIZE_MB`), type-checked at ingestion.
- Secrets: env-only, never committed; default DB creds are local-dev placeholders.
- No remote code execution, no payload analysis executing samples, no offensive automation.

## 16. Integration strategy

- Canonical Pydantic schemas are the single integration surface (`backend/app/schemas`).
- Cross-module behaviour verified by `integration/tests` (later phases).
- Each phase must keep `pytest tests/smoke -v` green; smoke tests enforce structure, config, and importability of contracts.
