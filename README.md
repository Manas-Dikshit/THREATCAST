# THREATCAST

AI-powered **Predictive Cyber Defence** / Cyber Threat World Model.

THREATCAST learns the evolving state of a computer network from traffic telemetry and predicts how that state may evolve in the future — turning raw traffic into early warnings for defenders.

## Problem statement

Traditional IDS tools are reactive: they classify what already happened. THREATCAST models network dynamics as a temporal sequence of *network states* and learns the transition function `P(S[t+1] | S[t])`, enabling:

- K-step forward simulation of future network states
- malicious activity probability / risk scoring
- infiltration and attack-progression prediction
- MITRE ATT&CK stage mapping with explainability

## Conceptual architecture

```
INPUT (PCAP / NetFlow-IPFIX / CSV flows / auth logs)
  -> DATA INGESTION            (data_pipeline)
  -> FEATURE EXTRACTION        (data_pipeline)
  -> TIME WINDOW ENGINE        (data_pipeline, TIME_WINDOW_SECONDS)
  -> NETWORK STATE S[t]        (canonical NetworkState contract)
  -> WORLD MODEL               (ml: Temporal Transformer, LSTM baseline)
  -> FORWARD SIMULATION S[t+1..t+K]
  -> ATTACK PREDICTOR          (risk / malicious probability / stage)
  -> MITRE ATT&CK MAPPING      (security)
  -> EXPLAINABILITY            (security)
  -> BACKEND API  /api/v1      (backend, FastAPI + PostgreSQL)
  -> DEFENDER DASHBOARD        (frontend, React + TypeScript + Recharts)
```

The original conceptual diagram lives in [`predictive_cyber_defence_architecture.mmd`](predictive_cyber_defence_architecture.mmd).

## Technology stack

| Layer     | Technology |
|-----------|------------|
| Backend   | Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy, Alembic |
| Database  | PostgreSQL 16 |
| ML        | PyTorch, scikit-learn, NumPy, Pandas |
| Data      | Scapy (PyShark only where necessary) |
| Frontend  | React, TypeScript, Vite, Recharts |
| Testing   | pytest, Vitest |
| Infra     | Docker, Docker Compose |
| Config    | `.env` (see `.env.example`) |

Everything runs **offline**. No cloud AI APIs. The primary dataset is CIC-IDS2018 (supplied manually, never auto-downloaded).

## Repository structure

```
THREATCAST/
├── backend/          FastAPI app, DB, REST API, model serving
├── frontend/         Defender dashboard (React + TS + Vite)
├── ml/               World model, training, inference, evaluation
├── data_pipeline/    Ingestion, parsers, features, time windows
├── integration/      Cross-module tests, fixtures, scripts
├── security/         MITRE ATT&CK mapping, explainability
├── docs/             API, schemas, model, deployment docs
├── architectures/    Detailed architecture document
├── tests/smoke/      Foundation validation tests
├── docker/           Dockerfiles per service
└── scripts/          setup / development / validation scripts
```

Module ownership boundaries are defined in [CONTRACT.md](CONTRACT.md); full architecture in [ARCHITECTURE.md](ARCHITECTURE.md) and [architectures/ARCHITECTURE.md](architectures/ARCHITECTURE.md).

## Development phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Foundation, contracts, skeleton, docs | **current** |
| 2 | Data pipeline (ingestion → features → windows) | planned |
| 3 | Network state representation | planned |
| 4 | World model training | planned |
| 5 | Forward simulation + prediction | planned |
| 6 | Attack predictor + MITRE mapping | planned |
| 7 | Explainability | planned |
| 8 | Backend API + database | planned |
| 9 | Frontend dashboard | planned |
| 10 | Integration, evaluation, streaming readiness | planned |

## Setup (Windows)

```powershell
# 1. Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configuration
Copy-Item .env.example .env

# 3. Validation tests
pytest tests\smoke -v
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for daily workflows and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker Compose usage.

## Ports (global contract)

| Service  | Port |
|----------|------|
| Frontend | 5173 |
| Backend  | 8000 |
| PostgreSQL | 5432 |

API prefix: `/api/v1`

## Environment variables

All configuration flows through environment variables defined in [`.env.example`](.env.example): `APP_ENV`, `BACKEND_HOST`, `BACKEND_PORT`, `FRONTEND_PORT`, `DATABASE_URL`, `ML_MODEL_PATH`, `ML_METADATA_PATH`, `ML_DEVICE`, `ML_SEQUENCE_LENGTH`, `ML_PREDICTION_HORIZON`, `TIME_WINDOW_SECONDS`, `MAX_UPLOAD_SIZE_MB`, `LOG_LEVEL`.

## ML pipeline (future)

1. Telemetry ingestion → flow/packet features (`data_pipeline`)
2. Time-windowed canonical `NetworkState` sequences (length 5 default)
3. Lightweight Temporal Transformer learns `P(S[t+1] | S[t])`; LSTM kept as baseline
4. Forward simulation K=3 steps; attack predictor emits risk/malicious probability
5. MITRE ATT&CK mapping + feature-level explanations

Details in [docs/MODEL.md](docs/MODEL.md), [ml/MODEL.md](ml/MODEL.md), [ml/TRAINING.md](ml/TRAINING.md). Hardware target: RTX 3050 Laptop 6 GB VRAM (CUDA when available, CPU fallback, mixed precision).

## Testing strategy

- `tests/smoke` — foundation/structure/config validation (this phase)
- `backend/tests`, `ml/tests`, `data_pipeline/tests` — unit tests per module (later phases)
- `integration/tests` — cross-module contract tests (later phases)

## Offline operation

No component requires internet at runtime: datasets are supplied manually, models train locally on CUDA/CPU, and all services run via local Docker or native processes.

## Security scope

THREATCAST is **defensive** security software: traffic analysis, anomaly detection, attack prediction, monitoring, and defender decision support only. No exploit execution, malware deployment, credential theft, persistence, or offensive automation. See the security contract in [CONTRACT.md](CONTRACT.md).
