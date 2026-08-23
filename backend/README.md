# THREATCAST Backend

FastAPI + SQLAlchemy + PostgreSQL service (Module 3). Serves the Phase 2 data
pipeline and the Phase 3 world model behind the CONTRACT.md API.

## Layout

```
app/
├── api/             routers (/api/v1): health, models, ingestion, predict, states
├── core/            settings (pydantic-settings), logging, error hierarchy
├── db/              engine/session factory + declarative Base
├── models/          ORM tables: datasets, ingestion_jobs, network_states,
│                    predictions, future_predictions, explanations, models
├── schemas/         canonical contracts: NetworkState, sequence, PredictionResult
├── repositories/    DB access layer
├── services/        model_service (artifact loading), ingestion_service
│                    (upload validation + pipeline), prediction_service
└── main.py          FastAPI factory, CORS, request-id middleware, error handlers
alembic/             migrations (baseline = full schema)
tests/               pytest suite (SQLite in-memory/file DBs, tiny real model)
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
# first time / after schema changes:
python -m alembic upgrade head          # run from backend\
uvicorn app.main:app --reload --app-dir backend   # http://localhost:8000/docs
```

Environment (see `.env.example`): `DATABASE_URL`, `ML_ARTIFACTS_DIR`
(default `./ml/artifacts`), `ML_DEVICE`, `CORS_ORIGINS`, `MAX_UPLOAD_SIZE_MB`.

## Test

```powershell
pytest backend\tests -q
```

Tests use SQLite and train a tiny real world model per session — no mocking of
the ML path.

The `schemas` package is the integration surface for all other modules (see CONTRACT.md).
