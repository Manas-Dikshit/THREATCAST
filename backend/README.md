# THREATCAST Backend

FastAPI + PostgreSQL service (Module 3).

**Phase 1 status:** app shell with `/api/v1/health`, global error envelope, typed settings, and the canonical Pydantic contracts. Full API/DB arrives in later phases.

## Layout

```
app/
├── api/           routers (/api/v1) — health implemented
├── core/          settings (pydantic-settings), logging
├── models/        SQLAlchemy ORM (later phase)
├── schemas/       canonical contracts: NetworkState, sequence, PredictionResult, errors
├── services/      orchestration (later phase)
├── repositories/  DB access (later phase)
└── main.py        FastAPI factory with global error handlers
```

## Run

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --app-dir backend   # http://localhost:8000/api/v1/health
```

## Test

```powershell
pytest backend\tests -v
```

The `schemas` package is the integration surface for all other modules (see CONTRACT.md).
