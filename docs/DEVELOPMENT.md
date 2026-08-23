# Development Guide

## Prerequisites (Windows)

- Python 3.11+
- Node.js 20 LTS
- Docker Desktop (for PostgreSQL/Compose)

## One-time setup

```powershell
.\scripts\setup\setup_env.ps1
```

or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
cd frontend; npm install; cd ..
```

## Daily workflow

| Task | Command |
|------|---------|
| Backend dev server | `uvicorn app.main:app --reload --app-dir backend` |
| Frontend dev server | `cd frontend; npm run dev` |
| All services | `docker compose up --build` |
| Foundation tests | `pytest tests\smoke -v` |
| Data pipeline tests | `pytest data_pipeline\tests -v` |
| Full test suite | `pytest -v` |
| Run data pipeline (CSV) | `python -m data_pipeline.cli --input <file.csv> --out out\ml_dataset` |
| Run data pipeline (PCAP) | `python -m data_pipeline.cli --input <file.pcap> --out out\ml_dataset` |
| Profile only, no export | add `--profile-only` |
| Full validation | `.\scripts\validation\validate.ps1` |

### Data pipeline example

```powershell
python -m data_pipeline.cli --input data\flows.csv --out out\ml_dataset `
    --window 10 --seq-len 5 --horizon 3 --source-name cic_ids2018
```

Produces `states.parquet`, `sequences.jsonl`, `tensors.npz`, `feature_schema.json`,
`preprocessing_metadata.json`, `label_mappings.json`, `dataset_profile.json`.
See `data_pipeline/README.md` for the full pipeline documentation.

## Module boundaries

See CONTRACT.md §4. Work inside your module's path; cross-module data flows only through the Pydantic schemas in `backend/app/schemas`.

## Conventions

Naming, logging, artifacts: CONTRACT.md §13–§15. Type hints everywhere in Python. No hard-coded absolute paths — use settings/env vars.

## Adding a dependency

Justify it against the ladder first: stdlib → existing dep → platform feature → new dep. Update root `requirements.txt` (Python) or `frontend/package.json`.
