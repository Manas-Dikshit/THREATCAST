# Troubleshooting

## Port already in use

```
Error: bind 0.0.0.0:8000 failed
```
Something else owns the port (global contract: 8000/5173/5432 — do not switch ports). Find and stop it:
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <pid>
```

## Docker Compose issues

- `docker compose config` validates syntax without starting anything.
- PostgreSQL container unhealthy → check port 5432 conflicts with a local Postgres service.
- Rebuild after Dockerfile changes: `docker compose up --build`.

## Python environment

- `ModuleNotFoundError: app` → run pytest/uvicorn from repository root, or use the provided scripts which set paths.
- PyTorch install without CUDA on Windows: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (offline-friendly once downloaded).

## Frontend

- `npm run dev` fails on port 5173 → another Vite instance is running; stop it rather than switching ports.
- API calls fail → ensure the backend is up and the Vite proxy target (`http://localhost:8000`) is correct.

## Tests failing after structure changes

Smoke tests enforce the documented folder/config layout. If you intentionally restructure, update `tests/smoke/test_structure.py` **and** CONTRACT.md in the same change.

## Backend (Phase 4)

- `DATABASE_ERROR` responses — `DATABASE_URL` unreachable or migrations missing. Run `python -m alembic upgrade head` from `backend\`; check Postgres is up (`docker compose up postgres`).
- `MODEL_NOT_LOADED` (503) on `/predict` — artifacts missing/corrupt at `ML_ARTIFACTS_DIR`. Train or copy a bundle first (`ml/artifacts/world_model.pt` + metadata); health shows the load error in `model.error`.
- Upload rejected with `INVALID_INPUT` — extension must be `.csv/.pcap/.pcapng`, content must sniff as CSV header or PCAP magic, size ≤ `MAX_UPLOAD_SIZE_MB`.
- `JOB_FAILED` with "No usable flow records" — CSV needs an ISO-8601 `timestamp` column (unix epoch ints are not parsed by the Phase 2 loader).
- CORS errors in the browser — set `CORS_ORIGINS=http://localhost:5173` in `.env`.
- SQLite vs Postgres: unit tests use SQLite automatically; only dev/deploy need Postgres.

## Dataset problems (future phases)

CIC-IDS2018 files must be inspected first — column mismatches, infinities and duplicates are expected and handled by preprocessing rules in docs/MODEL.md.
