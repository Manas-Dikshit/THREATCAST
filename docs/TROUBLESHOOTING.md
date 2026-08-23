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

## Dataset problems (future phases)

CIC-IDS2018 files must be inspected first — column mismatches, infinities and duplicates are expected and handled by preprocessing rules in docs/MODEL.md.
