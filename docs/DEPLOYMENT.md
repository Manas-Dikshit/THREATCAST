# Deployment Guide

## Option A — Docker Compose (recommended for services)

```powershell
Copy-Item .env.example .env   # adjust if needed
docker compose up --build
```

Services:

| Service | URL / port | Health |
|---------|-----------|--------|
| PostgreSQL | localhost:5432 | `pg_isready` healthcheck |
| Backend | http://localhost:8000/api/v1/health | HTTP healthcheck |
| Frontend | http://localhost:5173 | — |

Data persists in the `pgdata` volume. ML training runs host-side (GPU-in-container is deliberately out of scope until required).

## Option B — native processes (dev)

```powershell
# terminal 1: PostgreSQL (local install or container only)
docker compose up postgres

# terminal 2: backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir backend

# terminal 3: frontend
cd frontend; npm install; npm run dev
```

## Configuration

All configuration flows through environment variables ([`.env.example`](../.env.example)); Compose sets service-internal values (e.g. `DATABASE_URL` pointing at the `postgres` host). Never bake secrets into images.

## Production notes (future phases)

- Build frontend (`npm run build`) and serve statically instead of the dev server.
- Run Alembic migrations before starting the backend once the DB phase lands.
- Pin model artifact versions in `models` table; health endpoint should report loaded model version.
