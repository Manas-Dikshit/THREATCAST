# THREATCAST Frontend

Defender dashboard (Module 4). React 18 + TypeScript + Vite + Recharts.

**Phase 1 status:** placeholder shell + contract types only. Dashboard pages are built in a later phase.

## Structure

```
src/
├── components/   reusable UI
├── pages/        Overview, Prediction detail, Ingestion jobs (later phases)
├── services/     REST wrappers for /api/v1 (backend on :8000)
├── hooks/        shared React hooks
├── types/        TS mirrors of backend contracts (contracts.ts)
└── utils/
```

## Run

```powershell
npm install
npm run dev      # http://localhost:5173, proxies /api -> :8000
npm test         # vitest
```
