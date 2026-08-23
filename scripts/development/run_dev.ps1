# Start development services natively (backend + frontend) in separate windows.

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
Set-Location $root

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "Set-Location '$root'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --app-dir backend"

Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "Set-Location '$root\frontend'; npm run dev"

Write-Host "Backend: http://localhost:8000/api/v1/health  Frontend: http://localhost:5173"
Write-Host "PostgreSQL: docker compose up postgres"
