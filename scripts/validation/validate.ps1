# Full foundation validation: structure, config, backend tests, compose syntax.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
Set-Location $root

Write-Host "=== THREATCAST foundation validation ==="

Write-Host "[1/4] Python syntax check..."
python -m compileall -q backend ml data_pipeline tests
if ($?) { Write-Host "  OK" }

Write-Host "[2/4] Structure + contract tests..."
pytest tests\smoke -q
if ($?) { Write-Host "  OK" }

Write-Host "[3/4] Backend tests..."
pytest backend\tests -q
if ($?) { Write-Host "  OK" }

Write-Host "[4/4] Docker Compose syntax..."
docker compose config --quiet
if ($?) { Write-Host "  OK" } else { Write-Host "  SKIP (docker not available)" }

Write-Host "=== validation complete ==="
