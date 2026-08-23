# Create the Python virtual environment and install dependencies.

$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSCommandPath))
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    if ($?) { Write-Host "Created .venv" }
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "Created .env from example"
}

Write-Host "Setup complete. Run: pytest tests\smoke -v"
