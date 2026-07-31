<#
.SYNOPSIS
    AcousticSpace — one-time local development environment setup (Windows).

.DESCRIPTION
    - Creates a Python virtual environment for the backend and installs
      dependencies.
    - Copies backend/.env.example to backend/.env if it doesn't exist yet.
    - Installs frontend npm dependencies.
    - Prints next steps.

.USAGE
    Run from the repo root in PowerShell:
        .\setup.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "==> AcousticSpace local setup starting..." -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 1. Backend — Python virtual environment
# ---------------------------------------------------------------------------
Write-Host "`n[1/4] Setting up backend Python environment..." -ForegroundColor Yellow

Push-Location backend

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "  Created virtual environment (.venv)"
} else {
    Write-Host "  Virtual environment already exists, skipping creation"
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "  Backend dependencies installed" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Backend — .env file
# ---------------------------------------------------------------------------
Write-Host "`n[2/4] Checking backend/.env..." -ForegroundColor Yellow

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  Created backend/.env from .env.example — EDIT THIS FILE before running" -ForegroundColor Magenta
    Write-Host "  At minimum set: JWT_SECRET_KEY (openssl rand -hex 32) and DATABASE_URL" -ForegroundColor Magenta
} else {
    Write-Host "  backend/.env already exists, leaving it untouched"
}

Pop-Location

# ---------------------------------------------------------------------------
# 3. Frontend — npm dependencies
# ---------------------------------------------------------------------------
Write-Host "`n[3/4] Installing frontend dependencies..." -ForegroundColor Yellow

Push-Location frontend
npm install
Pop-Location

Write-Host "  Frontend dependencies installed" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Done — next steps
# ---------------------------------------------------------------------------
Write-Host "`n[4/4] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit backend\.env (JWT_SECRET_KEY, DATABASE_URL, CORS_ORIGINS)"
Write-Host "  2. Run the backend:"
Write-Host "       cd backend"
Write-Host "       .\.venv\Scripts\Activate.ps1"
Write-Host "       uvicorn app:app --reload"
Write-Host "  3. In a second terminal, run the frontend:"
Write-Host "       cd frontend"
Write-Host "       npm run dev"
Write-Host ""
Write-Host "  Or, to run the full stack via Docker instead:" -ForegroundColor Cyan
Write-Host "       docker compose up --build"
Write-Host ""