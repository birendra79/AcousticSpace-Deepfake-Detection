# AcousticSpace

**Spatial Reverb & Respiratory Cadence Forensics for Audio Deepfake Detection**

AcousticSpace analyzes room acoustics, reverberation, and breathing patterns — not just the voice itself — to catch synthetic/deepfake audio that traditional detectors miss.

---

## Features

- **Deepfake detection** — a custom-trained CNN classifies uploaded audio as real or synthetic, with a confidence score.
- **Room acoustics analysis** — reverberation and RT60-based room impulse response signals.
- **Breathing cadence analysis** — flags inconsistent or absent respiratory patterns typical of synthetic speech.
- **JWT authentication** — register/login with short-lived access tokens and refresh-token rotation.
- **Analysis history** — every result is stored per user and retrievable later.
- **PDF reports** — generate a downloadable report for any past analysis.
- **Structured audit logging & health monitoring** — `/health`, `/health/detailed`, `/metrics` for operational visibility.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript + Vite, React Router, Axios, Recharts, wavesurfer.js |
| Backend | FastAPI, SQLAlchemy, Uvicorn |
| Auth | JWT (python-jose), bcrypt |
| ML / Audio | PyTorch (CPU), librosa, soundfile, scikit-learn |
| Database | PostgreSQL (production) / SQLite (local dev) |
| Reports | reportlab (PDF generation) |
| Infra | Docker, Docker Compose, nginx |
| Hosting | Render (backend), Vercel (frontend) |
| CI | GitHub Actions |

---

## Project Structure

```
.
├── backend/               FastAPI application
│   ├── api/                REST routes (analyze, history, dashboard, health...)
│   ├── auth/                Registration, login, JWT, database session
│   ├── services/             Audio processing, inference, explanation
│   ├── security/               File validation, audit logging
│   ├── middleware/               Logging, security headers
│   ├── monitoring/                 Metrics
│   └── Dockerfile
├── frontend/                React + Vite SPA
│   └── Dockerfile           (multi-stage: build → static nginx)
├── ml/                     Model, feature extraction, prediction pipeline
│   └── prediction/predictor.py    Loads deepfake_cnn.pth, runs inference
├── nginx/                  Reverse proxy config (self-hosted stack)
├── docs/                   Detailed documentation (see below)
├── docker-compose.yml      Full local/self-hosted stack
├── setup.ps1               One-command local dev setup (Windows)
└── .github/workflows/ci.yml   CI: backend/frontend/docker build checks
```

---

## Quick Start (local development)

### Option A — Windows, one command

```powershell
.\setup.ps1
```
This creates a Python venv, installs backend + frontend dependencies, and copies `backend/.env.example` → `backend/.env`. Edit that `.env` (at minimum `JWT_SECRET_KEY`), then:

```powershell
# Terminal 1
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app:app --reload

# Terminal 2
cd frontend
npm run dev
```

### Option B — Docker Compose (full stack, any OS)

```bash
cp backend/.env.example backend/.env
# edit backend/.env

docker compose up --build
```
Open `http://localhost`.

Full details: [`docs/02-docker-setup.md`](docs/02-docker-setup.md).

---

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/register` | Create an account |
| `POST` | `/login` | Authenticate, receive access + refresh tokens |
| `POST` | `/refresh` | Rotate an expired access token |
| `POST` | `/logout` | Invalidate the current session |
| `GET`  | `/me` | Current authenticated user |
| `POST` | `/analyze` | Upload audio, run deepfake detection |
| `GET`  | `/history` | List past analyses for the current user |
| `GET`  | `/report/{id}` | Download a PDF report for a past analysis |
| `GET`  | `/dashboard-stats` | Summary stats for the dashboard |
| `GET`  | `/health` | Liveness probe |
| `GET`  | `/health/detailed` | Deep readiness probe — DB, disk, memory, model |
| `GET`  | `/model-info` | Active inference mode / device |
| `GET`  | `/metrics` | In-process counters/histograms |

Interactive docs at `/docs` (Swagger) and `/redoc` — disabled automatically when `APP_ENV=production`.

---

## Documentation

Detailed docs live in [`docs/`](docs/):

| Doc | Covers |
|---|---|
| [`01-architecture.md`](docs/01-architecture.md) | System diagram, request flows, auth & analyze pipelines |
| [`02-docker-setup.md`](docs/02-docker-setup.md) | Docker Compose services, running locally, common pitfalls |
| [`03-environment-variables.md`](docs/03-environment-variables.md) | Every `.env` variable, what it does, production values |
| [`04-deployment.md`](docs/04-deployment.md) | Render + Vercel deployment steps, CI, post-deploy validation checklist |
| [`05-troubleshooting.md`](docs/05-troubleshooting.md) | Real issues hit in this project and how they were fixed |
| [`architecture-diagram.drawio`](docs/architecture-diagram.drawio) | Visual architecture diagram (open in draw.io) |

For the full incident-by-incident debugging history (database persistence, memory limits, crash root-causes), see [`AcousticSpace_Debug_Summary.md`](AcousticSpace_Debug_Summary.md).

---

## Deployment

- **Backend:** Render (Docker), connected to a Render-hosted PostgreSQL instance.
- **Frontend:** Vercel, auto-deployed from this repo.
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) validates backend imports, frontend build, and Docker builds on every push.

See [`docs/04-deployment.md`](docs/04-deployment.md) for full setup steps.

---

## Known Limitations

- Render's free-tier 512 MB RAM instance leaves little headroom for the ML pipeline — see [`docs/05-troubleshooting.md`](docs/05-troubleshooting.md).
- `/model-info` currently reports a "heuristic" AST-transformer status that isn't the model actually used for predictions (which is always the trained CNN) — a known, non-blocking, documented quirk.

## Dataset (training)

Model training used the ASVspoof2019 LA dataset:
https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download