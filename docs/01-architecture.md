# Architecture Overview

## Components

- **Frontend** — React + Vite + TypeScript SPA. In production, compiled to static files and served by nginx (Docker) or by Vercel directly.
- **Backend** — FastAPI (single Uvicorn worker), JWT authentication via SQLAlchemy, audio deepfake detection pipeline.
- **Database** — PostgreSQL in production (Render-hosted). SQLite is acceptable for local development only.
- **ML pipeline** — `librosa` for audio decoding/feature extraction, a custom-trained CNN (`deepfake_cnn.pth`) for the actual prediction.
- **Reverse proxy (self-hosted stack only)** — nginx routes `/` to the frontend static files and `/api/` to the backend.

## Request flow (self-hosted / docker-compose)

```
Browser → nginx (:80)
            ├─ /            → frontend container (static files)
            └─ /api/*       → backend container (:8000)
                                  └─ PostgreSQL / SQLite
```

## Request flow (current live deployment)

```
Browser → Vercel (frontend, static)
            └─ /api/* (VITE_API_BASE_URL) → Render (backend, FastAPI)
                                                 └─ Render PostgreSQL
```

These two deployment paths are independent. Docker Compose (nginx + frontend + backend containers) is for running the whole stack locally or on a self-managed VM — it is **not** what's currently serving production traffic. Render and Vercel deploy and scale the backend and frontend separately.

## Auth flow

1. `POST /auth/register` / `POST /auth/login` → issues a short-lived JWT access token + a longer-lived refresh token.
2. Every protected request sends `Authorization: Bearer <token>`.
3. `get_current_user()` decodes the token and looks up the user by ID in the database.
4. Access tokens auto-refresh via the axios interceptor when they expire.

## Analyze flow

1. Frontend uploads audio via `POST /analyze` (multipart form).
2. Backend validates the file (`security/file_validator.py`).
3. `services/audio_processing.py` decodes to mono float32 at the target sample rate (soundfile primary, librosa/audioread fallback).
4. `services/inference.py` (`InferenceEngine.predict`) calls the trained CNN (`ml/prediction/predictor.py`) and assembles the response (prediction, confidence, suspicious segments, room-acoustics/breathing heuristics).
5. Result is stored and returned to the frontend; also retrievable later via `/history`.
6. A PDF report can be generated on demand via `reportlab`.

See `../AcousticSpace_Debug_Summary.md` for the incident history behind several of these design decisions (why Postgres instead of SQLite, why CPU-only torch, etc.).
