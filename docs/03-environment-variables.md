# Environment Variables

All backend configuration lives in `backend/.env` (copy from `backend/.env.example`, which documents every variable inline with defaults).

## Application

| Variable | Purpose | Production value |
|---|---|---|
| `APP_ENV` | Enables production logging format; disables `/docs` & `/redoc` | `production` |
| `LOG_LEVEL` | Logging verbosity | `INFO` (use `DEBUG` only for troubleshooting) |

## Database

| Variable | Purpose | Production value |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy connection string | A PostgreSQL DSN, e.g. `postgresql+psycopg2://user:pass@host:5432/db`. **Never SQLite in production** — Render's filesystem is ephemeral and a SQLite file gets wiped on every restart/redeploy, silently deleting all users. |

## Authentication

| Variable | Purpose | Production value |
|---|---|---|
| `JWT_SECRET_KEY` | Signs auth tokens (HS256) | A strong random value — generate with `openssl rand -hex 32`. The app warns at startup if the unsafe default placeholder is still in use. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime | `30` (default) |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime | `7` (default) |
| `REFRESH_TOKEN_REMEMBER_DAYS` | "Remember me" refresh token lifetime | `30` (default) |
| `JWT_ISSUER` / `JWT_AUDIENCE` | JWT claims validation | Must match between issuing and verifying — leave as default unless you have a reason to change both |

## CORS / networking

| Variable | Purpose | Production value |
|---|---|---|
| `CORS_ORIGINS` | Comma-separated list of origins allowed to call the API | Must include every domain the frontend is actually served from (e.g. the Vercel URL, and/or `http://localhost` for the nginx stack). Never use `*`. |

## Audio processing

| Variable | Purpose | Production value |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | Upload size cap | `25` (default) |
| `SAMPLE_RATE` | Audio resample target (Hz) | `16000` — must match what the CNN was trained on; do not change without retraining |
| `DEVICE` | Torch device | `cpu` (Render has no GPU) |
| `MODEL_CHECKPOINT_PATH` | Path to an optional AST-transformer checkpoint | Not currently used for live predictions — see the note in `05-troubleshooting.md` about `/model-info` |

## Monitoring

| Variable | Purpose | Production value |
|---|---|---|
| `AUDIT_LOG_PATH` | Append-only audit log file (JSON Lines) | `audit.log` (default) |
| `RATE_LIMIT_STORAGE` | slowapi rate-limit backend | `memory://` for single-instance; use Redis for multi-instance deployments |
| `DISK_WARN_MB` / `DISK_CRITICAL_MB` | `/health/detailed` disk thresholds | `500` / `100` (defaults) |

## Frontend

| Variable | Purpose | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Backend API base URL, baked in at build time | `/api` — correct for both the nginx-proxied stack and the Vercel+Render setup, so it usually doesn't need to be set explicitly |
