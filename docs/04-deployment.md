# Deployment Steps

## Backend — Render

1. Connect the GitHub repo to a Render **Web Service**.
   - Root directory: repo root
   - Dockerfile path: `backend/Dockerfile`
2. Add a Render **PostgreSQL** instance (Render dashboard → New → PostgreSQL). Copy its **Internal Database URL**.
3. Set environment variables on the Render service (see `03-environment-variables.md`) — at minimum `DATABASE_URL` (from step 2), `JWT_SECRET_KEY`, `CORS_ORIGINS`, `APP_ENV=production`.
4. Deploy. Confirm:
   ```bash
   curl https://<your-service>.onrender.com/health
   curl https://<your-service>.onrender.com/health/detailed
   ```
   `health/detailed` → `database.status: "ok"` confirms Postgres is connected correctly.

**Free tier note:** Render's 512 MB RAM free instance is close to the limit for this app's dependencies (`torch` + `librosa` + CNN inference). If you see requests fail with no error and a `Detected service running on port 8000` log line shortly after (i.e. a silent restart), that's an out-of-memory kill, not an application bug — see `05-troubleshooting.md`.

## Frontend — Vercel

1. Connect the repo to Vercel, root directory = `frontend/`.
2. Build command: `npm run build`. Output directory: `dist`.
3. No environment variable is required by default (`VITE_API_BASE_URL` defaults to `/api`). Only set it if the backend is served from a different path/origin than the frontend expects.

## Full stack locally / on a VM (Docker Compose)

See `02-docker-setup.md`. Use this for local development, demos, or a self-managed server — it's a separate path from the Render/Vercel deployment above, not a prerequisite for it.

## CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push to `main`/`birendra79` and every PR into `main`:

- **backend** job — installs Python deps, compiles all backend modules (`python -m compileall`) as a basic syntax/import sanity check.
- **frontend** job — installs npm deps, runs `npm run build`.
- **docker** job — builds both Docker images (backend build uses repo-root context, matching `docker-compose.yml`) and validates `docker compose config`.

This doesn't deploy anything automatically — Render and Vercel both auto-deploy from the connected branch independently when you push. CI is a safety net to catch build breakage before/alongside that.

## Post-deploy validation checklist

Run this after every deploy — see `05-troubleshooting.md` if any step fails:

| Check | How | Expected |
|---|---|---|
| Health | `curl https://<backend>/health` | `{"status":"ok", ...}`, HTTP 200 |
| Health (detailed) | `curl https://<backend>/health/detailed` | `database.status: "ok"`; `analyses_stored` non-decreasing across deploys |
| Register | Sign up with a new email in the UI | Account created |
| Login | Log in with that account | Dashboard loads |
| Analyze | Upload a real (not corrupted/tiny) audio file | Prediction result returned |
| History | Open the history page | Past analyses for the logged-in user are listed |
| Report download | Download a PDF report | PDF downloads and opens correctly |
