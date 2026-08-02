# Docker Setup

Three services, defined in `docker-compose.yml` at the repo root.

| Service | Build context | Exposes | Notes |
|---|---|---|---|
| `backend` | `.` (repo root), Dockerfile at `backend/Dockerfile` | `8000` (internal only) | Needs repo-root context — the Dockerfile copies `ml/` and `deepfake_cnn.pth` from outside `backend/` |
| `frontend` | `./frontend` | `80` (internal only) | Multi-stage build: compiles the app with Node, then serves the static output via its own nginx |
| `nginx` | `./nginx` | `80` (published to host) | The only container reachable from outside the Docker network |

## Running it

```bash
cp backend/.env.example backend/.env
# edit backend/.env — at minimum set JWT_SECRET_KEY and DATABASE_URL

docker compose up --build
```

Open `http://localhost`. The API is reachable at `http://localhost/api/...`.

Or run `setup.ps1` (Windows) first to prepare a native (non-Docker) dev environment instead — see the root `setup.ps1` script.

## Rebuilding after code changes

```bash
docker compose up --build          # rebuild everything
docker compose up --build backend  # rebuild just one service
```

## Common pitfalls (already hit and fixed in this project)

- **Backend build context must be the repo root, not `backend/`.** The Dockerfile does `COPY ml ./ml` and `COPY deepfake_cnn.pth .`, both of which live outside the `backend/` folder. Setting `context: ./backend` in `docker-compose.yml` (or in a CI `docker build` command) breaks the build — always use `context: .` with `dockerfile: backend/Dockerfile`.
- **`torch` installs the CUDA build by default**, which is unnecessarily large/heavy for a CPU-only host like Render. Pin the CPU wheel explicitly:
  ```
  --extra-index-url https://download.pytorch.org/whl/cpu
  torch==2.3.1+cpu
  torchaudio==2.3.1+cpu
  ```
- **`librosa.display` requires `matplotlib`** even if you never call the plotting function. If you don't need plotting in production, import `matplotlib`/`librosa.display` lazily inside the function that uses them, not at module load time — this keeps the dependency (and its memory footprint) out of the server image entirely.

## Health check

The backend image includes a `HEALTHCHECK` hitting `/health`; `docker compose ps` will show `(healthy)` once it passes (after the 60s start period).
