# Troubleshooting

Real issues hit during this project, and their fixes. Kept here so they don't get rediscovered from scratch.

## `404 User not found` on every request after login works fine

**Cause:** SQLite on an ephemeral filesystem (Render free tier resets storage on restart/redeploy). The user row disappears; the JWT (which just carries a user ID) still looks "valid" to itself, but the ID no longer resolves.

**Fix:** Use a real database service (Postgres), not a SQLite file on disk, in production. Set `DATABASE_URL` accordingly — see `03-environment-variables.md`.

## Docker build fails: `COPY deepfake_cnn.pth .` — file not found

**Cause:** Build context set to `./backend` instead of the repo root. The Dockerfile assumes it can see `ml/` and `deepfake_cnn.pth`, which live one level up from `backend/`.

**Fix:** Build context must be `.` (repo root) with `dockerfile: backend/Dockerfile` — both in `docker-compose.yml` and in any CI `docker build` command.

## `/analyze` crashes the server mid-request with no error, then it restarts

Two distinct causes were found here, in order:

1. **A method defined outside its class.** `InferenceEngine.predict()` was accidentally indented at module level, not inside the class — calling `engine.predict(...)` would raise `AttributeError`. Fix: correct indentation.
2. **Out-of-memory kill.** Once (1) was fixed, the pipeline actually ran — and `torch` + `transformers` + `librosa` + CNN inference exceeded Render's 512 MB free-tier limit, so the OS silently killed the process (no Python traceback, since the kill happens below the interpreter). Logs show `analysis.started` then silence, then `Detected service running on port 8000` (a fresh restart).

**Fix (free-tier mitigations):**
- Use the CPU-only torch wheel, not the default CUDA build (see `02-docker-setup.md`).
- Remove training-only dependencies from the server image (`pandas`, `matplotlib`, `kagglehub` — none are imported anywhere in the live serving path).
- `torch.set_num_threads(1)` and `torch.set_grad_enabled(False)` at startup.
- Don't warm up / preload the model at server startup — load it lazily on first use instead, to keep the baseline memory footprint lower.

If crashes persist after these changes, the free tier's 512 MB is genuinely insufficient — upgrading the Render instance is the durable fix.

## `Audio load error: LibsndfileError` for a specific file

**Cause (usually):** the file itself is corrupt or truncated, not a code/server bug. A ~7 KB file claiming to be 5 seconds of MP3 audio (~12 kbps effective bitrate) is implausible for a real recording.

**How to confirm:** test with a known-good MP3/WAV file. If that works, the original file was the problem. Both `soundfile` and `librosa`/`audioread` were verified independently to decode a properly-encoded MP3 correctly — the two-loader fallback in `services/audio_processing.py` is not the issue.

## `/analyze` times out on the frontend (`AxiosError: timeout of 60000ms exceeded`)

**Cause:** Render's free-tier CPU is slow for `torch`/`librosa` inference, and the frontend's axios client and/or nginx's default 60s proxy timeout is shorter than actual processing time.

**Fix:** `nginx.conf` in this repo sets `proxy_read_timeout 180s;` / `proxy_send_timeout 180s;` for `/api/`. If still timing out, raise the axios client timeout in `frontend/src/api/client.ts` as well, and/or upgrade the Render instance for faster CPU.

## `/model-info` reports `"mode": "heuristic"`, `"checkpoint_found": false` — is detection actually working?

**Yes.** This field refers to an unused AST-transformer code path (`services/inference.py`) that looks for a checkpoint at `ml/checkpoints/ast_deepfake.pt`, which doesn't exist in this repo. It has no effect on predictions — every `/analyze` request always goes through the trained CNN (`deepfake_cnn.pth`) via `ml/prediction/predictor.py`, regardless of what `self.mode` says. This is a misleading status field, not a functional bug. Worth cleaning up eventually (wire the AST path in for real, or remove the reporting).

## `ModuleNotFoundError` for a package that's clearly in `requirements.txt`

**Common causes seen in this project:**
- The dependency was added to the **wrong** `requirements.txt` — this repo has one at the root (unused by Docker) and one at `backend/requirements.txt` (the one the Dockerfile actually installs from).
- The file got corrupted by a Windows `echo >> file` command writing UTF-16 instead of UTF-8, producing garbled/invalid lines pip can't parse. Edit requirements files in a proper editor (UTF-8), not via `echo >>` in PowerShell.
- A library was removed from `requirements.txt` but another module still imports it implicitly — e.g. removing `matplotlib` broke `ml/features/feature_extractor.py` because `librosa.display` requires it internally, even though the actual plotting function is never called by the live server. Fix: make the import lazy (inside the function that needs it) instead of re-adding the whole dependency, if it's not otherwise needed.
