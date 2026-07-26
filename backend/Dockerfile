# =============================================================================
# AcousticSpace Backend — Production Dockerfile
# =============================================================================
# Multi-stage-friendly single-stage image with:
#   - tini as PID-1 init process (proper signal forwarding + zombie reaping)
#   - Non-root user (appuser:appgroup) to reduce container attack surface
#   - Docker HEALTHCHECK using the /health endpoint
#   - Layer-cache-optimised dependency installation
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# 1. System packages
#    - tini         : lightweight init that forwards signals correctly
#    - libsndfile1  : required by soundfile (WAV/FLAC I/O)
#    - ffmpeg       : required by librosa for MP3/OGG/M4A decoding
#    - build-essential : needed by some Python packages with C extensions
#    - curl         : used by the HEALTHCHECK instruction below
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        libsndfile1 \
        ffmpeg \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Non-root user
#    Running as root inside a container is a security anti-pattern.
#    All subsequent instructions run as appuser.
# ---------------------------------------------------------------------------
RUN groupadd --system appgroup && \
    useradd  --system --gid appgroup --no-create-home appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# 3. Python dependencies
#    Copy requirements.txt first so Docker cache is reused on code-only
#    changes (the most common rebuild pattern).
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install \
    --default-timeout=1000 \
    --retries 20 \
    --no-cache-dir \
    -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Application code
# ---------------------------------------------------------------------------
COPY --chown=appuser:appgroup . .

# ---------------------------------------------------------------------------
# 5. Non-root user switch
# ---------------------------------------------------------------------------
USER appuser

# ---------------------------------------------------------------------------
# 6. Exposed port
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# 7. Docker HEALTHCHECK
#    - interval  : check every 30 s (gives time for model loading)
#    - timeout   : single check must respond within 10 s
#    - start-period: allow 60 s on first start for model / DB init
#    - retries   : mark unhealthy after 3 consecutive failures
#    Uses curl to hit the lightweight /health liveness probe.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# 8. Entrypoint — tini as PID 1
#    tini ensures uvicorn receives SIGTERM correctly and child processes
#    are reaped, preventing zombie accumulation.
# ---------------------------------------------------------------------------
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info", "--no-access-log"]
