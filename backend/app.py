"""
AcousticSpace Backend — Application Entrypoint
===============================================
Composition root for the FastAPI application.

Security additions (over the original)
---------------------------------------
- slowapi rate limiter mounted as application state + exception handler.
- RequestLoggingMiddleware  — structured JSON per-request log + X-Request-ID.
- SecurityHeadersMiddleware — CSP, HSTS, X-Frame-Options, etc.
- Tightened CORS.
- Global RFC 7807 exception handlers.
- Startup lifecycle: weak-key warning, DB initialisation, system info log.
- Shutdown lifecycle: audit log flush.

Monitoring additions
--------------------
- Structured JSON logging (all loggers emit JSON records in production).
- GET /health         — lightweight liveness probe.
- GET /health/detailed — deep readiness probe (DB, disk, memory, model).
- GET /metrics         — in-process counters + histograms (JSON or Prometheus).
- Startup system-info log record.
- Unhandled exceptions increment the errors_total counter.
"""

from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import logging
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, status          # noqa: E402
from fastapi.exceptions import RequestValidationError                # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                   # noqa: E402
from fastapi.responses import JSONResponse                           # noqa: E402
from slowapi.errors import RateLimitExceeded                        # noqa: E402

# The limiter singleton lives in limiter.py to avoid a circular import between
# app.py and auth/routes.py (both need to reference the same limiter object).
from limiter import limiter                                          # noqa: E402

from api.routes import router as api_router                         # noqa: E402
from auth.routes import router as auth_router                       # noqa: E402
from auth.database import create_all_tables                         # noqa: E402
from middleware.logging_middleware import RequestLoggingMiddleware   # noqa: E402
from middleware.security_headers import SecurityHeadersMiddleware    # noqa: E402

logger = logging.getLogger("acousticspace.app")

# ---------------------------------------------------------------------------
# Structured JSON logging configuration
# ---------------------------------------------------------------------------
_APP_ENV = os.getenv("APP_ENV", "development").lower()
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class _JsonFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        # If the message is already valid JSON (from logging_middleware), emit as-is
        msg = record.getMessage()
        try:
            json.loads(msg)
            return msg  # already structured JSON
        except (json.JSONDecodeError, TypeError):
            pass
        doc = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": msg,
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc, ensure_ascii=False)


def _configure_logging() -> None:
    """
    Configure all loggers to emit structured JSON in production;
    fall back to the human-readable format in development.
    """
    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    if _APP_ENV == "production":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
        )
    root.addHandler(handler)


_configure_logging()

# ---------------------------------------------------------------------------
# CORS — tightened from the original wildcard methods/headers
# ---------------------------------------------------------------------------
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
)
CORS_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Only the methods and headers our API actually uses
_CORS_METHODS  = ["GET", "POST", "DELETE", "PATCH", "OPTIONS"]
_CORS_HEADERS  = [
    "Authorization",
    "Content-Type",
    "Accept",
    "X-Request-ID",
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AcousticSpace API",
    description=(
        "Deepfake audio detection using Room Impulse Response (RIR), "
        "environmental reverberation time (RT60), breathing cadence analysis, "
        "and spatial acoustics."
    ),
    version="1.0.0",
    # Hide schema endpoints in production to reduce attack surface
    docs_url="/docs" if os.getenv("APP_ENV", "development") != "production" else None,
    redoc_url="/redoc" if os.getenv("APP_ENV", "development") != "production" else None,
)

# Attach limiter to app state — required by SlowAPI so it can look up the
# limiter instance at request time (used by @limiter.limit() decorated routes).
app.state.limiter = limiter

# ---------------------------------------------------------------------------
# Middleware  (added in reverse order — last added = outermost = first to run)
# ---------------------------------------------------------------------------

# 1. CORS — must be outermost so OPTIONS preflights are handled before auth
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=_CORS_METHODS,
    allow_headers=_CORS_HEADERS,
    max_age=600,
)

# 2. Security headers — applied to every response after CORS
app.add_middleware(SecurityHeadersMiddleware)

# 3. Request logging — innermost so it measures actual handler time
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Uniform error response helper  (RFC 7807 Problem Details)
# ---------------------------------------------------------------------------

def _problem(
    status_code: int,
    title: str,
    detail: str,
    request: Request | None = None,
    errors: list[dict] | None = None,
) -> JSONResponse:
    body: dict = {
        "status": status_code,
        "title":  title,
        "detail": detail,
    }
    if request:
        body["instance"] = str(request.url.path)
    if errors:
        body["errors"] = errors
    # Surface the request-id if the logging middleware injected it
    headers: dict[str, str] = {}
    if request and hasattr(request.state, "request_id"):
        headers["X-Request-ID"] = request.state.request_id
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    from security.audit_log import AuditEvent, emit
    emit(
        AuditEvent.RATE_LIMIT_HIT,
        outcome="failure",
        ip=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        detail=str(exc.detail),
    )
    return _problem(
        status_code=429,
        title="Too Many Requests",
        detail="Rate limit exceeded. Please slow down and try again shortly.",
        request=request,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        {
            "field": ".".join(str(loc) for loc in err["loc"]),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return _problem(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        title="Validation Error",
        detail="One or more request fields failed validation.",
        request=request,
        errors=errors,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    titles = {
        400: "Bad Request",
        401: "Unauthorised",
        403: "Forbidden",
        404: "Not Found",
        409: "Conflict",
        413: "Payload Too Large",
        415: "Unsupported Media Type",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
    }
    title = titles.get(exc.status_code, "Error")
    response = _problem(
        status_code=exc.status_code,
        title=title,
        detail=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        request=request,
    )
    # Preserve WWW-Authenticate header on 401 responses
    if exc.headers:
        for k, v in exc.headers.items():
            response.headers[k] = v
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    try:
        from monitoring.metrics import REGISTRY
        REGISTRY.errors_total.inc()
    except Exception:
        pass
    return _problem(
        status_code=500,
        title="Internal Server Error",
        detail="An unexpected error occurred. Please try again later.",
        request=request,
    )


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    from auth.utils import SECRET_KEY, _warn_weak_secret
    _warn_weak_secret(SECRET_KEY)

    create_all_tables()

    # --- Speed fix: use all available CPU cores for torch ---
    import torch
    torch.set_num_threads(1)

# --- Warm-up: run one dummy prediction at startup so torch's internal
    # kernels/buffers are already initialized before the first real user request
    from services.inference import get_inference_engine
    import numpy as np

    engine = get_inference_engine()
    try:
        silent_audio = np.zeros(16000, dtype=np.float32)  # 1 sec of silence
        dummy_features = {"breathing": {"cadence_regularity_score": 0.5}}
        engine.predict(silent_audio, 16000, dummy_features)
        logger.info("Model warm-up prediction completed successfully")
    except Exception:
        logger.exception("Model warm-up prediction failed (non-fatal, continuing startup)")
    
    import torch
    torch.set_grad_enabled(False)
    # Log structured system info record on startup
    import platform
    import sys
    import socket
    from monitoring.metrics import REGISTRY

    logger.info(json.dumps({
        "event":          "startup",
        "app_version":    "1.0.0",
        "app_env":        os.getenv("APP_ENV", "development"),
        "python_version": sys.version.split()[0],
        "platform":       platform.system(),
        "hostname":       socket.gethostname(),
        "cpu_count":      os.cpu_count(),
        "cors_origins":   CORS_ORIGINS,
        "log_level":      _LOG_LEVEL,
        "docs_enabled":   _APP_ENV != "production",
        "process_start_time": REGISTRY.process_start_time.get(),
    }, ensure_ascii=False))


@app.on_event("shutdown")
def on_shutdown() -> None:
    import logging as _logging
    # Flush audit log handlers so no records are lost on graceful shutdown
    audit = _logging.getLogger("acousticspace.audit")
    for handler in audit.handlers:
        handler.flush()
    logger.info("AcousticSpace API shutting down — audit log flushed.")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(api_router)
app.include_router(auth_router)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "AcousticSpace API", "status": "running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)