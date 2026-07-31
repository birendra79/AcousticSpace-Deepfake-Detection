"""
AcousticSpace Backend — API Route Definitions
=============================================
Week 1 deliverable (Pydantic model + route skeleton).
Week 2 deliverable (full /analyze pipeline wired to service layer).
Week 3 deliverable (persistent DB storage — replaced in-memory list).

This module defines all HTTP endpoints exposed by the AcousticSpace API:

    GET  /health       — service liveness probe                    [FR-5.2]
    POST /analyze      — upload audio file, run deepfake analysis  [FR-5.3]
    GET  /history      — list all past analyses (database-backed)  [FR-5.4]
    GET  /model-info   — currently loaded inference engine info    [FR-5.5]

The AnalyzeResponse Pydantic model and all endpoint signatures are
unchanged from the previous implementation.  The only internal change
is that analysis results are now written to and read from the
`analysis_history` SQLite table via SQLAlchemy instead of an in-process
list.  [FR-6.1, FR-6.2, FR-6.3]
"""

import json
import os
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.database import get_db
from auth.dependencies import get_current_active_user
from models.analysis_history import AnalysisHistory
from models.user import User
from services.audio_processing import load_and_validate_audio
from services.feature_extraction import extract_all_features
from services.inference import get_inference_engine
from services.pdf_report import generate_analysis_pdf
from services.explanation import build_explanation
from security.audit_log import AuditEvent, emit
from security.file_validator import FileValidationError, validate_upload
from ml.prediction.predictor import predict_audio
print("✅ CNN IMPORT SUCCESS")

router = APIRouter()

# ---------------------------------------------------------------------------
# Configuration  [NFR-6.1]
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 25))


# ---------------------------------------------------------------------------
# Response model  [FR-5.6]
# Unchanged — all fields identical to the previous implementation.
# ---------------------------------------------------------------------------
class AnalyzeResponse(BaseModel):
    """Structured response returned by POST /analyze and stored in history."""

    id: str                          # UUID v4 — unique identifier for this analysis
    filename: str                    # Original uploaded filename
    prediction: str                  # "Real" | "Deepfake"
    confidence: float                # 0.00 – 100.00
    suspicious_segments: list[dict]  # [{start_sec, end_sec}] — empty list if Real
    room_acoustics_match: str        # "High" | "Low"
    breathing_consistency: str       # "Consistent" | "Suspicious"
    inference_time_sec: float        # Wall-clock time for the full pipeline (seconds)
    timestamp: str                   # UTC ISO-8601 string
    explanation: dict | None = None  # XAI explanation block (None for legacy rows)
    viz: dict | None = None          # Visualization data (None for legacy rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: AnalysisHistory) -> dict:
    """
    Convert an AnalysisHistory ORM row to the dict shape that AnalyzeResponse
    expects.  `suspicious_segments` is deserialised from its JSON string
    representation back to a list[dict].
    """
    return {
        "id": row.id,
        "filename": row.filename,
        "prediction": row.prediction,
        "confidence": row.confidence,
        "suspicious_segments": json.loads(row.suspicious_segments),
        "room_acoustics_match": row.room_acoustics_match,
        "breathing_consistency": row.breathing_consistency,
        "inference_time_sec": row.inference_time_sec,
        "timestamp": row.timestamp.isoformat() if isinstance(row.timestamp, datetime) else row.timestamp,
    }


# ---------------------------------------------------------------------------
# GET /dashboard-stats  — aggregate statistics for the Dashboard page
# ---------------------------------------------------------------------------
@router.get("/dashboard-stats")
def get_dashboard_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Return pre-aggregated statistics consumed by the frontend Dashboard.

    Computed fields
    ---------------
    total_analyses      : int   — all-time analysis count
    deepfake_count      : int   — analyses predicted "Deepfake"
    real_count          : int   — analyses predicted "Real"
    avg_confidence      : float — mean confidence across all analyses (0–100)
    today_uploads       : int   — analyses submitted today (UTC date)
    weekly_uploads      : int   — analyses submitted in the last 7 days
    recent_history      : list  — last 10 analyses (same shape as /history)
    daily_counts        : list  — [{date, deepfake, real}] for last 30 days
    weekly_trend        : list  — [{week, deepfake, real}] for last 8 ISO weeks
    confidence_dist     : list  — [{range, count}] histogram in 10-pt buckets
    """
    from datetime import timezone, timedelta
    from sqlalchemy import func, case

    now_utc = datetime.utcnow()
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now_utc - timedelta(days=7)

    rows_all = (
        db.query(AnalysisHistory)
        .order_by(AnalysisHistory.timestamp.desc())
        .all()
    )

    total = len(rows_all)
    deepfake_count = sum(1 for r in rows_all if r.prediction == "Deepfake")
    real_count = total - deepfake_count
    avg_confidence = (
        round(sum(r.confidence for r in rows_all) / total, 2) if total else 0.0
    )
    today_uploads = sum(
        1 for r in rows_all
        if (r.timestamp if isinstance(r.timestamp, datetime) else datetime.fromisoformat(r.timestamp)) >= today_start
    )
    weekly_uploads = sum(
        1 for r in rows_all
        if (r.timestamp if isinstance(r.timestamp, datetime) else datetime.fromisoformat(r.timestamp)) >= week_start
    )

    recent_history = [_row_to_dict(r) for r in rows_all[:10]]

    # --- Daily counts for last 30 days (bar chart) -------------------------
    thirty_days_ago = now_utc - timedelta(days=30)
    daily: dict[str, dict] = {}
    for r in rows_all:
        ts = r.timestamp if isinstance(r.timestamp, datetime) else datetime.fromisoformat(r.timestamp)
        if ts < thirty_days_ago:
            continue
        day = ts.strftime("%Y-%m-%d")
        if day not in daily:
            daily[day] = {"date": day, "deepfake": 0, "real": 0}
        if r.prediction == "Deepfake":
            daily[day]["deepfake"] += 1
        else:
            daily[day]["real"] += 1
    daily_counts = sorted(daily.values(), key=lambda x: x["date"])

    # --- Weekly trend for last 8 ISO weeks (line chart) --------------------
    eight_weeks_ago = now_utc - timedelta(weeks=8)
    weekly: dict[str, dict] = {}
    for r in rows_all:
        ts = r.timestamp if isinstance(r.timestamp, datetime) else datetime.fromisoformat(r.timestamp)
        if ts < eight_weeks_ago:
            continue
        iso = ts.isocalendar()
        key = f"{iso[0]}-W{iso[1]:02d}"
        if key not in weekly:
            weekly[key] = {"week": key, "deepfake": 0, "real": 0}
        if r.prediction == "Deepfake":
            weekly[key]["deepfake"] += 1
        else:
            weekly[key]["real"] += 1
    weekly_trend = sorted(weekly.values(), key=lambda x: x["week"])

    # --- Confidence distribution histogram (pie / histogram) ---------------
    buckets = {f"{i}-{i+10}": 0 for i in range(0, 100, 10)}
    for r in rows_all:
        bucket_index = min(int(r.confidence // 10) * 10, 90)
        key = f"{bucket_index}-{bucket_index + 10}"
        buckets[key] += 1
    confidence_dist = [{"range": k, "count": v} for k, v in buckets.items() if v > 0]

    return {
        "total_analyses": total,
        "deepfake_count": deepfake_count,
        "real_count": real_count,
        "avg_confidence": avg_confidence,
        "today_uploads": today_uploads,
        "weekly_uploads": weekly_uploads,
        "recent_history": recent_history,
        "daily_counts": daily_counts,
        "weekly_trend": weekly_trend,
        "confidence_dist": confidence_dist,
    }


# ---------------------------------------------------------------------------
# GET /health  [FR-5.2]  — liveness probe (no auth, used by Docker HEALTHCHECK)
# ---------------------------------------------------------------------------
@router.get("/health")
def health_check() -> dict:
    """
    Lightweight liveness probe.

    Returns the service status and current UTC timestamp.
    No authentication required — used by Docker HEALTHCHECK and the
    frontend connectivity test.
    """
    from monitoring.health import get_liveness
    return get_liveness()


# ---------------------------------------------------------------------------
# GET /health/detailed  — deep readiness probe (no auth, internal use)
# ---------------------------------------------------------------------------
@router.get("/health/detailed")
def health_detailed(db: Session = Depends(get_db)) -> dict:
    """
    Deep readiness probe that checks every runtime dependency:
    database connectivity, disk space, memory, model state, and uptime.

    No authentication required so load-balancers and orchestrators can
    poll it without credentials.  Sensitive values (paths, hostnames) are
    included intentionally — this endpoint should NOT be exposed publicly.
    Restrict it at the network / reverse-proxy layer in production.

    Returns HTTP 200 with ``status: "ok" | "degraded" | "critical"``.
    """
    from monitoring.health import get_health_report
    return get_health_report(db)


# ---------------------------------------------------------------------------
# GET /metrics  — in-process metrics (JSON + Prometheus text format)
# ---------------------------------------------------------------------------
@router.get("/metrics")
def metrics_endpoint(
    format: str = "json",   # noqa: A002  — intentional shadowing of builtin
) -> Response:
    """
    Expose in-process metrics.

    Query parameters
    ----------------
    format : "json" (default) | "prometheus"
        ``json``       — returns a JSON object with counters, histograms, gauges.
        ``prometheus`` — returns Prometheus text exposition format (v0.0.4),
                         suitable for scraping by a Prometheus server or
                         Grafana agent.

    No authentication required so monitoring infrastructure can scrape it
    without credentials.  Restrict at the network layer in production.
    """
    from monitoring.metrics import REGISTRY
    from fastapi.responses import PlainTextResponse

    if format.lower() == "prometheus":
        return PlainTextResponse(
            content=REGISTRY.to_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    return REGISTRY.to_dict()


# ---------------------------------------------------------------------------
# GET /model-info  [FR-5.5, FR-4.7]
# ---------------------------------------------------------------------------
@router.get("/model-info")
def model_info() -> dict:
    """
    Return metadata about the currently loaded inference engine.

    Reports the active mode (heuristic or ast-transformer), whether a
    checkpoint was found, the compute device, and the architecture description.
    """
    engine = get_inference_engine()
    return engine.model_info()


# ---------------------------------------------------------------------------
# GET /history  [FR-5.4, FR-6.2, FR-6.3]
# ---------------------------------------------------------------------------
@router.get("/history")
def get_history(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Return all past analysis results in reverse chronological order.
    Requires a valid access token.
    """
    rows = (
        db.query(AnalysisHistory)
        .order_by(AnalysisHistory.timestamp.desc())
        .all()
    )
    results = [_row_to_dict(row) for row in rows]
    return {"count": len(results), "results": results}


# ---------------------------------------------------------------------------
# DELETE /history/{analysis_id}  — remove a single analysis record
# ---------------------------------------------------------------------------
@router.delete("/history/{analysis_id}", status_code=200)
def delete_analysis(
    analysis_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Permanently delete a single analysis record by its UUID.
    Requires a valid access token. Emits an audit event.
    Returns 404 if the record does not exist.
    """
    row = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
    if row is None:
        emit(
            AuditEvent.HISTORY_DELETE_ONE,
            outcome="failure",
            user_id=current_user.id,
            user_email=current_user.email,
            ip=request.client.host if request.client else None,
            request_id=getattr(request.state, "request_id", None),
            resource=analysis_id,
            detail="Record not found",
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found.",
        )
    db.delete(row)
    db.commit()
    emit(
        AuditEvent.HISTORY_DELETE_ONE,
        outcome="success",
        user_id=current_user.id,
        user_email=current_user.email,
        ip=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        resource=analysis_id,
    )
    return {"deleted": analysis_id}


# ---------------------------------------------------------------------------
# DELETE /history  — remove every analysis record (bulk wipe)
# ---------------------------------------------------------------------------
@router.delete("/history", status_code=200)
def delete_all_analyses(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Permanently delete all analysis records.
    Requires a valid access token. Emits an audit event.
    """
    count = db.query(AnalysisHistory).delete()
    db.commit()
    emit(
        AuditEvent.HISTORY_DELETE_ALL,
        outcome="success",
        user_id=current_user.id,
        user_email=current_user.email,
        ip=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        detail=f"Deleted {count} records",
    )
    return {"deleted": count}


# ---------------------------------------------------------------------------
# GET /report/{analysis_id}  — download a PDF report for one analysis
# ---------------------------------------------------------------------------
@router.get("/report/{analysis_id}")
def download_report(
    analysis_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Generate and stream a PDF report for a single analysis.
    Requires a valid access token. Emits an audit event.
    """
    row = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found.",
        )

    data = {
        "id": row.id,
        "filename": row.filename,
        "prediction": row.prediction,
        "confidence": row.confidence,
        "suspicious_segments": row.suspicious_segments,
        "room_acoustics_match": row.room_acoustics_match,
        "breathing_consistency": row.breathing_consistency,
        "inference_time_sec": row.inference_time_sec,
        "inference_mode": row.inference_mode,
        "timestamp": (
            row.timestamp.isoformat()
            if isinstance(row.timestamp, datetime)
            else row.timestamp
        ),
    }

    pdf_bytes = generate_analysis_pdf(data)
    safe_name = f"acousticspace-report-{analysis_id}.pdf"

    emit(
        AuditEvent.REPORT_GENERATED,
        outcome="success",
        user_id=current_user.id,
        user_email=current_user.email,
        ip=request.client.host if request.client else None,
        request_id=getattr(request.state, "request_id", None),
        resource=analysis_id,
    )

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


# ---------------------------------------------------------------------------
# POST /analyze  [FR-5.3]
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Upload an audio file and run the full deepfake detection pipeline.

    Security hardening (over original)
    ------------------------------------
    - Requires a valid access token (get_current_active_user).
    - File is validated by the secure file_validator before ANY decoding:
        1. Filename sanitisation (path traversal, null bytes, control chars)
        2. Extension allow-list
        3. File size cap (defence-in-depth, also enforced in file_validator)
        4. Magic-byte MIME validation
        5. Shannon entropy check (zip-bomb / encrypted payload guard)
    - All upload and analysis events are written to the structured audit log.
    - Errors return RFC 7807 Problem Details bodies (via app.py handlers).
    - No prediction logic is changed.

    Processing steps
    ----------------
      1. Secure file validation                              [FR-1.3, FR-1.4]
      2. Load and normalise audio (mono, 16 kHz, float32)   [FR-2.1 – FR-2.3]
      3. Extract acoustic features                          [FR-3.1 – FR-3.4]
      4. Run inference (heuristic or AST-transformer)       [FR-4.1 – FR-4.5]
      5. Build XAI explanation (additive only)
      6. Persist AnalysisHistory row to database            [FR-6.1]
      7. Return AnalyzeResponse
    """
    ip         = request.client.host if request.client else None
    request_id = getattr(request.state, "request_id", None)

    # Read the file once; all subsequent steps work from this buffer.
    raw_bytes = await file.read()

    # -- Step 1: Secure file validation -------------------------------------
    try:
        safe_filename = validate_upload(
            filename=file.filename or "upload",
            content=raw_bytes,
            max_mb=MAX_UPLOAD_SIZE_MB,
        )
    except FileValidationError as exc:
        emit(
            AuditEvent.UPLOAD_REJECTED,
            outcome="failure",
            user_id=current_user.id,
            user_email=current_user.email,
            ip=ip,
            request_id=request_id,
            resource=file.filename,
            detail=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    emit(
        AuditEvent.UPLOAD_RECEIVED,
        outcome="success",
        user_id=current_user.id,
        user_email=current_user.email,
        ip=ip,
        request_id=request_id,
        resource=safe_filename,
        detail=f"{len(raw_bytes)} bytes",
    )

    start = time.time()

    emit(
        AuditEvent.ANALYSIS_STARTED,
        outcome="success",
        user_id=current_user.id,
        user_email=current_user.email,
        ip=ip,
        request_id=request_id,
        resource=safe_filename,
    )

    # Track in-flight gauge for dashboard visibility
    from monitoring.metrics import inc_in_flight, dec_in_flight, record_analysis
    inc_in_flight()

    # -- Step 2: Load and validate audio  [FR-2.1 – FR-2.3] ----------------
    try:
        audio_array, sample_rate = load_and_validate_audio(raw_bytes, safe_filename)
    except Exception as exc:
        dec_in_flight()
        emit(
            AuditEvent.ANALYSIS_FAILED,
            outcome="failure",
            user_id=current_user.id,
            user_email=current_user.email,
            ip=ip,
            request_id=request_id,
            resource=safe_filename,
            detail=f"Audio load error: {type(exc).__name__}",
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode the audio file. Ensure it is a valid, non-corrupted audio file.",
        )

    # -- Step 3: Extract features  [FR-3.1 – FR-3.4] -----------------------
    print("ROUTE STEP 1 - Before feature extraction")

    features = extract_all_features(audio_array, sample_rate)

    print("ROUTE STEP 2 - Feature extraction completed")

    # -- Step 4: Run inference  [FR-4.1 – FR-4.5] --------------------------
    engine = get_inference_engine()

    print("ROUTE STEP 3 - Engine loaded")

    try:
        result = engine.predict(audio_array, sample_rate, features)
        print("ROUTE STEP 4 - Prediction completed")
    except Exception:
        import traceback

        print("=" * 80)
        print("ANALYSIS ERROR")
        print(traceback.format_exc())
        print("=" * 80)
        raise

    inference_mode: str = engine.model_info().get("mode", "heuristic")

    # -- Step 4b: Build XAI explanation (additive — no prediction change) ---
    explanation = build_explanation(result, features, inference_mode)

    elapsed = round(time.time() - start, 3)
    now     = datetime.utcnow()
    analysis_id = str(uuid.uuid4())

    # -- Step 5: Persist to database  [FR-6.1] ------------------------------
    row = AnalysisHistory(
        id=analysis_id,
        filename=safe_filename,
        prediction=result["prediction"],
        confidence=result["confidence"],
        suspicious_segments=json.dumps(result["suspicious_segments"]),
        room_acoustics_match=result["room_acoustics_match"],
        breathing_consistency=result["breathing_consistency"],
        inference_time_sec=elapsed,
        inference_mode=inference_mode,
        timestamp=now,
    )
    db.add(row)
    db.commit()

    dec_in_flight()
    record_analysis(
        prediction=result["prediction"],
        mode=inference_mode,
        duration_sec=elapsed,
        file_bytes=len(raw_bytes),
    )

    emit(
        AuditEvent.ANALYSIS_COMPLETED,
        outcome="success",
        user_id=current_user.id,
        user_email=current_user.email,
        ip=ip,
        request_id=request_id,
        resource=analysis_id,
        detail=f"prediction={result['prediction']} confidence={result['confidence']:.1f} elapsed={elapsed}s",
    )

    # -- Step 6: Return response  -------------------------------------------
    return {
        "id":                   analysis_id,
        "filename":             safe_filename,
        "prediction":           result["prediction"],
        "confidence":           result["confidence"],
        "suspicious_segments":  result["suspicious_segments"],
        "room_acoustics_match": result["room_acoustics_match"],
        "breathing_consistency":result["breathing_consistency"],
        "inference_time_sec":   elapsed,
        "timestamp":            now.isoformat(),
        "explanation":          explanation,
        "viz":                  features.get("viz"),
    }
