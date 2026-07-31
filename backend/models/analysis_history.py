"""
AcousticSpace — AnalysisHistory SQLAlchemy Model
=================================================
Defines the `analysis_history` table that replaces the in-memory
ANALYSIS_HISTORY list used in the original routes.py.

Design notes
------------
- `suspicious_segments` is stored as a JSON string (TEXT column) so the
  schema is database-agnostic (SQLite + PostgreSQL both support TEXT).
  Serialisation/deserialisation is handled by the repository helpers in
  routes.py — the API response shape is unchanged.
- `inference_mode` captures "heuristic" vs "ast-transformer" so historical
  records remain interpretable after model upgrades.
- All columns that map to the existing AnalyzeResponse Pydantic model are
  present.  Extra columns (inference_mode) are additive and do not affect
  the API contract.
- The same `Base` declared in models/user.py is reused so that
  `Base.metadata.create_all()` creates both `users` and `analysis_history`
  in a single call.

Columns
-------
id                  : TEXT primary key — UUID v4 string
filename            : TEXT — original uploaded filename
prediction          : TEXT — "Real" | "Deepfake"
confidence          : REAL — 0.00 – 100.00
suspicious_segments : TEXT — JSON-encoded list[dict]
room_acoustics_match: TEXT — "High" | "Low"
breathing_consistency: TEXT — "Consistent" | "Suspicious"
inference_time_sec  : REAL — wall-clock pipeline time in seconds
inference_mode      : TEXT — "heuristic" | "ast-transformer"
timestamp           : DATETIME — UTC timestamp of the analysis
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from models.user import Base  # reuse the shared declarative base


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id: str = Column(String(36), primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    filename: str = Column(Text, nullable=False)
    prediction: str = Column(String(16), nullable=False)        # "Real" | "Deepfake"
    confidence: float = Column(Float, nullable=False)
    suspicious_segments: str = Column(Text, nullable=False, default="[]")  # JSON string
    room_acoustics_match: str = Column(String(8), nullable=False)   # "High" | "Low"
    breathing_consistency: str = Column(String(16), nullable=False) # "Consistent" | "Suspicious"
    inference_time_sec: float = Column(Float, nullable=False)
    inference_mode: str = Column(String(32), nullable=False, default="heuristic")
    timestamp: datetime = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<AnalysisHistory id={self.id!r} prediction={self.prediction!r} "
            f"confidence={self.confidence:.2f}>"
        )