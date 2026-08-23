"""ORM tables (docs/DATA_SCHEMA.md section 3).

Portability notes:
- UUIDs are stored as String(36) so the same models run on PostgreSQL and
  SQLite (tests). ponytail: switch to sqlalchemy.Uuid / JSONB when PG-only.
- Raw PCAP/flow bytes are never stored here — datasets carry metadata and the
  original file reference only.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ..db.base import Base, TimestampMixin, utcnow


def _uuid() -> str:
    import uuid

    return uuid.uuid4().hex


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_format: Mapped[str] = mapped_column(String(16), nullable=False)  # csv|pcap|pcapng
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    jobs: Mapped[list["IngestionJob"]] = relationship(back_populates="dataset")
    states: Mapped[list["NetworkStateRow"]] = relationship(back_populates="dataset")


class IngestionJob(Base, TimestampMixin):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_format: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    # PENDING | PROCESSING | COMPLETED | FAILED
    states_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sequences_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    dataset: Mapped[Dataset | None] = relationship(back_populates="jobs")


class NetworkStateRow(Base, TimestampMixin):
    """One canonical NetworkState (CONTRACT.md section 5)."""

    __tablename__ = "network_states"
    __table_args__ = (
        Index("ix_states_dataset_time", "dataset_id", "timestamp_start"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    state_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=True, index=True
    )
    timestamp_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    timestamp_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_seconds: Mapped[float] = mapped_column(Float, default=10.0)
    features: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    label_source: Mapped[str | None] = mapped_column(String(128), nullable=True)

    dataset: Mapped[Dataset | None] = relationship(back_populates="states")


class PredictionRow(Base, TimestampMixin):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    sequence_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    malicious_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    stage_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stage_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stage_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    stage_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_name: Mapped[str] = mapped_column(String(128), default="threatcast-world-model")
    model_version: Mapped[str] = mapped_column(String(32), default="0.0.0")
    input_sequence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    future_states: Mapped[list["FuturePrediction"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan",
        order_by="FuturePrediction.step",
    )
    explanations: Mapped[list["Explanation"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan",
    )


class FuturePrediction(Base):
    """One step of the K-step rollout attached to a prediction."""

    __tablename__ = "future_predictions"
    __table_args__ = (Index("ix_future_prediction_step", "prediction_row_id", "step"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_row_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    features: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    prediction: Mapped[PredictionRow] = relationship(back_populates="future_states")


class Explanation(Base):
    __tablename__ = "explanations"
    __table_args__ = (Index("ix_explanations_pred_contrib", "prediction_row_id", "contribution"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prediction_row_id: Mapped[str] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    feature: Mapped[str] = mapped_column(String(128), nullable=False)
    contribution: Mapped[float] = mapped_column(Float, nullable=False)

    prediction: Mapped[PredictionRow] = relationship(back_populates="explanations")


class ModelRecord(Base, TimestampMixin):
    """Registry of world-model artifacts seen by the backend."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), default="threatcast-world-model")
    version: Mapped[str] = mapped_column(String(32), default="0.0.0")
    artifact_path: Mapped[str] = mapped_column(String(1024), default="")
    device: Mapped[str] = mapped_column(String(16), default="cpu")
    status: Mapped[str] = mapped_column(String(16), default="unavailable")  # loaded|unavailable
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    sequence_length: Mapped[int] = mapped_column(Integer, default=5)
    prediction_horizon: Mapped[int] = mapped_column(Integer, default=3)
    loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


__all__ = [
    "Dataset", "IngestionJob", "NetworkStateRow", "PredictionRow",
    "FuturePrediction", "Explanation", "ModelRecord",
]
