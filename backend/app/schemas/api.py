"""API response schemas not covered by CONTRACT.md canonical models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "threatcast-backend"
    version: str = "0.1.0"
    model: dict[str, Any] = Field(default_factory=dict)
    database: str = "ok"


class UploadResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    format: str
    status: str
    dataset_id: str | None = None
    states_generated: int = 0
    sequences_generated: int = 0
    prediction_id: str | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TimelineStep(BaseModel):
    step: int
    timestamp: datetime | None = None
    risk_score: float
    malicious_probability: float | None = None
    confidence: float | None = None


class TimelineResponse(BaseModel):
    prediction_id: str
    generated_at: datetime
    risk_timeline: list[TimelineStep]


class ModelStatusInfo(BaseModel):
    name: str = "threatcast-world-model"
    version: str | None = None
    device: str | None = None
    status: str  # loaded | unavailable
    sequence_length: int | None = None
    prediction_horizon: int | None = None
    loaded_at: datetime | None = None
    artifacts_dir: str | None = None
    reason: str | None = None


class ModelsListResponse(BaseModel):
    active: ModelStatusInfo
    registry: list[ModelStatusInfo] = Field(default_factory=list)


__all__ = [
    "HealthResponse", "UploadResponse", "JobStatusResponse", "TimelineResponse",
    "TimelineStep", "ModelsListResponse", "ModelStatusInfo",
]
