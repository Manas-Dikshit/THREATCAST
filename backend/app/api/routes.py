"""API v1 route aggregation.

Each resource lives in its own module; this file wires them together and owns
health + model status.
"""

import logging

from fastapi import APIRouter
from sqlalchemy import text

from ..core.config import get_settings
from ..repositories.repositories import ModelRepository
from ..schemas.api import HealthResponse, ModelsListResponse, ModelStatusInfo
from .deps import DbSession, ModelSvc
from .routers import ingestion, predictions, states

logger = logging.getLogger("BACKEND")

router = APIRouter(prefix="/api/v1")
router.include_router(ingestion.router)
router.include_router(predictions.router)
router.include_router(states.router)


@router.get("/health", response_model=HealthResponse)
def health(db: DbSession, model_service: ModelSvc) -> HealthResponse:
    """Liveness/readiness probe incl. model + database status."""
    database = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - health must never raise
        database = "unavailable"

    live = dict(model_service.status) if model_service is not None else {"loaded": False}
    return HealthResponse(
        status="ok",
        service="threatcast-backend",
        version="0.1.0",
        model=live,
        database=database,
    )


@router.get("/models", response_model=ModelsListResponse)
def models(db: DbSession, model_service: ModelSvc) -> ModelsListResponse:
    """Live world-model status plus what the backend has served before."""
    live = dict(model_service.status) if model_service is not None else {"loaded": False}
    info = ModelStatusInfo(
        status="loaded" if live.get("loaded") else "unavailable",
        version=live.get("version"),
        device=live.get("device"),
        sequence_length=live.get("sequence_length"),
        prediction_horizon=live.get("prediction_horizon"),
        artifacts_dir=live.get("artifacts_dir"),
        reason=live.get("reason"),
    )
    reg_row = ModelRepository(db).active()
    registry = []
    if reg_row is not None:
        registry.append(ModelStatusInfo(
            name=reg_row.name, version=reg_row.version, device=reg_row.device,
            status=reg_row.status, sequence_length=reg_row.sequence_length,
            prediction_horizon=reg_row.prediction_horizon,
            loaded_at=reg_row.loaded_at, artifacts_dir=reg_row.artifact_path or None,
        ))
    return ModelsListResponse(active=info, registry=registry)
