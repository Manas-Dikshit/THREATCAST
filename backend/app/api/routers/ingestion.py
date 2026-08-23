"""POST /ingestion/upload + GET /ingestion/jobs/{job_id}."""

from fastapi import APIRouter, File, Form, UploadFile

from ..deps import DbSession, ModelSvc
from ...core.errors import NotFoundError
from ...repositories.repositories import IngestionJobRepository
from ...schemas.api import JobStatusResponse, UploadResponse
from ...services.ingestion_service import IngestionService
from ...services.model_service import ModelService

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingestion/upload", response_model=UploadResponse, status_code=202)
async def upload(
    db: DbSession,
    model_service: ModelSvc,
    file: UploadFile = File(...),
    source_type: str | None = Form(default=None),
) -> UploadResponse:
    """Upload telemetry (CSV/PCAP/PCAPNG), run the Phase 2 pipeline and the
    world model synchronously. Returns the ingestion job id (completed)."""
    service = IngestionService(db)
    outcome = service.process_upload(file, source_name=source_type,
                                     model_service=model_service)
    return UploadResponse(job_id=outcome["job_id"], status="queued")


@router.get("/ingestion/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, db: DbSession) -> JobStatusResponse:
    job = IngestionJobRepository(db).get(job_id)
    if job is None:
        raise NotFoundError(f"Ingestion job '{job_id}' not found")
    return JobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        format=job.file_format,
        status=job.status.lower(),
        dataset_id=job.dataset_id,
        states_generated=job.states_generated,
        sequences_generated=job.sequences_generated,
        prediction_id=getattr(job.meta or {}, "prediction_id", None),
        error=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
