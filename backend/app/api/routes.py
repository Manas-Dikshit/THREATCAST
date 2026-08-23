"""API v1 routers. Only /health is implemented in Phase 1; the rest are contract stubs."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health() -> dict:
    """Liveness/readiness probe."""
    return {"status": "ok", "service": "threatcast-backend"}
