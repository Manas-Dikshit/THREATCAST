"""GET /states/{state_id}."""

from fastapi import APIRouter

from ..deps import DbSession
from ...core.errors import NotFoundError
from ...repositories.repositories import StateRepository
from ...schemas.network_state import (
    FlowSummary, NetworkState as NetworkStateSchema,
)

router = APIRouter(tags=["states"])


@router.get("/states/{state_id}", response_model=NetworkStateSchema)
def get_state(state_id: str, db: DbSession) -> NetworkStateSchema:
    row = StateRepository(db).get_by_state_id(state_id)
    if row is None:
        raise NotFoundError(f"State '{state_id}' not found")
    return NetworkStateSchema(
        state_id=row.state_id,
        timestamp_start=row.timestamp_start,
        timestamp_end=row.timestamp_end,
        window_seconds=row.window_seconds,
        features=row.features or {},
        flow_summary=FlowSummary(),
        label=row.label,
        label_source=row.label_source,
    )
