"""Bridge helpers between backend persistence and the Phase 2/3 layers."""

import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models.tables import NetworkStateRow, Dataset

# make `data_pipeline` and `ml` importable when the app runs from anywhere
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def latest_sequence_for_dataset(db: Session, dataset_id: str | None):
    """Rebuild the most recent NetworkStateSequence from stored states.

    Uses the Phase 3 vectorizer's expected length via the sequence contract
    defaults; falls back to None when fewer states than a full window exist.
    """
    if dataset_id is None:
        return None
    rows = db.scalars(
        select(NetworkStateRow)
        .where(NetworkStateRow.dataset_id == dataset_id)
        .order_by(NetworkStateRow.timestamp_start)
    ).all()
    from .schemas.network_state import (
        FlowSummary, NetworkState, NetworkStateSequence,
    )

    settings = _ml_sequence_length()
    L = settings
    if len(rows) < L:
        return None
    window = rows[-L:]
    states = [
        NetworkState(
            state_id=r.state_id,
            timestamp_start=r.timestamp_start,
            timestamp_end=r.timestamp_end,
            window_seconds=r.window_seconds,
            features=r.features or {},
            label=r.label,
            label_source=r.label_source,
            flow_summary=FlowSummary(),
        )
        for r in window
    ]
    return NetworkStateSequence(
        sequence_id=f"seq_{dataset_id[:12]}_{window[-1].state_id}",
        states=states, sequence_length=L,
        window_seconds=window[0].window_seconds or 10.0,
    )


def datasets_exist(db: Session) -> bool:
    return db.scalar(select(Dataset.id).limit(1)) is not None


__all__ = ["latest_sequence_for_dataset", "datasets_exist"]
