"""Canonical NetworkState / NetworkStateSequence (CONTRACT.md sections 5-6).

Kept schema-identical to backend/app/schemas/network_state.py; a compat test
guards drift between the two implementations.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlowSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    unique_source_hosts: int = 0
    unique_destination_hosts: int = 0


class NetworkState(BaseModel):
    """One time-windowed snapshot of network behaviour.

    `features` is an open dict: the feature set is NOT hard-coded and may be
    extended additively without breaking consumers.
    """

    model_config = ConfigDict(extra="allow")

    state_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    window_seconds: float = Field(default_factory=lambda: 10.0)
    features: Dict[str, float] = Field(default_factory=dict)
    flow_summary: FlowSummary = Field(default_factory=FlowSummary)
    label: Optional[str] = None
    label_source: Optional[str] = None


class NetworkStateSequence(BaseModel):
    """Ordered window of states consumed by the world model."""

    model_config = ConfigDict(extra="allow")

    sequence_id: str
    states: list[NetworkState] = Field(default_factory=list)
    sequence_length: int = 5
    window_seconds: float = 10.0
    target_state: Optional[NetworkState] = None


def new_state_id() -> str:
    return f"state_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "NetworkState",
    "NetworkStateSequence",
    "FlowSummary",
    "new_state_id",
    "utcnow",
]
