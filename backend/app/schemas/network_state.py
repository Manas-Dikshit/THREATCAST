"""Canonical NetworkState and sequence contracts (CONTRACT.md §5-§6)."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlowSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    unique_source_hosts: int = 0
    unique_destination_hosts: int = 0


class NetworkState(BaseModel):
    """One time-windowed snapshot of network behaviour.

    `features` is intentionally an open dict: the feature set is NOT fixed by
    this contract and may grow in later phases without breaking consumers.
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
