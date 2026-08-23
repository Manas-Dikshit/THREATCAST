"""FlowRecord: the normalized intermediate representation between parsers and
feature extraction.

Every field except `timestamp` is Optional. A value of None means "the source
did not provide it" — the pipeline never fabricates unavailable values.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .metadata import LabelKind


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class FlowRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: datetime
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = Field(default=None, ge=0, le=65535)
    dst_port: Optional[int] = Field(default=None, ge=0, le=65535)
    protocol: Optional[str] = None

    total_bytes: Optional[float] = None
    total_packets: Optional[float] = None
    duration_s: Optional[float] = None

    # TCP flag letters present on the flow, e.g. "S", "SA", "FA".
    flags: Optional[str] = None

    # Per-flow statistics when the source provides them (CIC-IDS2018 columns,
    # or packet-level aggregation from PCAP).
    iat_mean_s: Optional[float] = None
    iat_var_s: Optional[float] = None
    iat_max_s: Optional[float] = None
    fwd_bytes: Optional[float] = None
    bwd_bytes: Optional[float] = None
    fwd_packets: Optional[float] = None
    bwd_packets: Optional[float] = None
    ttl_mean: Optional[float] = None
    ttl_var: Optional[float] = None
    tcp_window_mean: Optional[float] = None
    retransmission_count: Optional[int] = None
    fragmentation_count: Optional[int] = None
    payload_mean: Optional[float] = None
    payload_max: Optional[float] = None

    # Dataset-provided ground-truth label (preserved verbatim) or None.
    label: Optional[str] = None
    label_kind: LabelKind = LabelKind.UNKNOWN

    @field_validator("timestamp")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _as_utc(v)

    def flag_letters(self) -> set[str]:
        return set(self.flags) if self.flags else set()
