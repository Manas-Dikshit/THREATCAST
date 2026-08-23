"""Typed contracts for the whole pipeline (CONTRACT.md sections 5-7, Phase 2 additions)."""

from .network_state import NetworkState, NetworkStateSequence, FlowSummary
from .records import FlowRecord
from .metadata import (
    LabelKind,
    FeatureSchema,
    PreprocessingMetadata,
    DatasetProfile,
    ColumnProfile,
)

__all__ = [
    "NetworkState",
    "NetworkStateSequence",
    "FlowSummary",
    "FlowRecord",
    "LabelKind",
    "FeatureSchema",
    "PreprocessingMetadata",
    "DatasetProfile",
    "ColumnProfile",
]
