"""THREATCAST data pipeline (Module 1).

Raw telemetry -> parsers -> normalization -> feature extraction -> time
windows -> canonical NetworkState / NetworkStateSequence -> ML-ready dataset.
"""

from .src.pipeline import PipelineResult, run_pipeline
from .src.schemas import (
    DatasetProfile,
    FeatureSchema,
    FlowRecord,
    LabelKind,
    NetworkState,
    NetworkStateSequence,
    PreprocessingMetadata,
)
from .src.utils.config import PREPROCESSING_VERSION, PipelineConfig

__all__ = [
    "run_pipeline",
    "PipelineResult",
    "NetworkState",
    "NetworkStateSequence",
    "FlowRecord",
    "FeatureSchema",
    "PreprocessingMetadata",
    "DatasetProfile",
    "LabelKind",
    "PipelineConfig",
    "PREPROCESSING_VERSION",
]
