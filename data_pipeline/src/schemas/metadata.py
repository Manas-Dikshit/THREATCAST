"""Pipeline metadata contracts: FeatureSchema, PreprocessingMetadata,
DatasetProfile and the LabelKind taxonomy (ground_truth / derived / unknown).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LabelKind(str, Enum):
    """Provenance of a label attached to flows/states.

    GROUND_TRUTH: label verifiably provided by the source dataset.
    DERIVED:      label inferred by THREATCAST logic (never claimed as dataset truth).
    UNKNOWN:      no label available.
    """

    GROUND_TRUTH = "ground_truth"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class FeatureSchema(BaseModel):
    """Ordered feature list consumed by ML. The order IS the contract for tensors."""

    model_config = ConfigDict(extra="allow")

    version: str = "1"
    feature_names: List[str] = Field(default_factory=list)
    description: str = ""
    created_at: Optional[datetime] = None
    source: Optional[str] = None


class PreprocessingMetadata(BaseModel):
    """Everything needed to reproduce preprocessing at inference time.

    Normalization statistics are fitted on TRAINING data only (leakage rule).
    """

    model_config = ConfigDict(extra="allow")

    preprocessing_version: str
    feature_names: List[str]
    mean: Dict[str, float] = Field(default_factory=dict)
    std: Dict[str, float] = Field(default_factory=dict)
    fit_start: Optional[datetime] = None
    fit_end: Optional[datetime] = None
    source_dataset: Optional[str] = None
    window_seconds: float = 10.0
    sequence_length: int = 5
    prediction_horizon: int = 3
    label_mappings: Dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    dtype: str
    null_count: int = 0
    null_fraction: float = 0.0
    inf_count: int = 0
    unique_count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    sample_values: List[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Dynamic inspection report of an actual dataset — nothing assumed up front."""

    model_config = ConfigDict(extra="allow")

    source_file: Optional[str] = None
    detected_format: str = "unknown"
    row_count: int = 0
    column_count: int = 0
    duplicate_rows: int = 0
    columns: List[ColumnProfile] = Field(default_factory=list)
    mapped_columns: Dict[str, str] = Field(default_factory=dict)   # canonical -> original header
    unmapped_columns: List[str] = Field(default_factory=list)
    timestamp_min: Optional[datetime] = None
    timestamp_max: Optional[datetime] = None
    label_distribution: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
