"""Configuration helpers for the data pipeline (CONTRACT.md section 3 env vars)."""

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class PipelineConfig:
    """All temporal knobs of the pipeline. Defaults follow the global contract."""

    window_seconds: float = 10.0      # TIME_WINDOW_SECONDS
    sequence_length: int = 5          # ML_SEQUENCE_LENGTH
    prediction_horizon: int = 3       # ML_PREDICTION_HORIZON

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            window_seconds=_env_float("TIME_WINDOW_SECONDS", 10.0),
            sequence_length=_env_int("ML_SEQUENCE_LENGTH", 5),
            prediction_horizon=_env_int("ML_PREDICTION_HORIZON", 3),
        )


PREPROCESSING_VERSION = "1.0.0"
