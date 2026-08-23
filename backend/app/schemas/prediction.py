"""Canonical PredictionResult contract (CONTRACT.md §7) and ML interface (§8)."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    name: str = "threatcast-world-model"
    version: str = "0.1.0"


class PredictedStage(BaseModel):
    """Derived attack stage. Never claimed as dataset ground truth unless verified."""

    id: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class FutureStateEntry(BaseModel):
    """K-step forward-simulated state (schema may be enriched later)."""

    step: int
    timestamp: Optional[datetime] = None
    features: Dict[str, float] = Field(default_factory=dict)
    confidence: Optional[float] = None


class PredictionResult(BaseModel):
    """Stable output of the future world-model interface:

        predict(NetworkStateSequence) -> PredictionResult
    """

    model_config = ConfigDict(extra="allow")

    prediction_id: str
    timestamp: datetime
    risk_score: float = Field(ge=0.0, le=1.0)
    malicious_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    predicted_stage: PredictedStage = Field(default_factory=PredictedStage)
    future_states: List[FutureStateEntry] = Field(default_factory=list)
    feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    model: ModelInfo = Field(default_factory=ModelInfo)


def empty_result(prediction_id: str) -> PredictionResult:
    """Neutral placeholder result for scaffolding/tests (no fake intelligence)."""
    return PredictionResult(
        prediction_id=prediction_id,
        timestamp=datetime.utcnow(),
        risk_score=0.0,
        malicious_probability=0.0,
        confidence=0.0,
    )


# Type alias documenting the future ML entry point signature.
PredictFn = Any  # Callable[[NetworkStateSequence], PredictionResult] once implemented
