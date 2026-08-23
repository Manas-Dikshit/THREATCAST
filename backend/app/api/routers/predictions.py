"""POST /predict, GET /predictions/{id}, GET /predictions/{id}/timeline."""

from fastapi import APIRouter

from ..deps import DbSession, ModelSvc
from ...schemas.api import TimelineResponse, TimelineStep
from ...schemas.network_state import NetworkStateSequence
from ...schemas.prediction import PredictionResult as PredictionResultSchema
from ...services.prediction_service import PredictionService

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResultSchema)
def predict(
    sequence: NetworkStateSequence,
    db: DbSession,
    model_service: ModelSvc,
) -> PredictionResultSchema:
    """Run the world model on a state sequence; persists the full result."""
    service = PredictionService(db, model_service)
    return service.predict_and_store(sequence)


@router.get("/predictions/{prediction_id}", response_model=PredictionResultSchema)
def get_prediction(prediction_id: str, db: DbSession) -> PredictionResultSchema:
    result, _row = PredictionService(db, None).get(prediction_id)
    return result


@router.get("/predictions/{prediction_id}/timeline", response_model=TimelineResponse)
def timeline(prediction_id: str, db: DbSession) -> TimelineResponse:
    result, row = PredictionService(db, None).get(prediction_id)
    steps = [
        TimelineStep(step=0, timestamp=result.timestamp,
                     risk_score=result.risk_score,
                     malicious_probability=result.malicious_probability,
                     confidence=result.confidence)
    ]
    for fs in result.future_states:
        steps.append(TimelineStep(step=fs.step, timestamp=fs.timestamp,
                                  risk_score=result.risk_score,
                                  malicious_probability=None,
                                  confidence=fs.confidence))
    return TimelineResponse(
        prediction_id=result.prediction_id, generated_at=row.created_at,
        risk_timeline=steps,
    )
