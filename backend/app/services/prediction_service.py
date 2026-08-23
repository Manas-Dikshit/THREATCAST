"""PredictionService: validate -> model.predict -> persist full result."""

import logging

from ..core.errors import ModelNotLoadedError
from ..models.tables import Dataset
from ..repositories.repositories import PredictionRepository
from ...schemas.prediction import PredictionResult
from ...schemas.network_state import NetworkStateSequence

logger = logging.getLogger("BACKEND")


class PredictionService:
    def __init__(self, db, model_service):
        self.db = db
        self.model_service = model_service
        self.repo = PredictionRepository(db)

    def predict_and_store(self, sequence: NetworkStateSequence,
                          dataset_id: str | None = None) -> PredictionResult:
        if self.model_service is None or not self.model_service.status.loaded:
            raise ModelNotLoadedError(
                "World model is not available", artifacts="ml/artifacts"
            )
        if not sequence.states:
            from ..core.errors import InvalidInputError

            raise InvalidInputError("Sequence contains no states.")
        if len(sequence.states) != sequence.sequence_length:
            raise InvalidInputError(
                f"INVALID_SEQUENCE: got {len(sequence.states)} states, "
                f"expected {sequence.sequence_length}"
            )

        result = self.model_service.predict(sequence)
        self.repo.persist_result(
            result, sequence_id=sequence.sequence_id, dataset_id=dataset_id,
            input_sequence=sequence.model_dump(mode="json"),
        )
        self.db.commit()
        return result

    def get(self, prediction_id: str) -> tuple[PredictionResult, object]:
        row = self.repo.get_by_prediction_id(prediction_id)
        if row is None:
            from ..core.errors import NotFoundError

            raise NotFoundError(f"Prediction '{prediction_id}' not found")
        return self.repo.to_result(row), row


__all__ = ["PredictionService"]
