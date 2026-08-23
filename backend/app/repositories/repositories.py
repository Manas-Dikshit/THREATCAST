"""Repositories: all SQL lives here; services never touch the Session directly."""

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.tables import (
    Dataset,
    Explanation,
    FuturePrediction,
    IngestionJob,
    ModelRecord,
    NetworkStateRow,
    PredictionRow,
)
from ..schemas.prediction import PredictionResult


def new_id() -> str:
    return uuid.uuid4().hex


class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, **fields) -> Dataset:
        row = Dataset(**fields)
        self.db.add(row)
        self.db.flush()
        return row

    def get(self, dataset_id: str) -> Dataset | None:
        return self.db.get(Dataset, dataset_id)


class IngestionJobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, filename: str, file_format: str) -> IngestionJob:
        job = IngestionJob(id=f"job_{new_id()[:12]}", filename=filename,
                           file_format=file_format, status="PENDING")
        self.db.add(job)
        self.db.flush()
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        return self.db.get(IngestionJob, job_id)

    def set_status(self, job: IngestionJob, status: str,
                   error: str | None = None) -> IngestionJob:
        job.status = status
        job.error_message = error
        now = datetime.utcnow()
        if status == "PROCESSING":
            job.started_at = now
        elif status in ("COMPLETED", "FAILED"):
            job.finished_at = now
        self.db.flush()
        return job


class StateRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(self, state_row: NetworkStateRow) -> NetworkStateRow:
        self.db.add(state_row)
        self.db.flush()
        return state_row

    def add_many(self, rows: list[NetworkStateRow]) -> int:
        self.db.add_all(rows)
        self.db.flush()
        return len(rows)

    def get_by_state_id(self, state_id: str) -> NetworkStateRow | None:
        return self.db.scalar(
            select(NetworkStateRow).where(NetworkStateRow.state_id == state_id)
        )

    def for_dataset(self, dataset_id: str) -> list[NetworkStateRow]:
        return list(self.db.scalars(
            select(NetworkStateRow)
            .where(NetworkStateRow.dataset_id == dataset_id)
            .order_by(NetworkStateRow.timestamp_start)
        ))


class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def persist_result(self, result: PredictionResult, *,
                       sequence_id: str | None = None,
                       dataset_id: str | None = None,
                       input_sequence: dict | None = None) -> PredictionRow:
        row = PredictionRow(
            prediction_id=result.prediction_id,
            sequence_id=sequence_id,
            dataset_id=dataset_id,
            timestamp=result.timestamp,
            risk_score=result.risk_score,
            malicious_probability=result.malicious_probability,
            confidence=result.confidence,
            stage_id=result.predicted_stage.id,
            stage_name=result.predicted_stage.name,
            stage_confidence=result.predicted_stage.confidence,
            stage_source=result.predicted_stage.source,
            model_name=result.model.name,
            model_version=result.model.version,
            input_sequence=input_sequence or {},
        )
        self.db.add(row)
        self.db.flush()

        self.db.add_all([
            FuturePrediction(prediction_row_id=row.id, step=fs.step,
                             timestamp=fs.timestamp, features=fs.features,
                             confidence=fs.confidence)
            for fs in result.future_states
        ])
        self.db.add_all([
            Explanation(prediction_row_id=row.id, feature=fc.feature,
                        contribution=fc.contribution)
            for fc in result.feature_contributions
        ])
        self.db.flush()
        return row

    def get_by_prediction_id(self, prediction_id: str) -> PredictionRow | None:
        return self.db.scalar(
            select(PredictionRow).where(PredictionRow.prediction_id == prediction_id)
        )

    def to_result(self, row: PredictionRow) -> PredictionResult:
        from ..schemas.prediction import (
            FeatureContribution, FutureStateEntry, ModelInfo, PredictedStage,
        )

        return PredictionResult(
            prediction_id=row.prediction_id,
            timestamp=row.timestamp,
            risk_score=row.risk_score,
            malicious_probability=row.malicious_probability,
            confidence=row.confidence,
            predicted_stage=PredictedStage(
                id=row.stage_id, name=row.stage_name,
                confidence=row.stage_confidence, source=row.stage_source,
            ),
            future_states=[
                FutureStateEntry(step=fs.step, timestamp=fs.timestamp,
                                 features=fs.features, confidence=fs.confidence)
                for fs in row.future_states
            ],
            feature_contributions=[
                FeatureContribution(feature=e.feature, contribution=e.contribution)
                for e in sorted(row.explanations, key=lambda x: -abs(x.contribution))
            ],
            model=ModelInfo(name=row.model_name, version=row.model_version),
        )


class ModelRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert_active(self, *, name: str, version: str, artifact_path: str,
                      device: str, status: str, sequence_length: int,
                      prediction_horizon: int) -> ModelRecord:
        current = self.db.scalar(select(ModelRecord).where(ModelRecord.is_active.is_(True)))
        if current is None:
            current = ModelRecord()
            self.db.add(current)
        current.name = name
        current.version = version
        current.artifact_path = artifact_path
        current.device = device
        current.status = status
        current.sequence_length = sequence_length
        current.prediction_horizon = prediction_horizon
        current.is_active = True
        current.loaded_at = datetime.utcnow() if status == "loaded" else None
        self.db.flush()
        return current

    def active(self) -> ModelRecord | None:
        return self.db.scalar(select(ModelRecord).where(ModelRecord.is_active.is_(True)))


__all__ = [
    "DatasetRepository", "IngestionJobRepository", "StateRepository",
    "PredictionRepository", "ModelRepository", "new_id",
]
