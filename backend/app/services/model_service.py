"""ModelService: owns the Phase 3 predictor. The backend never imports torch.

Loading is lazy and safe: a missing artifact leaves the service in
"unavailable" status (health + /models report it; predict -> 503) instead of
crashing the app.
"""

import logging
from pathlib import Path
from typing import Any

from ...core.errors import ModelNotLoadedError

logger = logging.getLogger("BACKEND")


class ModelStatus(dict):
    @property
    def loaded(self) -> bool:
        return bool(self.get("loaded"))


class ModelService:
    """Wraps ml.src.inference.WorldModelPredictor behind a torch-free facade."""

    def __init__(self, artifacts_dir: str, device: str = "auto"):
        self.artifacts_dir = str(Path(artifacts_dir))
        self.device = device
        self._predictor: Any | None = None
        self._status = ModelStatus(loaded=False, reason="not_loaded_yet")

    def load(self) -> ModelStatus:
        """Attempt to load the world model; never raises."""
        try:
            from ml.src.inference import WorldModelPredictor  # deferred: torch heavy

            self._predictor = WorldModelPredictor.load(self.artifacts_dir,
                                                       device=self.device)
            meta_path = Path(self.artifacts_dir) / "model_metadata.json"
            version = self._predictor.model_version
            self._status = ModelStatus(
                loaded=True, version=version, device=str(self._predictor.device),
                sequence_length=getattr(self._predictor.model, "sequence_length", 5),
                prediction_horizon=self._predictor.horizon,
                artifacts_dir=self.artifacts_dir,
                metadata_file=meta_path.name if meta_path.exists() else None,
            )
            logger.info("World model loaded: v%s on %s", version, self._predictor.device)
        except FileNotFoundError as exc:
            self._predictor = None
            self._status = ModelStatus(loaded=False, reason="artifact_missing",
                                       detail=str(exc), artifacts_dir=self.artifacts_dir)
            logger.warning("World model artifact missing: %s", exc)
        except Exception as exc:  # corrupt bundle, torch missing, bad device...
            self._predictor = None
            self._status = ModelStatus(loaded=False, reason="load_failed", detail=str(exc))
            logger.exception("World model failed to load")
        return self._status

    @property
    def status(self) -> ModelStatus:
        return self._status

    def predict(self, sequence):
        """sequence: backend NetworkStateSequence -> backend PredictionResult.

        Both are pydantic-identical to the data_pipeline/ml schemas, so we can
        hand them across directly.
        """
        if self._predictor is None:
            raise ModelNotLoadedError(
                "World model not available", **{"status": dict(self._status)}
            )
        try:
            result = self._predictor.predict(sequence)
        except ValueError as exc:
            from ...core.errors import InvalidInputError

            raise InvalidInputError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference failure")
            from ...core.errors import AppError

            raise AppError("Inference failure", code_hint="INFERENCE_FAILED") from exc
        return result

    def rollout(self, sequence, horizon: int | None = None):
        if self._predictor is None:
            raise ModelNotLoadedError("World model not available")
        return self._predictor.rollout(sequence, horizon=horizon)


__all__ = ["ModelService"]
