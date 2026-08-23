"""StateVectorizer: NetworkStateSequence -> normalized float tensor [L, F].

Consumes the Phase 2 artifacts directly:
  - feature_schema.json  (ordered feature_names = tensor column order)
  - preprocessing_metadata.json (per-feature mean/std, fitted on train only)

Missing features are imputed with the train mean (i.e. become 0.0 after
normalization) — consistent between training and inference. This differs from
data_pipeline's states_to_matrix (raw 0.0 fill for parquet export); the ML
path must not inject fake zeros into z-scores.
"""

import json
from pathlib import Path

import numpy as np

from data_pipeline.src.schemas.network_state import NetworkState, NetworkStateSequence


class StateVectorizer:
    def __init__(self, feature_names: list[str], mean: dict[str, float], std: dict[str, float]):
        if mean.keys() != set(feature_names) or std.keys() != set(feature_names):
            raise ValueError("normalization stats do not match feature schema")
        self.feature_names = list(feature_names)
        self.mean = np.array([mean[n] for n in self.feature_names], dtype=np.float32)
        self.std = np.array([std[n] for n in self.feature_names], dtype=np.float32)
        self.std[self.std < 1e-8] = 1.0  # constant-feature guard (mirrors FeatureScaler)

    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path) -> "StateVectorizer":
        d = Path(artifacts_dir)
        names = json.loads((d / "feature_schema.json").read_text(encoding="utf-8"))["feature_names"]
        meta = json.loads((d / "preprocessing_metadata.json").read_text(encoding="utf-8"))
        return cls(names, meta["mean"], meta["std"])

    def state_vector(self, state: NetworkState) -> np.ndarray:
        raw = np.array(
            [state.features.get(n, np.nan) for n in self.feature_names], dtype=np.float32
        )
        norm = (raw - self.mean) / self.std
        return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def sequence_tensor(self, sequence: NetworkStateSequence) -> np.ndarray:
        """[L, F] float32; validates length against the model's expectation."""
        L = len(sequence.states)
        expected = sequence.sequence_length
        if L != expected:
            raise ValueError(f"INVALID_SEQUENCE: got {L} states, expected {expected}")
        return np.stack([self.state_vector(s) for s in sequence.states])

    def denormalize(self, normalized: np.ndarray) -> dict[str, float]:
        """normalized [F] -> named feature values in original units."""
        vec = np.asarray(normalized, dtype=np.float32) * self.std + self.mean
        return {n: float(v) for n, v in zip(self.feature_names, vec)}

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)


__all__ = ["StateVectorizer"]
