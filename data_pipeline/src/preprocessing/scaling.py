"""Feature normalization with strict train-only fitting (leakage prevention).

The scaler is ALWAYS fitted on the training split; validation/test data are
only ever transformed with those frozen statistics.
"""

import numpy as np

from ..schemas.metadata import FeatureSchema, PreprocessingMetadata
from ..schemas.network_state import NetworkState
from ..utils.config import PREPROCESSING_VERSION

_EPS = 1e-12


class FeatureScaler:
    """Per-feature z-score normalizer over NetworkState.features."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, states: list[NetworkState]) -> "FeatureScaler":
        matrix = states_to_matrix(states, self.feature_names)
        self.mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        self.std = np.where(std < _EPS, 1.0, std)  # constant features stay unscaled
        return self

    def transform(self, states: list[NetworkState]) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureScaler.transform called before fit().")
        return ((states_to_matrix(states, self.feature_names) - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, states: list[NetworkState]) -> np.ndarray:
        return self.fit(states).transform(states)

    def to_metadata(
        self,
        *,
        source_dataset: str | None,
        window_seconds: float,
        sequence_length: int,
        prediction_horizon: int,
        label_mappings: dict[str, int] | None = None,
        fit_start=None,
        fit_end=None,
    ) -> PreprocessingMetadata:
        return PreprocessingMetadata(
            preprocessing_version=PREPROCESSING_VERSION,
            feature_names=self.feature_names,
            mean={n: float(m) for n, m in zip(self.feature_names, self.mean)},
            std={n: float(s) for n, s in zip(self.feature_names, self.std)},
            source_dataset=source_dataset,
            window_seconds=window_seconds,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            label_mappings=label_mappings or {},
            fit_start=fit_start,
            fit_end=fit_end,
        )


def states_to_matrix(states: list[NetworkState], feature_names: list[str]) -> np.ndarray:
    """[N, F] float64 matrix. Missing feature values become column mean later at ML;
    here they are 0.0 and tracked by the schema."""
    out = np.zeros((len(states), len(feature_names)), dtype=np.float64)
    for i, state in enumerate(states):
        for j, name in enumerate(feature_names):
            value = state.features.get(name)
            if value is not None:
                out[i, j] = float(value)
    return out


def infer_feature_schema(states: list[NetworkState], *, source: str | None = None) -> FeatureSchema:
    """Union of feature keys across states, deterministic order: first-seen then sorted?"""
    names: dict[str, None] = {}
    for state in states:
        for key in state.features:
            names.setdefault(key, None)
    ordered = sorted(names)
    return FeatureSchema(version="1", feature_names=ordered, source=source)


__all__ = ["FeatureScaler", "states_to_matrix", "infer_feature_schema"]
