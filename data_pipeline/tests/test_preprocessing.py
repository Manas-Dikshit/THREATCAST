"""Preprocessing tests: cleaning, profiling, normalization, leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.preprocessing.cleaning import clean_flow_table
from data_pipeline.src.preprocessing.profiler import profile_dataframe
from data_pipeline.src.preprocessing.scaling import (
    FeatureScaler,
    infer_feature_schema,
    states_to_matrix,
)
from data_pipeline.src.schemas.metadata import LabelKind, PreprocessingMetadata
from datetime import datetime, timezone


def _table():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:02", "2026-01-01 10:00:01", "2026-01-01 10:00:01",
            "2026-01-01 10:00:03",
        ]),
        # rows 2+3 identical after +-inf -> NaN nulling -> exact duplicates
        "total_bytes": [100.0, np.inf, np.inf, -np.inf],
        "total_packets": [5, 5, 5, 7],
        "label": ["BENIGN"] * 4,
    })


def test_cleaning_removes_duplicates_and_infinity():
    cleaned, stats = clean_flow_table(_table())
    assert stats.rows_in == 4 and stats.rows_out == 3
    assert stats.duplicates_removed == 1
    assert stats.infinite_values_nulled == 3
    assert not np.isinf(cleaned["total_bytes"]).any()
    assert cleaned["total_bytes"].isna().sum() == 2   # one NaN row removed as duplicate


def test_cleaning_sorts_chronologically():
    cleaned, _ = clean_flow_table(_table())
    ts = list(cleaned["timestamp"])
    assert ts == sorted(ts)


def test_profiler_reports_raw_truth():
    profile = profile_dataframe(_table(), source_file="t.csv")
    assert profile.row_count == 4 and profile.column_count == 4
    assert profile.duplicate_rows == 1
    by_name = {c.name: c for c in profile.columns}
    assert by_name["total_packets"].mean_value == pytest.approx(5.5)
    assert by_name["label"].sample_values == ["BENIGN"]
    assert profile.label_distribution == {"BENIGN": 4}


def test_scaler_fit_transform_deterministic(sample_records):
    states = [
        _state([1.0, 10.0]),
        _state([3.0, 30.0]),
        _state([2.0, 20.0]),
    ]
    scaler = FeatureScaler(["a", "b"]).fit(states)
    out = scaler.transform(states)
    assert out.shape == (3, 2)
    assert out.mean(axis=0) == pytest.approx(0.0, abs=1e-6)
    again = FeatureScaler(["a", "b"]).fit(states).transform(states)
    assert np.allclose(out, again)


def test_constant_feature_guard_and_missing_values():
    states = [_state([5.0, 0.0]), _state([5.0, 2.0])]
    scaler = FeatureScaler(["const", "x"]).fit(states)
    # 'const' has zero variance -> std forced to 1 (no divide-by-zero blowup)
    assert scaler.std[0] == 1.0


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureScaler(["a"]).transform([])


def test_states_to_matrix_fills_missing_with_zero():
    s1 = _state({"a": 1.0})
    m = states_to_matrix([s1], ["a", "b"])   # 'b' absent -> 0
    assert m[0][0] == 1.0 and m[0][1] == 0.0


def test_feature_schema_union_sorted_deterministic():
    schema = infer_feature_schema([_state({"b": 1, "a": 2}), _state({"c": 3})])
    assert schema.feature_names == ["a", "b", "c"]


def test_leakage_scaler_uses_train_only(sample_records):
    """Fit on train window only; val/test must be transformed with frozen stats."""
    train = [_state({"f": 10.0}), _state({"f": 20.0})]
    val = [_state({"f": 900.0})]
    scaler = FeatureScaler(["f"]).fit(train)
    transformed_val = scaler.transform(val)[0][0]
    expected = (900.0 - 15.0) / 5.0     # train mean/std, NOT including val
    assert transformed_val == pytest.approx(expected, abs=1e-5)


def test_preprocessing_metadata_round_trip():
    meta = PreprocessingMetadata(
        preprocessing_version="1.0.0",
        feature_names=["a", "b"],
        mean={"a": 1.0}, std={"a": 2.0},
        fit_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        fit_end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        label_mappings={"BENIGN": 0},
    )
    restored = PreprocessingMetadata.model_validate_json(meta.model_dump_json())
    assert restored.mean["a"] == 1.0 and restored.fit_end is not None


def test_label_kind_taxonomy():
    assert {k.value for k in LabelKind} == {"ground_truth", "derived", "unknown"}


def _state(features) -> "object":
    from datetime import datetime, timedelta, timezone

    from data_pipeline.src.schemas.network_state import NetworkState

    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    feats = {k: float(v) for k, v in features.items()} if isinstance(features, dict) else {}
    if isinstance(features, (list, tuple)):
        names = ["a", "b", "f", "const", "x"]
        feats = dict(zip(names, map(float, features)))
    return NetworkState(
        state_id="state_000001",
        timestamp_start=t,
        timestamp_end=t + timedelta(seconds=10),
        features=feats,
    )
