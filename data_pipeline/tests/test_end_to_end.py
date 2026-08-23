"""End-to-end validation:

raw dataset -> parser -> normalization -> features -> windows ->
NetworkState -> NetworkStateSequence -> ML-ready artifacts.
"""

import json

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.pipeline import run_pipeline
from data_pipeline.src.schemas.network_state import NetworkState, NetworkStateSequence
from data_pipeline.src.utils.config import PipelineConfig


@pytest.fixture
def config():
    return PipelineConfig(window_seconds=10.0, sequence_length=2, prediction_horizon=1)


def test_full_pipeline_csv_to_ml_dataset(generic_csv, tmp_path, config):
    out = tmp_path / "ml_dataset"
    result = run_pipeline(generic_csv, out, config=config, source_name="synthetic")

    # --- stages ---
    assert result.source_type == "csv"
    assert len(result.states) == 3                      # windows [00-10) [10-20) [20-30)
    assert len(result.sequences) == 2                   # L=2 over 3 contiguous states

    seq0 = result.sequences[0]
    assert seq0.sequence_length == 2 and seq0.window_seconds == 10.0
    assert seq0.target_state is not None                # S(t+1) exists for the first
    assert result.sequences[1].target_state is None     # horizon beyond the run

    # --- artifacts ---
    for role in ("states_parquet", "sequences_jsonl", "tensors_npz",
                 "feature_schema", "label_mappings", "preprocessing_metadata",
                 "dataset_profile"):
        assert role in result.artifacts, f"missing artifact: {role}"

    frame = pd.read_parquet(result.artifacts["states_parquet"])
    assert len(frame) == 3 and {"state_id", "timestamp_start"} <= set(frame.columns)

    schema = json.loads((out / "feature_schema.json").read_text(encoding="utf-8"))
    tensors = np.load(result.artifacts["tensors_npz"])
    f_dim = len(schema["feature_names"])
    assert tensors["X"].shape[0] == len(result.sequences)
    assert tensors["X"].shape[2] == f_dim
    assert tensors["target_mask"].sum() == sum(
        1 for s in result.sequences if s.target_state is not None
    )

    with open(result.artifacts["sequences_jsonl"], encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == len(result.sequences)
    parsed = [NetworkStateSequence.model_validate_json(line) for line in lines]
    assert parsed[0].states[0].features == result.sequences[0].states[0].features

    metadata = json.loads((out / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["feature_names"]) == set(schema["feature_names"])
    assert set(metadata["mean"]) == set(schema["feature_names"])   # scaler fit happened
    assert metadata["window_seconds"] == 10.0

    mappings = json.loads((out / "label_mappings.json").read_text(encoding="utf-8"))
    assert "BENIGN" in mappings["labels"]


def test_states_validate_against_backend_canonical(generic_csv, tmp_path, config):
    """Guard against contract drift between data_pipeline and backend schemas."""
    from app.schemas import NetworkState as BackendNetworkState

    result = run_pipeline(generic_csv, None, config=config)
    for state in result.states[:2]:
        backend_view = BackendNetworkState.model_validate_json(state.model_dump_json())
        assert backend_view.features["flow_count"] == state.features["flow_count"]


def test_profile_sidecar_without_export_dir(generic_csv, tmp_path, config):
    result = run_pipeline(generic_csv, None, config=config)
    assert result.profile_json_path is not None
    assert result.profile_json_path.endswith(".profile.json")


def test_empty_input_raises(empty_csv):
    with pytest.raises(ValueError, match="No usable flow records"):
        run_pipeline(empty_csv, None)


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "nope.csv")


def test_cli_smoke(generic_csv, tmp_path, capsys):
    from data_pipeline.cli import main

    code = main(["--input", str(generic_csv), "--out", str(tmp_path / "ds")])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["states"] == 3 and summary["source_type"] == "csv"


def test_cli_rejects_bad_input(garbage_csv, capsys):
    from data_pipeline.cli import main

    assert main(["--input", str(garbage_csv)]) == 1
