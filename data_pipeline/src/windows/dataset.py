"""ML-ready dataset export: Parquet + JSON + NumPy tensors with full metadata.

Output layout (written into the target directory):

  states.parquet              flattened NetworkState table (one row per state)
  sequences.jsonl             full NetworkStateSequence contract objects
  tensors.npz                 X [N,L,F], Y [N,F], target_mask [N], label_ids [N]
  feature_schema.json         ordered feature names (the tensor column order)
  preprocessing_metadata.json normalization stats + run configuration
  label_mappings.json         label string -> integer id
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..preprocessing.scaling import FeatureScaler, states_to_matrix
from ..schemas.metadata import DatasetProfile, FeatureSchema, PreprocessingMetadata
from ..schemas.network_state import NetworkState, NetworkStateSequence


def states_to_frame(states: list[NetworkState], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for state in states:
        row = {
            "state_id": state.state_id,
            "timestamp_start": state.timestamp_start,
            "timestamp_end": state.timestamp_end,
            "window_seconds": state.window_seconds,
            "label": state.label,
            "label_source": state.label_source,
        }
        for name in feature_names:
            row[name] = state.features.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_tensors(
    sequences: list[NetworkStateSequence],
    feature_names: list[str],
    *,
    scaler: FeatureScaler | None = None,
) -> dict[str, np.ndarray]:
    """X: input windows; Y: target future-state features; label ids for targets."""
    n = len(sequences)
    L = len(feature_names)
    X = np.zeros((n, sequences[0].sequence_length if sequences else 0, L), dtype=np.float32) if n else np.zeros((0, 0, L), np.float32)
    Y = np.zeros((n, L), dtype=np.float32)
    target_mask = np.zeros(n, dtype=bool)
    labels: list[str | None] = []

    def vec(state: NetworkState) -> np.ndarray:
        return states_to_matrix([state], feature_names)[0]

    for i, seq in enumerate(sequences):
        inputs = np.stack([vec(s) for s in seq.states])
        if scaler is not None:
            inputs = (inputs - scaler.mean) / scaler.std
        X[i] = inputs.astype(np.float32)
        if seq.target_state is not None:
            Y[i] = vec(seq.target_state).astype(np.float32)
            target_mask[i] = True
        labels.append(seq.states[-1].label)

    known = sorted({lab for lab in labels if lab})
    mapping = {lab: idx for idx, lab in enumerate(known)}
    label_ids = np.array([mapping.get(lab, -1) for lab in labels], dtype=np.int64)

    return {"X": X, "Y": Y, "target_mask": target_mask, "label_ids": label_ids, "label_mapping": mapping}


def export_ml_dataset(
    out_dir: str | Path,
    *,
    states: list[NetworkState],
    sequences: list[NetworkStateSequence],
    feature_schema: FeatureSchema,
    metadata: PreprocessingMetadata | None = None,
    profile: DatasetProfile | None = None,
    scaler: FeatureScaler | None = None,
) -> dict[str, str]:
    """Write the complete ML-ready dataset. Returns artifact paths by role."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = feature_schema.feature_names

    paths: dict[str, str] = {}

    frame = states_to_frame(states, names)
    parquet_path = out / "states.parquet"
    frame.to_parquet(parquet_path, index=False)
    paths["states_parquet"] = str(parquet_path)

    jsonl_path = out / "sequences.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for seq in sequences:
            fh.write(seq.model_dump_json() + "\n")
    paths["sequences_jsonl"] = str(jsonl_path)

    tensors = build_tensors(sequences, names, scaler=scaler)
    mapping: dict[str, int] = tensors.pop("label_mapping")
    npz_path = out / "tensors.npz"
    np.savez_compressed(npz_path, **tensors)
    paths["tensors_npz"] = str(npz_path)

    schema_path = out / "feature_schema.json"
    schema_path.write_text(feature_schema.model_dump_json(indent=2), encoding="utf-8")
    paths["feature_schema"] = str(schema_path)

    labels_path = out / "label_mappings.json"
    labels_path.write_text(
        json.dumps({"labels": mapping, "unlabeled_id": -1}, indent=2), encoding="utf-8"
    )
    paths["label_mappings"] = str(labels_path)

    if metadata is not None:
        meta_path = out / "preprocessing_metadata.json"
        meta_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        paths["preprocessing_metadata"] = str(meta_path)

    if profile is not None:
        profile_path = out / "dataset_profile.json"
        profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        paths["dataset_profile"] = str(profile_path)

    return paths


__all__ = ["export_ml_dataset", "build_tensors", "states_to_frame"]
