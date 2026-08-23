"""End-to-end pipeline orchestration:

    raw dataset -> parser -> normalization -> windows -> NetworkState
                -> NetworkStateSequence -> ML-ready dataset artifacts
"""

from dataclasses import dataclass, field
from pathlib import Path

from .ingestion.loader import load_telemetry
from .preprocessing.profiler import save_profile
from .preprocessing.scaling import FeatureScaler, infer_feature_schema
from .schemas.metadata import FeatureSchema, PreprocessingMetadata
from .schemas.network_state import NetworkState, NetworkStateSequence
from .utils.config import PREPROCESSING_VERSION, PipelineConfig
from .windows.dataset import export_ml_dataset
from .windows.engine import build_states
from .windows.sequences import build_sequences, split_time_aware


@dataclass
class PipelineResult:
    states: list[NetworkState]
    sequences: list[NetworkStateSequence]
    source_type: str
    profile_json_path: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    config: PipelineConfig | None = None,
    source_name: str | None = None,
    fit_scaler_on_train: bool = True,
) -> PipelineResult:
    """Run ingestion -> features -> windows -> sequences (+ optional export).

    Args:
        input_path: telemetry file (.csv / .pcap / .pcapng).
        out_dir: when given, writes the ML-ready dataset artifacts there.
        config: temporal knobs (defaults from env / contract defaults).
        source_name: verified dataset name used as label_source for ground truth.
        fit_scaler_on_train: fit normalization on the train split only.
    """
    cfg = config or PipelineConfig.from_env()
    loaded = load_telemetry(input_path)
    if not loaded.records:
        raise ValueError(
            f"No usable flow records parsed from '{input_path}'. "
            f"Warnings: {loaded.profile.warnings}"
        )

    states = build_states(loaded.records, window_seconds=cfg.window_seconds, source_name=source_name)
    sequences = build_sequences(
        states,
        sequence_length=cfg.sequence_length,
        prediction_horizon=cfg.prediction_horizon,
    )
    feature_schema = infer_feature_schema(states, source=str(input_path))

    result = PipelineResult(states=states, sequences=sequences, source_type=loaded.source_type)

    scaler: FeatureScaler | None = None
    metadata: PreprocessingMetadata | None = None
    if out_dir is not None:
        splits = split_time_aware(sequences)
        train_states = [s for seq in splits["train"] for s in seq.states]
        if fit_scaler_on_train and train_states:
            scaler = FeatureScaler(feature_schema.feature_names).fit(train_states)
            metadata = scaler.to_metadata(
                source_dataset=source_name or str(input_path),
                window_seconds=cfg.window_seconds,
                sequence_length=cfg.sequence_length,
                prediction_horizon=cfg.prediction_horizon,
                label_mappings={},
                fit_start=train_states[0].timestamp_start if train_states else None,
                fit_end=train_states[-1].timestamp_end if train_states else None,
            )
            metadata.label_mappings = _label_ids(states)

        result.artifacts = export_ml_dataset(
            out_dir,
            states=states,
            sequences=sequences,
            feature_schema=feature_schema,
            metadata=metadata,
            profile=loaded.profile,
            scaler=scaler,
        )
        result.profile_json_path = result.artifacts.get("dataset_profile")
    elif loaded.profile:
        # Even without an export dir, persist the profiling report next to the input.
        sidecar = Path(input_path).with_suffix(".profile.json")
        result.profile_json_path = save_profile(loaded.profile, sidecar)

    return result


def _label_ids(states: list[NetworkState]) -> dict[str, int]:
    labels = sorted({s.label for s in states if s.label})
    return {label: idx for idx, label in enumerate(labels)}


__all__ = ["run_pipeline", "PipelineResult", "PREPROCESSING_VERSION", "FeatureSchema"]
