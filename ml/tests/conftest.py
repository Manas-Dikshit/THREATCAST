"""Synthetic fixtures — no CIC-IDS2018 data required."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

FEATURES = [
    "flow_count", "packet_count", "byte_count", "syn_ratio", "ack_ratio",
    "mean_iat", "iat_variance", "unique_dst_ports", "port_scan_score",
]
L, F = 5, len(FEATURES)
N = 64
WINDOW_SECONDS = 10.0


def make_state(i: int, values: np.ndarray, label: str | None = None) -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i * WINDOW_SECONDS)
    return {
        "state_id": f"state_{i:06d}",
        "timestamp_start": start,
        "timestamp_end": start + timedelta(seconds=WINDOW_SECONDS),
        "window_seconds": WINDOW_SECONDS,
        "features": {n: float(v) for n, v in zip(FEATURES, values)},
        "label": label,
        "label_source": "synthetic" if label else None,
    }


def make_sequence(values_window: np.ndarray, seq_id: str, labels: list[str] | None = None,
                  target_values: np.ndarray | None = None):
    from data_pipeline.src.schemas.network_state import NetworkStateSequence

    states = [make_state(i, v, (labels[i] if labels else None))
              for i, v in enumerate(values_window)]
    target = None
    if target_values is not None:
        target = make_state(len(values_window), target_values)
    return NetworkStateSequence(
        sequence_id=seq_id, states=states, sequence_length=L,
        window_seconds=WINDOW_SECONDS, target_state=target,
    )


@pytest.fixture(scope="session")
def synthetic_export(tmp_path_factory):
    """A learnable linear-AR dataset in Phase 2 export format.

    Dynamics: x[t+1] = A @ x[t] + small noise. The world model CAN learn this.
    Half the sequences end labeled BENIGN, half with an attack-family label.
    """
    rng = np.random.default_rng(7)
    d = tmp_path_factory.mktemp("phase2_export")

    A = rng.normal(0, 0.3, size=(F, F)) * 0.5 + np.eye(F) * 0.5
    X, Y, labels = [], [], []
    for i in range(N):
        x = rng.normal(0, 1, size=(L, F)).astype(np.float32)
        for t in range(L):  # give the window real internal dynamics
            nxt = (A @ x[t] + rng.normal(0, 0.05, F)).astype(np.float32)
            if t < L - 1:
                x[t + 1] = nxt
        X.append(x)
        Y.append((A @ x[-1]).astype(np.float32))  # true S(t+1), no noise
        labels.append("BENIGN" if i % 2 == 0 else "DDoS")

    X = np.stack(X)
    Y = np.stack(Y)
    np.savez_compressed(
        d / "tensors.npz",
        X=X.astype(np.float32), Y=Y.astype(np.float32),
        target_mask=np.ones(N, dtype=bool),
        label_ids=np.array([0 if lab == "BENIGN" else 1 for lab in labels], dtype=np.int64),
    )
    (d / "feature_schema.json").write_text(
        json.dumps({"schema_version": 1, "source": "synthetic", "feature_names": FEATURES}),
        encoding="utf-8",
    )
    mean = {n: float(v) for n, v in zip(FEATURES, X.reshape(-1, F).mean(axis=0))}
    std = {n: max(float(v), 1e-6) for n, v in zip(FEATURES, X.reshape(-1, F).std(axis=0))}
    (d / "preprocessing_metadata.json").write_text(
        json.dumps({
            "preprocessing_version": "1.0", "feature_names": FEATURES,
            "mean": mean, "std": std,
            "fit_start": "2026-01-01T00:00:00+00:00",
            "fit_end": datetime.now(timezone.utc).isoformat(),
            "normalization": "zscore_train_only", "source_dataset": "synthetic",
        }),
        encoding="utf-8",
    )
    (d / "label_mappings.json").write_text(
        json.dumps({"labels": {"BENIGN": 0, "DDoS": 1}, "unlabeled_id": -1}), encoding="utf-8"
    )
    return {"dir": d, "mean": mean, "std": std}


def make_tiny_config():
    """Minimal config for fast CPU tests."""
    from ml.src.utils.config import WorldModelConfig

    cfg = WorldModelConfig()
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.d_ff = 64
    cfg.sequence.length = L
    cfg.training.epochs = 3
    cfg.training.batch_size = 16
    cfg.training.grad_accumulation = 1
    cfg.training.amp = False
    cfg.training.device = "cpu"
    cfg.training.num_workers = 0
    cfg.training.checkpoint_every = 1
    return cfg


@pytest.fixture(scope="session")
def trained_artifacts(synthetic_export, tmp_path_factory):
    """Train a tiny world model once per test session; reuse across tests.

    Returns (artifacts_dir, predictor, cfg).
    """
    from ml.src.inference.predictor import WorldModelPredictor
    from ml.src.training.train import save_artifacts
    from ml.src.trainer import Trainer

    cfg = make_tiny_config()
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    result = Trainer(cfg, artifacts_dir).train(synthetic_export["dir"])
    save_artifacts(result, cfg, artifacts_dir, synthetic_export["dir"],
                   dataset_info={"generator": "synthetic_linear_ar"})
    predictor = WorldModelPredictor.load(artifacts_dir, device="cpu")
    return artifacts_dir, predictor, cfg
