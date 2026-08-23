"""Backend test fixtures: SQLite DB, tiny REAL world-model artifact, client.

The model artifact is trained once per session on synthetic linear-AR data
(same generator as ml/tests) so the full upload->pipeline->inference->DB chain
runs with genuine inference — no CIC-IDS2018 required.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

FEATURES = [
    "flow_count", "packet_count", "byte_count", "syn_ratio", "ack_ratio",
    "mean_iat", "iat_variance", "unique_dst_ports", "port_scan_score",
]
L = 5


def _build_export(d: Path) -> None:
    rng = np.random.default_rng(7)
    F = len(FEATURES)
    A = rng.normal(0, 0.15, size=(F, F)) + np.eye(F) * 0.5
    X, Y = [], []
    for i in range(64):
        x = rng.normal(0, 1, size=(L, F)).astype(np.float32)
        for t in range(L):
            nxt = (A @ x[t] + rng.normal(0, 0.05, F)).astype(np.float32)
            if t < L - 1:
                x[t + 1] = nxt
        X.append(x)
        Y.append((A @ x[-1]).astype(np.float32))
    X, Y = np.stack(X), np.stack(Y)
    np.savez_compressed(
        d / "tensors.npz", X=X.astype(np.float32), Y=Y.astype(np.float32),
        target_mask=np.ones(len(X), bool),
        label_ids=np.array([0 if i % 2 == 0 else 1 for i in range(len(X))], np.int64),
    )
    mean = {n: float(v) for n, v in zip(FEATURES, X.reshape(-1, F).mean(0))}
    std = {n: max(float(v), 1e-6) for n, v in zip(FEATURES, X.reshape(-1, F).std(0))}
    (d / "feature_schema.json").write_text(json.dumps({"feature_names": FEATURES}))
    (d / "preprocessing_metadata.json").write_text(json.dumps({
        "preprocessing_version": "1", "feature_names": FEATURES,
        "mean": mean, "std": std, "fit_start": "2026-01-01T00:00:00Z",
        "fit_end": "2026-01-01T01:00:00Z",
        "normalization": "zscore_train_only", "source_dataset": "synthetic"}))
    (d / "label_mappings.json").write_text(json.dumps(
        {"labels": {"BENIGN": 0, "DDoS": 1}, "unlabeled_id": -1}))


def _train_tiny(export_dir: Path, out: Path) -> None:
    from ml.src.training.trainer import Trainer
    from ml.src.training.train import save_artifacts
    from ml.src.utils.config import WorldModelConfig

    cfg = WorldModelConfig()
    cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads, cfg.model.d_ff = 32, 2, 4, 64
    cfg.sequence.length = L
    cfg.training.epochs, cfg.training.batch_size = 3, 16
    cfg.training.device, cfg.training.amp = "cpu", False
    result = Trainer(cfg, out / "_ckpts").train(str(export_dir))
    save_artifacts(result, cfg, out, str(export_dir))


@pytest.fixture(scope="session")
def model_artifacts(tmp_path_factory):
    """Tiny but real Phase 3 artifact bundle."""
    root = tmp_path_factory.mktemp("ml_artifacts")
    export_dir = root / "export"
    export_dir.mkdir()
    _build_export(export_dir)
    _train_tiny(export_dir, root / "artifacts")
    return root / "artifacts"


@pytest.fixture(scope="session")
def engine(model_artifacts, tmp_path_factory):
    from app.db.session import make_engine
    from app.db.base import Base
    from app.models import tables  # noqa: F401

    path = tmp_path_factory.mktemp("db") / "test.db"
    eng = make_engine(f"sqlite:///{path}")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    """Fresh session per test with clean tables."""
    from app.db.base import Base
    from app.models import tables  # noqa: F401

    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    yield s
    s.close()


@pytest.fixture()
def client(model_artifacts, engine, db_session, monkeypatch):
    """TestClient with SQLite DB override + real tiny model loaded."""
    os.environ["ML_ARTIFACTS_DIR"] = str(model_artifacts)
    monkeypatch.setenv("ML_ARTIFACTS_DIR", str(model_artifacts))

    from app.core.config import get_settings
    get_settings.cache_clear()

    from sqlalchemy.orm import sessionmaker

    def override_get_db():
        yield db_session

    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_db_module()] = override_get_db
    with TestClient(app) as c:
        c.headers.update({"X-Test-Client": "1"})
        yield c
    get_settings.cache_clear()


def get_db_module():
    import app.db.session as m

    return m.get_db


CSV_HEADER = ("timestamp,src_ip,dst_ip,src_port,dst_port,protocol,"
              "total_bytes,total_packets,duration_s\n")


def make_csv(n_flows: int = 40, start_epoch: int = 0) -> bytes:
    rng = np.random.default_rng(11)
    rows = [CSV_HEADER]
    base_ts = 1_760_000_000 + start_epoch
    for i in range(n_flows):
        ts = base_ts + int(i // 4) * 10 + (i % 4)
        rows.append(
            f"{ts},10.0.0.{i % 5},10.0.1.{i % 3},{1000 + i},{80 if i % 3 else 443},"
            f"{'tcp'},{rng.integers(100, 5000)},{rng.integers(5, 50)},0.05\n"
        )
    return "".join(rows).encode()


@pytest.fixture()
def csv_bytes() -> bytes:
    return make_csv()


@pytest.fixture()
def pcap_bytes() -> bytes:
    """Minimal valid PCAP global header (magic only is enough to pass sniffing;
    pipeline-level failure is exercised separately)."""
    return b"\xd4\xc3\xb2\xa1" + b"\x00" * 20
