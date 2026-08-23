# Phase 1 smoke tests: validate the foundation only.
# These assert structure, contracts and configuration — NOT ML/backend functionality.

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DIRS = [
    "backend/app/api", "backend/app/core", "backend/app/models", "backend/app/schemas",
    "backend/app/services", "backend/app/repositories", "backend/tests",
    "frontend/src/components", "frontend/src/pages", "frontend/src/services",
    "frontend/src/hooks", "frontend/src/types", "frontend/src/utils", "frontend/tests",
    "ml/src/data", "ml/src/features", "ml/src/states", "ml/src/models",
    "ml/src/training", "ml/src/inference", "ml/src/evaluation", "ml/src/utils",
    "ml/notebooks", "ml/configs", "ml/artifacts", "ml/checkpoints", "ml/tests",
    "data_pipeline/src/ingestion", "data_pipeline/src/parsers", "data_pipeline/src/features",
    "data_pipeline/src/windows", "data_pipeline/src/preprocessing", "data_pipeline/src/schemas",
    "integration/tests", "integration/fixtures", "integration/scripts",
    "security/attack_mapping", "security/explainability",
    "docs", "architectures", "tests/smoke", "docker",
    "scripts/setup", "scripts/development", "scripts/validation",
]


def test_expected_folder_structure():
    missing = [d for d in EXPECTED_DIRS if not (ROOT / d).is_dir()]
    assert not missing, f"missing directories: {missing}"


def test_required_root_files_exist():
    required = [
        "README.md", "ARCHITECTURE.md", "CONTRACT.md", ".env.example", ".gitignore",
        "requirements.txt", "docker-compose.yml", "pytest.ini",
        "docker/backend.Dockerfile", "docker/frontend.Dockerfile", "docker/ml.Dockerfile",
        "docs/API.md", "docs/DATA_SCHEMA.md", "docs/MODEL.md", "docs/MITRE_MAPPING.md",
        "docs/DEPLOYMENT.md", "docs/DEVELOPMENT.md", "docs/TROUBLESHOOTING.md",
        "architectures/ARCHITECTURE.md",
        "ml/README.md", "ml/MODEL.md", "ml/TRAINING.md",
        "data_pipeline/README.md", "backend/README.md", "frontend/README.md",
        "integration/README.md", "security/README.md",
    ]
    missing = [f for f in required if not (ROOT / f).is_file()]
    assert not missing, f"missing files: {missing}"


def test_env_contract_variables():
    env = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()

    expected = {
        "APP_ENV": "development",
        "BACKEND_HOST": "0.0.0.0",
        "BACKEND_PORT": "8000",
        "FRONTEND_PORT": "5173",
        "DATABASE_URL": "postgresql://threatcast:threatcast@localhost:5432/threatcast",
        "ML_MODEL_PATH": "./ml/artifacts/world_model.pt",
        "ML_METADATA_PATH": "./ml/artifacts/model_metadata.json",
        "ML_DEVICE": "auto",
        "ML_SEQUENCE_LENGTH": "5",
        "ML_PREDICTION_HORIZON": "3",
        "TIME_WINDOW_SECONDS": "10",
        "MAX_UPLOAD_SIZE_MB": "500",
        "LOG_LEVEL": "INFO",
    }
    for key, value in expected.items():
        assert env.get(key) == value, f"{key} must equal {value}, got {env.get(key)}"


def test_port_contract_documented_consistently():
    """Ports 8000 / 5173 / 5432 and prefix /api/v1 must appear in the contract docs."""
    contract = (ROOT / "CONTRACT.md").read_text(encoding="utf-8")
    for token in ["**8000**", "**5173**", "**5432**", "/api/v1"]:
        assert token in contract, f"CONTRACT.md missing {token}"
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for port in ['"8000:8000"', '"5173:5173"', '"5432:5432"']:
        assert port in compose, f"docker-compose.yml missing port mapping {port}"
    vite = (ROOT / "frontend/vite.config.ts").read_text(encoding="utf-8")
    assert "5173" in vite and "http://localhost:8000" in vite


def test_no_secrets_committed():
    """.env must not exist in git; example must contain no obvious secret patterns."""
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    assert ".env" not in tracked
    assert "backend/.env" not in tracked

    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    # local dev placeholder DSN is fine; real-looking API keys are not
    for bad in ("sk-", "AKIA", "BEGIN PRIVATE KEY"):
        assert bad not in example, f"possible secret pattern in .env.example: {bad}"


def test_canonical_schemas_import_and_validate():
    from app.schemas import (
        NetworkState,
        NetworkStateSequence,
        PredictionResult,
        ErrorEnvelope,
    )
    from datetime import datetime, timezone

    state = NetworkState(
        state_id="state_000001",
        timestamp_start=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        timestamp_end=datetime(2026, 1, 1, 10, 0, 10, tzinfo=timezone.utc),
        window_seconds=10.0,
        features={"flow_count": 120},
        label=None,
        label_source=None,
    )
    seq = NetworkStateSequence(sequence_id="seq_000001", states=[state])
    assert seq.sequence_length == 5 and seq.window_seconds == 10.0

    now = datetime.now(timezone.utc)
    res = PredictionResult(
        prediction_id="pred_123", timestamp=now,
        risk_score=0.87, malicious_probability=0.91, confidence=0.84,
    )
    payload = json.loads(res.model_dump_json())
    assert payload["model"]["name"] == "threatcast-world-model"
    ErrorEnvelope.model_validate({
        "error": {"code": "INVALID_INPUT", "message": "x", "details": {}, "request_id": "req_1"}
    })


def test_settings_load_from_env_contract():
    from app.core.config import Settings
    s = Settings(_env_file=str(ROOT / ".env.example"))
    assert s.backend_port == 8000
    assert s.ml_sequence_length == 5
    assert s.ml_prediction_horizon == 3
    assert s.time_window_seconds == 10.0
