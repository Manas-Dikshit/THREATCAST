"""Inference: predict/rollout/predict_next_state, CPU, invalid input,
PredictionResult schema conformance, artifact loading."""

import json

import pytest
from pydantic import ValidationError

from ml.src.inference.predictor import WorldModelPredictor
from ml.tests.conftest import FEATURES, L, make_sequence


@pytest.fixture(scope="module")
def predictor(trained_artifacts):
    return trained_artifacts[1]


@pytest.fixture(scope="module")
def artifacts_dir(trained_artifacts):
    return trained_artifacts[0]


def benign_sequence(rng) -> object:
    x = rng.normal(0, 1, size=(L, len(FEATURES)))
    target = rng.normal(0, 1, size=len(FEATURES))
    labels = ["BENIGN"] * (L - 1) + ["DDoS"]
    return make_sequence(x, "seq_test", labels=labels, target_values=target)


class TestPredictionResult:
    def test_schema_conformance(self, predictor):
        seq = benign_sequence(pytest.importorskip("numpy").random.default_rng(3))
        result = predictor.predict(seq)
        assert result.prediction_id.startswith("pred_")
        for v in (result.risk_score, result.malicious_probability, result.confidence):
            assert 0.0 <= v <= 1.0
        assert result.model.name == "threatcast-world-model"
        # backend canonical schema round-trips
        dumped = json.loads(result.model_dump_json())
        assert set(dumped) == {
            "prediction_id", "timestamp", "risk_score", "malicious_probability",
            "confidence", "predicted_stage", "future_states",
            "feature_contributions", "model",
        }
        assert dumped["predicted_stage"]["source"] is None  # no fabricated stages

    def test_scores_are_floats_in_range(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(5))
        r = predictor.predict(seq)
        assert isinstance(r.malicious_probability, float)
        assert isinstance(r.confidence, float)

    def test_future_states_match_horizon(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(6))
        r = predictor.predict(seq)
        assert [e.step for e in r.future_states] == [1, 2, 3]
        for e in r.future_states:
            assert set(e.features.keys()) == set(FEATURES)
            assert 0.0 <= e.confidence <= 1.0

    def test_invalid_prediction_values_rejected_by_schema(self):
        from backend.app.schemas.prediction import PredictionResult

        with pytest.raises(ValidationError):
            PredictionResult(prediction_id="p", risk_score=1.5)


class TestRollout:
    def test_rollout_horizon_override(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(7))
        entries = predictor.rollout(seq, horizon=5)
        assert len(entries) == 5 and entries[-1].step == 5

    def test_timestamps_advance_by_window(self, predictor):
        import numpy as np
        from datetime import timedelta

        seq = benign_sequence(np.random.default_rng(8))
        entries = predictor.rollout(seq, horizon=3)
        last_end = seq.states[-1].timestamp_end
        for k, e in enumerate(entries, start=1):
            assert e.timestamp == last_end + timedelta(seconds=10 * k)

    def test_autoregressive_feedback_consistency(self, predictor):
        """Rollout step 2 must equal a direct prediction from the SLID window
        [S(t-2)..S(t), S(t+1)] — proving predictions are actually fed back."""
        import copy
        from datetime import timedelta

        import numpy as np

        seq = benign_sequence(np.random.default_rng(9))
        entries = predictor.rollout(seq, horizon=2)

        # rebuild the slid window manually: drop oldest, append predicted state
        shifted = copy.deepcopy(seq.states[-1])
        shifted.state_id = f"{seq.sequence_id}_sim_1"
        shifted.timestamp_start = seq.states[-1].timestamp_end
        shifted.timestamp_end = shifted.timestamp_start + timedelta(
            seconds=seq.window_seconds or 10
        )
        shifted.features = dict(entries[0].features)

        from data_pipeline.src.schemas.network_state import NetworkStateSequence

        seq2 = NetworkStateSequence(
            sequence_id=f"{seq.sequence_id}_slid",
            states=seq.states[1:] + [shifted],
            sequence_length=L,
            window_seconds=seq.window_seconds,
        )
        step2_direct = predictor.predict_next_state(seq2)

        # Tolerance covers float32 math plus the 6-decimal denorm->renorm
        # roundtrip; a non-feedback implementation would differ by O(1).
        for name in FEATURES:
            assert abs(step2_direct[name] - entries[1].features[name]) < 0.05

    def test_invalid_horizon(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(10))
        with pytest.raises(ValueError, match="horizon"):
            predictor.rollout(seq, horizon=0)


class TestPredictNextState:
    def test_returns_named_features(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(11))
        nxt = predictor.predict_next_state(seq)
        assert set(nxt) == set(FEATURES)
        assert all(isinstance(v, float) for v in nxt.values())


class TestInvalidSequences:
    @pytest.mark.parametrize("bad_seq", ["none", "empty", "wrong_length"])
    def test_invalid_inputs_raise(self, predictor, bad_seq):
        import numpy as np

        if bad_seq == "none":
            seq = None
            match = "None"
        elif bad_seq == "empty":
            seq = make_sequence(np.zeros((0, len(FEATURES))), "seq_empty")
            match = "no states"
        else:
            seq = make_sequence(np.zeros((7, len(FEATURES))), "seq_long")
            match = "INVALID_SEQUENCE"
        with pytest.raises(ValueError, match=match):
            predictor.predict_next_state(seq)


class TestCpuAndArtifacts:
    def test_cpu_inference_explicit(self, trained_artifacts):
        artifacts_dir, _, _ = trained_artifacts
        p = WorldModelPredictor.load(artifacts_dir, device="cpu")
        import numpy as np

        seq = benign_sequence(np.random.default_rng(12))
        r = p.predict(seq)
        assert 0.0 <= r.risk_score <= 1.0
        assert str(p.device) == "cpu"

    def test_missing_artifacts_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            WorldModelPredictor.load(tmp_path)

    def test_artifacts_complete(self, artifacts_dir):
        d = artifacts_dir
        required = [
            "world_model.pt", "model_metadata.json", "feature_schema.json",
            "preprocessing_metadata.json", "label_mapping.json",
        ]
        for name in required:
            assert (d / name).exists(), f"missing artifact {name}"

    def test_metadata_contract_fields(self, artifacts_dir):
        meta = json.loads((artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
        for field in (
            "model_version", "feature_ordering", "sequence_length",
            "prediction_horizon", "training_config", "dataset", "metrics",
            "pytorch_version", "device", "timestamp_utc",
        ):
            assert field in meta, f"metadata missing {field}"
        assert meta["feature_ordering"] == FEATURES

    def test_checkpoint_reload_consistency(self, trained_artifacts):
        """Same sequence through two independently loaded predictors -> identical output."""
        import numpy as np

        artifacts_dir, _, _ = trained_artifacts
        p1 = WorldModelPredictor.load(artifacts_dir, device="cpu")
        p2 = WorldModelPredictor.load(artifacts_dir, device="cpu")
        seq = benign_sequence(np.random.default_rng(13))
        r1, r2 = p1.predict(seq), p2.predict(seq)
        assert abs(r1.risk_score - r2.risk_score) < 1e-6
        assert abs(r1.future_states[0].features["flow_count"]
                   - r2.future_states[0].features["flow_count"]) < 1e-4


class TestExplainabilityHooks:
    def test_attention_extraction(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(14))
        maps = predictor.get_attentions(seq)
        assert len(maps) >= 1
        for m in maps:
            assert m.shape == (L, L)
            assert np.allclose(m.sum(axis=-1), 1.0, atol=1e-3)

    def test_feature_attribution_ranking(self, predictor):
        import numpy as np

        seq = benign_sequence(np.random.default_rng(15))
        contribs = predictor.attribute_features(seq)
        names = [n for n, _ in contribs]
        assert sorted(names) == sorted(FEATURES)
        scores = [s for _, s in contribs]
        assert scores[0] >= scores[-1]  # sorted descending
