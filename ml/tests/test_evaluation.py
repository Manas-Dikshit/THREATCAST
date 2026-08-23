"""Evaluation metrics + logistic regression baseline."""

import numpy as np
import pytest

from ml.src.evaluation.baseline import evaluate_baseline
from ml.src.evaluation.metrics import classification_metrics, temporal_metrics

from ml.tests.conftest import FEATURES


class TestClassificationMetrics:
    def test_hand_computed(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 1])
        m = classification_metrics(y_true, y_pred)
        # cm = [[2,1],[1,3]] -> tn=2 fp=1 fn=1 tp=3
        assert m["accuracy"] == pytest.approx(5 / 7)
        assert m["precision"] == pytest.approx(3 / 4)
        assert m["recall"] == pytest.approx(3 / 4)
        assert m["fpr"] == pytest.approx(1 / 3)
        assert m["confusion_matrix"] == [[2, 1], [1, 3]]

    def test_single_class_degenerate(self):
        m = classification_metrics(np.zeros(5, dtype=int), np.zeros(5, dtype=int))
        assert m["degenerate"] and np.isnan(m["accuracy"])


class TestTemporalMetrics:
    def test_perfect_prediction_r2_one(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=(50, len(FEATURES)))
        t = temporal_metrics(y, y.copy(), FEATURES)
        assert t["next_state_mse"] == 0.0
        assert t["next_state_r2"] == pytest.approx(1.0)

    def test_known_error(self):
        y = np.ones((10, len(FEATURES)))
        pred = np.full((10, len(FEATURES)), 2.0)
        t = temporal_metrics(y, pred, FEATURES)
        assert t["next_state_mse"] == pytest.approx(1.0)
        assert t["next_state_mae"] == pytest.approx(1.0)

    def test_worst_features_sorted_descending(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(100, len(FEATURES)))
        pred = y + rng.normal(scale=np.linspace(0.1, 2.0, len(FEATURES)))
        t = temporal_metrics(y, pred, FEATURES)
        mses = [d["mse"] for d in t["worst_features_mse"]]
        assert mses == sorted(mses, reverse=True)


class TestBaseline:
    def test_baseline_separable_data(self):
        rng = np.random.default_rng(2)
        n, l, f = 200, 5, len(FEATURES)
        X_benign = rng.normal(0, 1, (n // 2, l, f))
        X_attack = rng.normal(0, 1, (n // 2, l, f)) * 1.0
        X_attack[:, -1, :] += 6.0  # last state strongly different -> separable
        X_train = np.concatenate([X_benign[:60], X_attack[:60]])
        y_train = np.array([0] * 60 + [1] * 60, dtype=float)
        X_test = np.concatenate([X_benign[60:], X_attack[60:]])
        y_test = np.array([0] * (n // 2 - 60) + [1] * (n // 2 - 60), dtype=float)

        res = evaluate_baseline(X_train, y_train, X_test, y_test)
        assert res.model is not None
        assert res.metrics["f1"] > 0.9

    def test_degenerate_single_class_train(self):
        X = np.random.default_rng(3).normal(size=(20, 5, len(FEATURES)))
        res = evaluate_baseline(X, np.zeros(20), X, np.zeros(20))
        assert res.model is None
        assert "single class" in res.metrics.get("note", "")
