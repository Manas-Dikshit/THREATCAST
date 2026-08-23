"""Logistic Regression baseline on the same features.

Uses only the LAST observed state of each sequence (X[:, -1, :]) — a static
classifier that cannot model temporal dynamics. Comparing against it shows
whether the world model's temporal modeling actually helps.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import classification_metrics


class BaselineResult:
    def __init__(self, metrics: dict, model: LogisticRegression | None):
        self.metrics = metrics
        self.model = model


def evaluate_baseline(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      seed: int = 42) -> BaselineResult:
    """y: binary malicious targets. Returns degenerate (NaN) metrics when the
    test split lacks both classes — never fabricates numbers."""
    if len(np.unique(y_train)) < 2:
        return BaselineResult(
            {"accuracy": float("nan"), "note": "train split has a single class; baseline not trainable",
             "n_samples": int(len(y_test))}, None)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train[:, -1, :], y_train)
    preds = clf.predict(X_test[:, -1, :])
    return BaselineResult(classification_metrics(y_test, preds), clf)


__all__ = ["evaluate_baseline", "BaselineResult"]
