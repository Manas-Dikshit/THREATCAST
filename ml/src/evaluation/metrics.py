"""Evaluation metrics: classification + temporal next-state prediction quality."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1 and FPR from the confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "fpr": float("nan"), "confusion_matrix": None,
                "n_samples": int(len(y_true)), "degenerate": True}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def temporal_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     feature_names: list[str]) -> dict:
    """Future-state prediction quality: MSE/MAE overall + per-feature + R2."""
    err = y_pred - y_true
    mse = (err**2).mean(axis=0)
    mae = np.abs(err).mean(axis=0)
    var = y_true.var(axis=0)
    ss_res = (err**2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.full_like(mse, np.nan)

    per_feature = sorted(
        ({"feature": n, "mse": float(m), "mae": float(a)}
         for n, m, a in zip(feature_names, mse, mae)),
        key=lambda d: d["mse"], reverse=True,
    )
    return {
        "next_state_mse": float(mse.mean()),
        "next_state_mae": float(mae.mean()),
        "next_state_r2": float(np.nanmean(r2)),
        "worst_features_mse": per_feature[:5],
        "best_features_mse": list(reversed(per_feature[-5:])),
        "n_samples": int(len(y_true)),
    }


__all__ = ["classification_metrics", "temporal_metrics"]
