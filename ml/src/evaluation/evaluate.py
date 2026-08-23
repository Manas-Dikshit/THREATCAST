"""Full evaluation: world model vs Logistic Regression baseline.

    python -m ml.src.evaluation.evaluate --dataset-dir <phase2_export> --artifacts-dir ml/artifacts

Writes evaluation_report.json next to the artifacts. The report states plainly
when a comparison is impossible (e.g. single-class test split) instead of
claiming superiority without evidence.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..inference.predictor import WorldModelPredictor
from ..training.dataset import chronological_split, load_tensors
from ..utils.seeding import set_seed
from .baseline import evaluate_baseline
from .metrics import classification_metrics, temporal_metrics


def evaluate_world_model(predictor: WorldModelPredictor,
                         tensors: dict, test_idx: list[int],
                         horizon: int = 1) -> dict:
    """Temporal quality = 1-step-ahead prediction on the test tail.
    Also collects malicious-head classification on labeled rows."""
    X = tensors["X"][test_idx]
    Y = tensors["Y"][test_idx]
    mask = tensors["target_mask"][test_idx]

    names = predictor.vectorizer.feature_names
    temporal = {"note": "no sequences with target states in the test split", "n_samples": int(mask.sum())}
    if mask.sum() >= 2:
        with torch.no_grad():
            xb = torch.from_numpy(X[mask]).to(predictor.device)
            preds = []
            for i in range(0, len(xb), 256):
                out = predictor.model(xb[i:i + 256])
                preds.append(out["next_state_pred"].float().cpu().numpy())
            y_pred = np.concatenate(preds)
        temporal = temporal_metrics(Y[mask], y_pred, names)

    mal_mask = tensors["mal_mask"][test_idx]
    if mal_mask.sum() >= 2 and np.unique(tensors["mal_target"][test_idx][mal_mask]).size >= 2:
        with torch.no_grad():
            xb = torch.from_numpy(X[mal_mask]).to(predictor.device)
            probs = []
            for i in range(0, len(xb), 256):
                out = predictor.model(xb[i:i + 256])
                probs.append(torch.sigmoid(out["malicious_logit"]).float().cpu().numpy())
            cls = classification_metrics(
                tensors["mal_target"][test_idx][mal_mask],
                (np.concatenate(probs) >= 0.5).astype(int),
            )
    else:
        cls = {"note": "test split lacks both classes; classification metrics not computed",
               "degenerate": True}

    return {"temporal_next_state": temporal, "malicious_classification": cls}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate world model vs LR baseline")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--artifacts-dir", default="ml/artifacts")
    parser.add_argument("--config", default="ml/configs/model.yaml")
    args = parser.parse_args(argv)

    from ..utils.config import WorldModelConfig
    cfg = WorldModelConfig.load(args.config)
    set_seed(cfg.training.seed)

    tensors = load_tensors(args.dataset_dir)
    n = len(tensors["X"])
    _, _, te_idx = chronological_split(n, cfg.training.val_fraction, cfg.training.test_fraction)
    if not te_idx:
        te_idx = [n - 1]

    report: dict = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "n_sequences": n, "test_size": len(te_idx)}

    # ---- baseline (same features, same split) ----
    baseline = evaluate_baseline(
        tensors["X"][[i for i in range(n) if i not in set(te_idx)]],
        tensors["mal_target"][[i for i in range(n) if i not in set(te_idx)]],
        tensors["X"][te_idx], tensors["mal_target"][te_idx], seed=cfg.training.seed,
    )
    report["logistic_regression_baseline"] = baseline.metrics

    # ---- world model ----
    try:
        predictor = WorldModelPredictor.load(args.artifacts_dir, device="cpu")
        report["world_model"] = evaluate_world_model(predictor, tensors, te_idx)
    except FileNotFoundError as exc:
        report["world_model"] = {"error": str(exc), "note": "train the model first"}

    out = Path(args.artifacts_dir) / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[evaluate] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
