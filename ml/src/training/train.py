"""Artifact export + training CLI.

    python -m ml.src.training.train --dataset-dir <phase2_export> [--artifacts-dir ml/artifacts] [--config ml/configs/model.yaml]

Artifacts written:
    world_model.pt             weights + normalization + schema (self-contained)
    model_metadata.json        version, metrics, torch/device/timestamp, config
    feature_schema.json        copied from the dataset dir
    preprocessing_metadata.json copied from the dataset dir
    label_mapping.json         copied/renamed from label_mappings.json
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..models.world_model import build_model
from ..utils.config import WorldModelConfig
from ..utils.seeding import count_parameters, get_torch_version, set_seed
from .dataset import load_tensors
from .trainer import Trainer

MODEL_VERSION = "1.0.0"


def save_artifacts(
    result: dict,
    cfg: WorldModelConfig,
    artifacts_dir: str | Path,
    dataset_dir: str | Path,
    dataset_info: dict | None = None,
) -> dict[str, str]:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(result["input_dim"], cfg)
    model.load_state_dict({k: v for k, v in result["state_dict"].items()})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # self-contained bundle: everything inference needs in one file
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": cfg.model.d_model, "n_layers": cfg.model.n_layers,
                "n_heads": cfg.model.n_heads, "d_ff": cfg.model.d_ff,
                "dropout": cfg.model.dropout, "backbone": cfg.model.backbone,
                "sequence_length": cfg.sequence.length,
                "prediction_horizon": cfg.sequence.prediction_horizon,
            },
            "torch_version": get_torch_version(),
            "model_version": MODEL_VERSION,
        },
        out / "world_model.pt",
    )

    ds = Path(dataset_dir)
    for src, dst in [
        ("feature_schema.json", "feature_schema.json"),
        ("preprocessing_metadata.json", "preprocessing_metadata.json"),
        ("label_mappings.json", "label_mapping.json"),
    ]:
        p = ds / src
        if p.exists():
            shutil.copy2(p, out / dst)

    final_metrics = {}
    if result["history"]:
        last = result["history"][-1]
        final_metrics = {k: v for k, v in last.items()}
    metadata = {
        "model_version": MODEL_VERSION,
        "architecture": cfg.model.backbone,
        "feature_ordering": None,  # filled by caller when schema available
        "sequence_length": cfg.sequence.length,
        "prediction_horizon": cfg.sequence.prediction_horizon,
        "training_config": {
            "epochs": cfg.training.epochs, "batch_size": cfg.training.batch_size,
            "grad_accumulation": cfg.training.grad_accumulation,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay, "amp": cfg.training.amp,
            "seed": cfg.training.seed,
            "loss_weights": vars(cfg.loss_weights),
        },
        "model_config": {
            "d_model": cfg.model.d_model, "n_layers": cfg.model.n_layers,
            "n_heads": cfg.model.n_heads, "d_ff": cfg.model.d_ff,
            "dropout": cfg.model.dropout,
        },
        "parameters": count_parameters(model),
        "dataset": {"source_dir": str(ds), **(dataset_info or {})},
        "metrics": final_metrics,
        "split_sizes": result["split_sizes"],
        "best_val_loss": result["best_val_loss"],
        "test_loss": result["test_loss"],
        "pytorch_version": get_torch_version(),
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out / "model_metadata.json"

    # fill feature ordering from the copied schema if present
    fs = out / "feature_schema.json"
    if fs.exists():
        names = json.loads(fs.read_text(encoding="utf-8"))["feature_names"]
        metadata["feature_ordering"] = names
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    hist_path = out / "training_history.json"
    hist_path.write_text(json.dumps(result["history"], indent=2), encoding="utf-8")

    return {
        "world_model": str(out / "world_model.pt"),
        "model_metadata": str(meta_path),
        "training_history": str(hist_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the THREATCAST world model")
    parser.add_argument("--dataset-dir", required=True, help="Phase 2 export dir (tensors.npz)")
    parser.add_argument("--artifacts-dir", default="ml/artifacts")
    parser.add_argument("--config", default="ml/configs/model.yaml")
    args = parser.parse_args(argv)

    cfg = WorldModelConfig.load(args.config)
    tensors = load_tensors(args.dataset_dir)
    print(
        f"[train] X={tensors['X'].shape} labeled={int(tensors['mal_mask'].sum())}/{len(tensors['X'])} "
        f"device={cfg.resolve_device()} epochs={cfg.training.epochs}"
    )

    trainer = Trainer(cfg, Path(args.artifacts_dir).parent)  # checkpoints under <artifacts>/../checkpoints
    result = trainer.train(args.dataset_dir)
    print(f"[train] best_val={result['best_val_loss']:.4f} test={result['test_loss']:.4f} "
          f"splits={result['split_sizes']} time={result['train_seconds']}s")

    paths = save_artifacts(result, cfg, args.artifacts_dir, args.dataset_dir)
    print("[train] artifacts:", json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
