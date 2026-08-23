"""Configuration for the THREATCAST world model.

Loaded from ml/configs/model.yaml; environment variables take precedence
(MODEL_*, TRAIN_* prefixes). Kept as a plain dataclass — no framework.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    backbone: str = "temporal_transformer"  # temporal_transformer | lstm


@dataclass
class SequenceConfig:
    length: int = 5
    prediction_horizon: int = 3


@dataclass
class LossWeights:
    next_state: float = 1.0
    malicious: float = 1.0
    risk: float = 0.5
    confidence: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    grad_accumulation: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True                 # mixed precision (CUDA only)
    device: str = "auto"             # auto | cuda | cpu
    seed: int = 42
    checkpoint_every: int = 5        # epochs between checkpoint saves
    early_stopping_patience: int = 8
    val_fraction: float = 0.15       # of the train+val portion
    test_fraction: float = 0.15      # chronological tail
    num_workers: int = 0             # windows-safe default


@dataclass
class WorldModelConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)

    def resolve_device(self) -> str:
        if self.training.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.training.device

    @classmethod
    def load(cls, path: str | Path | None = None) -> "WorldModelConfig":
        """Load YAML config; env vars override (MODEL_LAYERS, SEQ_LENGTH,
        PREDICTION_HORIZON, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR,
        TRAIN_DEVICE, ...)."""
        cfg = cls()
        p = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "model.yaml"
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg._apply_dict(raw)
        env_map = {
            ("model", "layers"): _env("MODEL_LAYERS", ""),
            ("model", "d_model"): _env("MODEL_D_MODEL", ""),
            ("model", "heads"): _env("MODEL_HEADS", ""),
            ("model", "dropout"): _env("MODEL_DROPOUT", ""),
            ("sequence", "length"): _env("ML_SEQUENCE_LENGTH", ""),
            ("sequence", "prediction_horizon"): _env("ML_PREDICTION_HORIZON", ""),
            ("training", "epochs"): _env("TRAIN_EPOCHS", ""),
            ("training", "batch_size"): _env("TRAIN_BATCH_SIZE", ""),
            ("training", "learning_rate"): _env("TRAIN_LR", ""),
            ("training", "device"): _env("TRAIN_DEVICE", ""),
            ("training", "seed"): _env("TRAIN_SEED", ""),
            ("training", "amp"): _env("TRAIN_AMP", ""),
        }
        overrides = {sect: {k: v} for (sect, k), v in env_map.items() if v != ""}
        cfg._apply_dict(overrides)
        return cfg

    def _apply_dict(self, raw: dict) -> None:
        sections = {
            "model": (self.model, ModelConfig),
            "sequence": (self.sequence, SequenceConfig),
            "training": (self.training, TrainingConfig),
            "loss_weights": (self.loss_weights, LossWeights),
        }
        for name, section in raw.items():
            if not isinstance(section, dict):
                continue
            target, dc = sections[name]
            known = {f for f in dc.__dataclass_fields__}
            for key, val in section.items():
                key = key.replace("-", "_")
                if key in known and val is not None:
                    cur = getattr(target, key)
                    setattr(target, key, type(cur)(val) if cur is not None and not isinstance(cur, bool) or isinstance(val, bool) else val)


__all__ = ["WorldModelConfig", "ModelConfig", "SequenceConfig", "TrainingConfig", "LossWeights"]
