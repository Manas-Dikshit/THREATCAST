"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_version() -> str:
    return torch.__version__


def pick_device(preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    try:
        available = torch.cuda.is_available()
    except Exception:  # pragma: no cover - defensive on odd builds
        available = False
    if preference == "cuda":
        if not available:
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if available else "cpu"


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


os.environ.setdefault("PYTHONHASHSEED", str(42))

__all__ = ["set_seed", "get_torch_version", "pick_device", "count_parameters"]
