"""Dataset loading for world-model training.

Consumes the Phase 2 export directory (tensors.npz + label_mappings.json).
Splitting is chronological over the sequence order (sequences are emitted in
time order by the Phase 2 window engine): train | val | test with no shuffling
across boundaries. Normalization was already fitted on train data by the data
pipeline — the test tail never influences it.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_tensors(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    d = Path(dataset_dir)
    npz = np.load(d / "tensors.npz")
    mapping = json.loads((d / "label_mappings.json").read_text(encoding="utf-8"))
    labels: dict[str, int] = mapping["labels"]
    benign_ids = {lid for name, lid in labels.items()
                  if any(w in name.lower() for w in ("benign", "normal", "background"))}
    label_ids = npz["label_ids"]

    mal_target = np.ones_like(label_ids)
    mal_target[np.isin(label_ids, list(benign_ids))] = 0
    mal_mask = label_ids >= 0  # unlabeled sequences contribute no classification signal

    return {
        "X": npz["X"].astype(np.float32),
        "Y": npz["Y"].astype(np.float32),
        "target_mask": npz["target_mask"],
        "mal_target": mal_target.astype(np.float32),
        "mal_mask": mal_mask,
    }


def chronological_split(n: int, val_fraction: float, test_fraction: float):
    train_end = max(1, int(n * (1.0 - val_fraction - test_fraction)))
    val_end = min(n - 1, max(train_end + 1, int(n * (1.0 - test_fraction)))) if test_fraction > 0 else n
    return list(range(0, train_end)), list(range(train_end, val_end)), list(range(val_end, n))


class SequenceDataset(Dataset):
    """Rows of the Phase 2 tensors, indexed by a precomputed split."""

    def __init__(self, tensors: dict[str, np.ndarray], indices: list[int]):
        self.X = torch.from_numpy(tensors["X"][indices])
        self.Y = torch.from_numpy(tensors["Y"][indices])
        self.target_mask = torch.from_numpy(tensors["target_mask"][indices])
        self.mal_target = torch.from_numpy(tensors["mal_target"][indices])
        self.mal_mask = torch.from_numpy(tensors["mal_mask"][indices])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "X": self.X[i],
            "Y": self.Y[i],
            "target_mask": self.target_mask[i],
            "mal_target": self.mal_target[i],
            "mal_mask": self.mal_mask[i],
        }


__all__ = ["load_tensors", "chronological_split", "SequenceDataset"]
