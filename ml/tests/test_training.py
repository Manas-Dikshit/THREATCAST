"""Training loop: splits, checkpointing, artifact loading, leakage prevention."""

import json

import numpy as np
import torch

from ml.src.training.dataset import chronological_split, load_tensors
from ml.src.training.trainer import Trainer

from ml.tests.conftest import make_tiny_config


class TestSplits:
    def test_chronological_order_no_overlap(self):
        tr, va, te = chronological_split(100, 0.15, 0.15)
        assert tr == list(range(70))
        assert va == list(range(70, 85))
        assert te == list(range(85, 100))
        assert set(tr) & set(va) == set() and set(va) & set(te) == set()

    def test_test_split_is_temporal_tail(self):
        _, _, te = chronological_split(50, 0.15, 0.15)
        assert te[0] > 40, "test data must be the most recent sequences"


class TestTensorLoading:
    def test_load_and_derive_targets(self, synthetic_export):
        t = load_tensors(synthetic_export["dir"])
        assert t["X"].shape[1:] == (5, 9)
        assert t["target_mask"].all()
        assert t["mal_target"].sum() == t["mal_target"].size / 2  # half attack-labeled
        assert t["mal_mask"].all()

    def test_unlabeled_excluded_from_mal_mask(self, synthetic_export, tmp_path):
        import shutil

        d = tmp_path / "partial"
        shutil.copytree(synthetic_export["dir"], d)
        npz = dict(np.load(d / "tensors.npz"))
        ids = npz["label_ids"].copy()
        ids[:4] = -1
        np.savez_compressed(d / "tensors.npz", **{**npz, "label_ids": ids})
        t = load_tensors(d)
        assert not t["mal_mask"][:4].any() and t["mal_mask"][4:].all()


class TestTrainer:
    def test_training_reduces_loss_on_learnable_dynamics(self, synthetic_export, tmp_path):
        cfg = make_tiny_config()
        cfg.training.epochs = 8
        result = Trainer(cfg, tmp_path).train(synthetic_export["dir"])

        first = result["history"][0]["train_next_state"]
        last = result["history"][-1]["train_next_state"]
        assert last < first * 0.9, (
            f"next-state loss should fall on the learnable AR process ({first} -> {last})"
        )
        assert result["split_sizes"]["test"] >= 1
        assert result["best_val_loss"] == result["best_val_loss"]  # finite (no NaN)

    def test_checkpoints_written(self, synthetic_export, tmp_path):
        cfg = make_tiny_config()
        Trainer(cfg, tmp_path).train(synthetic_export["dir"])
        ckpt = tmp_path / "checkpoints" / "latest.pt"
        best = tmp_path / "checkpoints" / "best.pt"
        assert ckpt.exists() and best.exists()
        blob = torch.load(best, map_location="cpu", weights_only=False)
        assert "model_state_dict" in blob and "config" in blob
        assert blob["epoch"] >= 1

    def test_amp_disabled_on_cpu(self, synthetic_export, tmp_path):
        cfg = make_tiny_config()
        trainer = Trainer(cfg, tmp_path)
        assert not trainer.amp_enabled or trainer.device.type == "cuda"

    def test_normalization_not_refitted_on_test(self, synthetic_export):
        """The preprocessing metadata stats come from the Phase 2 export and are
        never touched by training — guard that the loader doesn't mutate them."""
        before = json.loads(
            (synthetic_export["dir"] / "preprocessing_metadata.json").read_text()
        )["mean"]
        load_tensors(synthetic_export["dir"])
        after = json.loads(
            (synthetic_export["dir"] / "preprocessing_metadata.json").read_text()
        )["mean"]
        assert before == after
