"""Trainer: chronological splits, AMP on CUDA, grad accumulation, checkpoints.

Checkpoint layout (ml/checkpoints/<run_name>/):
    latest.pt    {model, optimizer, scheduler, epoch, history, config}
    best.pt      lowest validation loss
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.world_model import build_model
from ..utils.config import WorldModelConfig
from ..utils.seeding import pick_device, set_seed
from .dataset import SequenceDataset, chronological_split, load_tensors
from .losses import world_model_loss


class Trainer:
    def __init__(self, cfg: WorldModelConfig, artifacts_dir: str | Path):
        self.cfg = cfg
        self.device = torch.device(pick_device(cfg.training.device))
        self.amp_enabled = bool(cfg.training.amp and self.device.type == "cuda")
        self.artifacts_dir = Path(artifacts_dir)

    def train(self, dataset_dir: str | Path) -> dict:
        set_seed(self.cfg.training.seed)
        tensors = load_tensors(dataset_dir)
        n = len(tensors["X"])
        tr_idx, va_idx, te_idx = chronological_split(
            n, self.cfg.training.val_fraction, self.cfg.training.test_fraction
        )

        model = build_model(tensors["X"].shape[-1], self.cfg).to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.cfg.training.epochs))
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        loader = DataLoader(
            SequenceDataset(tensors, tr_idx), batch_size=self.cfg.training.batch_size,
            shuffle=True, num_workers=self.cfg.training.num_workers,
        )
        val_loader = DataLoader(
            SequenceDataset(tensors, va_idx), batch_size=self.cfg.training.batch_size,
            shuffle=False,
        ) if va_idx else None

        history: list[dict] = []
        best_val = float("inf")
        ckpt_dir = self.artifacts_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        patience_left = self.cfg.training.early_stopping_patience
        t0 = time.time()

        for epoch in range(self.cfg.training.epochs):
            model.train()
            epoch_terms: dict[str, float] = {}
            batches = 0
            opt.zero_grad()
            for step, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(self.device.type, enabled=self.amp_enabled):
                    outputs = model(batch["X"])
                    loss, terms = world_model_loss(outputs, batch, vars(self.cfg.loss_weights))
                    loss = loss / self.cfg.training.grad_accumulation
                scaler.scale(loss).backward()
                if (step + 1) % self.cfg.training.grad_accumulation == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                for k, v in terms.items():
                    epoch_terms[k] = epoch_terms.get(k, 0.0) + v
                batches += 1
            sched.step()
            train_metrics = {k: v / max(batches, 1) for k, v in epoch_terms.items()}

            val_loss = float("nan")
            if val_loader is not None and len(val_loader.dataset) > 0:
                val_loss = self._evaluate_loss(model, val_loader)

            record = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()},
                      "val_loss": val_loss}
            history.append(record)

            is_best = np.isfinite(val_loss) and val_loss < best_val
            if is_best:
                best_val = val_loss
                patience_left = self.cfg.training.early_stopping_patience
            else:
                patience_left -= 1
            if (epoch + 1) % self.cfg.training.checkpoint_every == 0 or is_best or epoch == self.cfg.training.epochs - 1:
                self._save_checkpoint(ckpt_dir, model, opt, sched, epoch + 1, history, "latest.pt")
                if is_best:
                    self._save_checkpoint(ckpt_dir, model, opt, sched, epoch + 1, history, "best.pt")

            if patience_left <= 0:
                break

        # final evaluation on the untouched chronological test tail
        test_loader = DataLoader(
            SequenceDataset(tensors, te_idx), batch_size=self.cfg.training.batch_size
        ) if te_idx else None
        test_loss = (
            self._evaluate_loss(model, test_loader) if test_loader is not None and len(test_loader.dataset) else float("nan")
        )

        return {
            "history": history,
            "best_val_loss": best_val,
            "test_loss": test_loss,
            "split_sizes": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
            "train_seconds": round(time.time() - t0, 2),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_dim": tensors["X"].shape[-1],
        }

    @torch.no_grad()
    def _evaluate_loss(self, model, loader: DataLoader) -> float:
        model.eval()
        losses = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(batch["X"])
            loss, _ = world_model_loss(outputs, batch, vars(self.cfg.loss_weights))
            losses.append(float(loss))
        return float(np.mean(losses)) if losses else float("nan")

    def _save_checkpoint(self, ckpt_dir: Path, model, opt, sched, epoch, history, name: str) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch": epoch,
                "history": history,
                "config": {
                    "d_model": self.cfg.model.d_model, "n_layers": self.cfg.model.n_layers,
                    "n_heads": self.cfg.model.n_heads, "d_ff": self.cfg.model.d_ff,
                    "dropout": self.cfg.model.dropout, "backbone": self.cfg.model.backbone,
                    "sequence_length": self.cfg.sequence.length,
                    "prediction_horizon": self.cfg.sequence.prediction_horizon,
                },
            },
            ckpt_dir / name,
        )


__all__ = ["Trainer"]
