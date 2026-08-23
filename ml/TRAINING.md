# Training Procedure

## Command

```powershell
python -m ml.src.training.train --dataset-dir <phase2_export> --artifacts-dir ml\artifacts [--config ml\configs\model.yaml]
python -m ml.src.evaluation.evaluate --dataset-dir <phase2_export> --artifacts-dir ml\artifacts
```

Config: `ml/configs/model.yaml`; env overrides: `MODEL_LAYERS, MODEL_D_MODEL, MODEL_HEADS,
MODEL_DROPOUT, ML_SEQUENCE_LENGTH, ML_PREDICTION_HORIZON, TRAIN_EPOCHS, TRAIN_BATCH_SIZE,
TRAIN_LR, TRAIN_DEVICE (auto|cuda|cpu), TRAIN_SEED, TRAIN_AMP`.

## Data

Consumes the Phase 2 export directory: `tensors.npz` (`X[N,L,F]`, `Y[N,F]`,
`target_mask[N]`, `label_ids[N]`) + `label_mappings.json`. Binary malicious
targets are derived at load time (`ml/src/training/dataset.py`): labels whose
name contains benign/normal/background → 0; other verified labels → 1;
unlabeled (-1) → excluded from classification losses via `mal_mask`.

## Splits — leakage prevention

Chronological over sequence order (sequences are emitted in time order):

| split | share | position | use |
|-------|-------|----------|-----|
| train | 70 % | head | fit weights |
| val   | 15 % | middle | early stopping / best checkpoint |
| test  | 15 % | **tail** | final evaluation only |

- Normalization stats come from the Phase 2 pipeline, fitted on train data **only**; the ML layer never refits them.
- No shuffling across the temporal boundary; test = most recent sequences.
- `chronological_split` is unit-tested for ordering and disjointness.

## Loop

- AdamW (lr 1e-3, wd 1e-4) + cosine annealing.
- Mixed precision via `torch.autocast("cuda")` + GradScaler when `amp: true`
  **and** device is CUDA; auto-disabled on CPU (Windows-safe).
- Gradient accumulation supported (`grad_accumulation`) for small VRAM.
- Early stopping patience 8 epochs on validation loss.
- Seeded (`TRAIN_SEED`, default 42) via `set_seed`.

## Checkpoints

`ml/checkpoints/` per run dir:
- `latest.pt` refreshed every `checkpoint_every` epochs and on improvement
- `best.pt` lowest validation loss

Blob contents: model state_dict, optimizer/scheduler state, epoch, history,
model config. Reload with `torch.load(..., weights_only=False)`.

## Artifacts (after training)

`ml/artifacts/`: `world_model.pt` (self-contained bundle: weights + config +
version), `model_metadata.json`, `training_history.json`, plus copies of the
dataset's `feature_schema.json`, `preprocessing_metadata.json`,
`label_mapping.json`. `model_metadata.json` records: model version, feature
ordering, sequence length, prediction horizon, full training + model configs,
parameter counts, dataset info, final metrics, split sizes, best/test loss,
PyTorch version, device, UTC timestamp.

## Evaluation

- World model temporal quality on the test tail: next-state MSE / MAE / R²
  aggregate + per-feature worst/best tables (`temporal_metrics`).
- Classification: accuracy, precision, recall, F1, FPR, confusion matrix —
  computed only when both classes exist in the test tail, else reported as
  degenerate (no fabricated numbers).
- Baseline: Logistic Regression on the last observed state `X[:,-1,:]`, same
  splits, same metrics → written to `evaluation_report.json`.
- Claim policy: superiority is asserted **only** from measured report values.

## Hardware

Target: RTX 3050 Laptop 6 GB (batch 32 @ d_model 128 fits). CPU fallback fully
supported and tested (~20 s synthetic run). Actual CIC-IDS2018 training is
**pending** dataset availability — no results are claimed until it runs.
