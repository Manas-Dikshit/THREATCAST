# THREATCAST Model Documentation

Design-level doc; implementation detail lives in [ml/MODEL.md](../ml/MODEL.md) and [ml/TRAINING.md](../ml/TRAINING.md).

> **Status (Phase 3): implemented.** All layers below exist in `ml/src/` and are
> covered by 52 synthetic-data tests (`pytest ml/tests`). Actual CIC-IDS2018
> training remains pending dataset availability — no performance numbers are
> claimed until it runs on the real files.

## What the model must be

A **world model** of the network: it learns the transition dynamics

```
P(S[t+1] | S[t-L+1], ..., S[t])
```

over canonical network states, and supports autoregressive rollout to arbitrary horizon K. It is explicitly **not** a static binary classifier.

## Layered architecture (each independently testable)

1. **State representation** — `NetworkState.features` (open dict) → fixed vector via saved `feature_schema.json` → `ml/src/states/encoder.py`.
2. **Temporal sequence modelling** — lightweight Temporal Transformer encoder over L=5 states (baseline: LSTM) → `ml/src/models/world_model.py`.
3. **Future state prediction** — next-state head; forward simulator unrolls S[t+1..K] → `WorldModelPredictor.rollout`.
4. **Malicious probability / risk score** — bounded [0,1] heads on the latent representation; separate calibration head, documented loss design in `ml/src/training/losses.py`.
5. **Attack progression** — intentionally not implemented yet; `predicted_stage.source = null` (never fabricated).
6. **Explainability** — per-layer attention maps + input-gradient feature attribution exposed now for Phase 5 (`get_attentions`, `attribute_features`).

## Hardware contract (RTX 3050 Laptop, 6 GB VRAM, Windows)

- CUDA when available (`ML_DEVICE=auto|cuda|cpu`), automatic CPU fallback
- mixed precision (AMP), configurable batch size and sequence length
- gradient accumulation for effective-batch scaling
- checkpointing to `ml/checkpoints/`
- configurable dataloader workers

## Dataset rules (CIC-IDS2018)

- Inspect actual downloaded files before any assumption about columns
- dynamically profile columns; handle inconsistent naming, missing values, duplicates, infinities, categoricals, IP/port/protocol fields
- avoid leakage; time-aware train/val/test splits where appropriate
- handle class imbalance deliberately
- save preprocessing metadata, feature schema, label mappings; make training reproducible (seeded, config-driven)
- dataset is supplied manually — never auto-downloaded

## Stable inference interface (consumed by backend, Phase 4)

```python
from ml.src.inference import WorldModelPredictor

predictor = WorldModelPredictor.load("ml/artifacts")   # torch-free usage afterwards
result: PredictionResult = predictor.predict(sequence)  # backend schema object
future = predictor.rollout(sequence, horizon=3)
nxt = predictor.predict_next_state(sequence)
```

The backend depends only on `WorldModelPredictor` and the canonical
`backend.app.schemas.prediction` models — never on torch or model internals.
