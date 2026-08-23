# THREATCAST ML

World model, temporal sequence modelling, forward simulation, training and evaluation.

**Phase 1 status:** skeleton + contracts only. No model code, no trained artifacts, no fabricated results.

## Scope (Module 2)

- `src/states/` — NetworkState ↔ tensor conversion using the saved feature schema
- `src/models/` — Temporal Transformer world model; LSTM baseline
- `src/training/` — offline training on CIC-IDS2018-derived state sequences
- `src/inference/` — `predict(NetworkStateSequence) -> PredictionResult`
- `src/evaluation/` — metrics vs baselines
- `src/data/`, `src/features/`, `src/utils/` — dataset loading, preprocessing helpers shared with contracts from `data_pipeline`

## Hard rules (from CONTRACT.md)

- Learn `P(S[t+1] | S[t])`; support rollout to S[t+K]. Not a static binary classifier.
- Public interface: `predict(sequence: NetworkStateSequence) -> PredictionResult`.
- Hardware budget: RTX 3050 Laptop 6 GB VRAM — mixed precision, gradient accumulation, CPU fallback (`ML_DEVICE=auto|cuda|cpu`).
- CIC-IDS2018: profile actual files; never assume columns; no leakage; time-aware splits; save `feature_schema.json` + label mappings.
- Never fabricate ground truth or evaluation numbers.

See [MODEL.md](MODEL.md), [TRAINING.md](TRAINING.md), [../docs/DATA_SCHEMA.md](../docs/DATA_SCHEMA.md).
