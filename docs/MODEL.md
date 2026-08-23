# THREATCAST Model Documentation

Design-level doc; implementation detail lives in [ml/MODEL.md](../ml/MODEL.md).

## What the model must be

A **world model** of the network: it learns the transition dynamics

```
P(S[t+1] | S[t-L+1], ..., S[t])
```

over canonical network states, and supports autoregressive rollout to arbitrary horizon K. It is explicitly **not** a static binary classifier.

## Layered architecture (each independently testable)

1. **State representation** — `NetworkState.features` (open dict) → fixed vector via saved `feature_schema.json`.
2. **Temporal sequence modelling** — lightweight Temporal Transformer encoder over L=5 states (baseline: LSTM).
3. **Future state prediction** — next-state head; forward simulator unrolls S[t+1..K].
4. **Malicious probability / risk score** — bounded [0,1] heads on the latent representation.
5. **Attack progression** — optional derived-stage logits (Recon → Exfiltration); always marked derived.
6. **Explainability** — attention/attribution → ordered feature contributions per prediction.

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
