# THREATCAST World Model — design notes

Status: design contract for later implementation phases. Nothing here is implemented yet.

## Conceptual layers (kept strictly separate)

1. **Network state representation** — canonical `NetworkState` → fixed-length vector via saved `feature_schema.json`.
2. **Temporal sequence modelling** — encoder over `S[t-L+1..t]` (default L=5): lightweight Temporal Transformer (~4 layers, d_model 128, 4 heads); LSTM baseline.
3. **Future state prediction** — head predicting the next-state distribution; autoregressive rollout to K steps (default K=3) with uncertainty accumulation.
4. **Malicious probability / risk score** — heads over the latent sequence representation, outputs bounded to [0,1].
5. **Attack progression** — optional derived-stage logits over the Reconnaissance→Exfiltration taxonomy (see docs/MITRE_MAPPING.md). Derived stages are never presented as dataset ground truth unless verified.
6. **Explainability** — attention inspection and/or post-hoc attribution producing `feature_contributions` per prediction.

## Interface contract

```python
def predict(sequence: NetworkStateSequence) -> PredictionResult
```

Implemented in a later phase inside `src/inference/`. The backend calls this through a thin serving facade.

## Model size rationale

Target hardware: RTX 3050 Laptop, 6 GB VRAM. Budget: batch ≤ 64 sequences of length 5 × ~50 float features at fp16 fits trivially; the constraint is keeping training epochs fast on a laptop GPU, hence a small transformer. Exact hyperparameters are set in `configs/model.yaml` at implementation time.

## Artifacts produced by training

- `artifacts/world_model.pt` (+ versioned copies)
- `artifacts/model_metadata.json` — config, git SHA, metrics
- `artifacts/feature_schema.json`, `artifacts/label_mappings.json`

Naming conventions: CONTRACT.md §15.
