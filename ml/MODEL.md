# World Model Architecture & Objective

## Model: `TemporalTransformerWorldModel` (ml/src/models/world_model.py)

```
NetworkStateSequence [B, L, F]        L=5 states, F=|feature_names|
        │
  Linear(F → d_model) + LayerNorm + Dropout          state encoding
        │
  + learned positional embedding [1, L, d_model]
        │
  N × TemporalEncoderBlock                           temporal modelling
      (causal self-attention, avg-attn exposed,
       pre-norm, GELU FFN d_ff, residual + dropout)
        │
  z(t) = hidden at last position [B, d_model]        latent network state
        │
  ┌────────────┬──────────────┬────────────┬────────────────┐
  next_state   malicious      risk         confidence       (all Linear(d_model, ·))
  head         head           head         head
  S(t+K) pred  P(malicious)   risk score   self-consistency
```

- **backbone**: `temporal_transformer` (default) or `lstm` (baseline, same heads/interface).
- Causal attention mask: position t only attends to ≤ t.
- Default size: d_model=128, layers=4, heads=4, d_ff=256 → ~800 K params. Sized for RTX 3050 6 GB with batch 32; runs comfortably on CPU for inference.

## Mathematical objective

Let x_{1..L} be the normalized input window, y = normalized true future state
S(t+K), m ∈ {0,1} malicious label (verified labels only), c* the confidence target.

```
L_total = w_ns·MSE(ŷ_ns, y)·[target exists]      ← world-model dynamics term
        + w_mal·BCE(z_mal, m)·[labeled]           ← classification
        + w_risk·BCE(z_risk, m)·[labeled]         ← separate calibration head
        + w_conf·MSE(σ(z_conf), c*)              ← self-consistency

c* = exp( MSE(ŷ_ns, y) / mean(y²) )             per-sample, detached
```

Defaults: w_ns=1.0, w_mal=1.0, w_risk=0.5, w_conf=0.2 (`loss_weights` in config).

Design notes:
- The **next-state term is the world model**: it forces the encoder+latent to represent traffic dynamics, not merely discriminate labels. Classification heads are auxiliary consumers of the same latent.
- Malicious and risk are separate linear heads on shared latent so their calibration can diverge later (e.g., ordinal risk bands) without retraining the encoder.
- Confidence is trained against its own realized forecast error — a measurable self-consistency signal, not an invented probability.

## Inputs / outputs

- Input tensor: `[L, F]` float32, z-scored with train-only stats from Phase 2 (`preprocessing_metadata.json`). Missing features → train mean (= 0 after normalization). Column order = `feature_schema.json:feature_names`.
- Outputs: next-state vector (denormalized to named features by `StateVectorizer.denormalize`), malicious prob ∈ [0,1], risk score ∈ [0,1], confidence ∈ [0,1].

## K-step rollout (`WorldModelPredictor.rollout`)

Autoregressive in normalized space:

```
ŝ(t+1) = f(x_{t-L+1..t});  ŝ(t+2) = f(x_{t-L+2..t}, ŝ(t+1));  ...  ŝ(t+K)
```

Each step's confidence is read from the confidence head at that step; timestamps advance by `window_seconds`. Default K = `ML_PREDICTION_HORIZON` = 3.

## Explainability hooks (for Phase 5)

- `get_attentions(seq)` → per-layer averaged attention `[L, L]` (custom blocks expose `need_weights`; no forward-hook hacks).
- `attribute_features(seq)` → input-gradient saliency for the malicious head: `mean_t |∂logit/∂x_tf · x_tf|`, sorted descending; surfaced as `PredictionResult.feature_contributions`.
- SHAP integration deferred (Phase 5); the hooks above give the same per-feature ranking signal deterministically.

## What this model is NOT

- Not a flow classifier: it predicts future state vectors and is evaluated on them.
- No attack-stage output (`predicted_stage.source = null`) — no fabricated ground truth.
