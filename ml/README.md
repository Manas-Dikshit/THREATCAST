# THREATCAST ML Layer (Phase 3)

Temporal world model for network-state forecasting and risk prediction.

**Status: fully implemented. Validated end-to-end on synthetic data; CIC-IDS2018 training pending dataset availability.**

## Layout

```
ml/
├── src/
│   ├── models/       TemporalTransformerWorldModel (multi-head) + LSTM variant
│   ├── training/     dataset loading, losses, Trainer, train CLI
│   ├── inference/    WorldModelPredictor — the stable Phase 4 interface
│   ├── evaluation/   metrics, Logistic Regression baseline, compare CLI
│   ├── states/       StateVectorizer (schema + normalization)
│   └── utils/        config (yaml + env), seeding
├── configs/model.yaml
├── notebooks/        01..08 workflow notebooks
├── artifacts/        world_model.pt, metadata, schemas (gitkeep'd until trained)
├── checkpoints/      latest.pt / best.pt per run
└── tests/            52 synthetic-data tests (no dataset required)
```

## Quick start

```powershell
# 1. produce a Phase 2 dataset export (CSV or PCAP input)
python -m data_pipeline.cli --input <file> --out <export_dir>

# 2. train (config: ml/configs/model.yaml, env overrides win)
python -m ml.src.training.train --dataset-dir <export_dir> --artifacts-dir ml/artifacts

# 3. evaluate vs baseline
python -m ml.src.evaluation.evaluate --dataset-dir <export_dir> --artifacts-dir ml/artifacts

# 4. tests (synthetic only, ~20 s, CPU)
pytest ml/tests -q
```

## The one interface Phase 4 needs

```python
from ml.src.inference import WorldModelPredictor
from data_pipeline.src.schemas.network_state import NetworkStateSequence

predictor = WorldModelPredictor.load("ml/artifacts")          # device auto
result = predictor.predict(sequence)                           # backend PredictionResult
future = predictor.rollout(sequence, horizon=3)                # [FutureStateEntry]
nxt    = predictor.predict_next_state(sequence)                # dict[str, float]
attn   = predictor.get_attentions(sequence)                    # explainability hook
contribs = predictor.predict(sequence, explain=True).feature_contributions
```

The backend never imports torch or model classes — it consumes
`backend.app.schemas.prediction.PredictionResult` objects directly.

See `MODEL.md` (architecture/objective), `TRAINING.md` (procedure/splits),
`docs/MODEL.md` (contract-level view).

## Limitations

- Real-dataset training not yet run (no CIC-IDS2018 files present in repo).
- CPU-only validation so far; AMP/CUDA paths are unit-tested but unexercised on GPU hardware.
- Attack-stage prediction intentionally left null (`source: null`) — no fabricated ground truth.
