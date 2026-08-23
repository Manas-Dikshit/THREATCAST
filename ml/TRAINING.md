# THREATCAST Training Guide (contract for later implementation)

Status: specification only — no training code exists yet.

## Reproducibility requirements

- Config-driven runs (`ml/configs/model.yaml` + CLI overrides); no magic numbers in code
- Fixed random seeds logged into `model_metadata.json`
- Git SHA + dataset file hash recorded per run
- Artifacts versioned: `world_model.pt`, `world_model_v<semver>.pt`

## Pipeline

1. Load state sequences produced by `data_pipeline` (canonical `NetworkStateSequence`)
2. Vectorise via saved `feature_schema.json`; persist any fitted scalers as artifacts
3. Split **time-aware** where possible (train on past, validate/test on later windows); no leakage across the temporal boundary
4. Train with AMP + gradient accumulation on CUDA; CPU fallback must produce identical logic
5. Evaluate against baselines (persistence predictor, LSTM) and save metrics to metadata
6. Register final artifact in `ml/artifacts/` and record in DB `models` table once backend phase lands

## Dataset handling (CIC-IDS2018)

Profile real files before use; handle missing/duplicate/infinite values and inconsistent column names; map labels to the canonical label set only after verification; document every transformation in saved preprocessing metadata.
