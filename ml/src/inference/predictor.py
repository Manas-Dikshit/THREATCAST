"""Stable inference interface - the only module Phase 4 (backend) should import.

    predictor = WorldModelPredictor.load("ml/artifacts")
    result = predictor.predict(sequence)          # -> backend PredictionResult
    future = predictor.rollout(sequence, horizon=3)
    nxt    = predictor.predict_next_state(sequence)

Returns backend/app/schemas/prediction.py objects so the API layer needs no
knowledge of torch/model internals.
"""

import json
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch

from backend.app.schemas.prediction import (
    FeatureContribution,
    FutureStateEntry,
    ModelInfo,
    PredictionResult,
)
from data_pipeline.src.schemas.network_state import NetworkStateSequence

from ..models.world_model import TemporalTransformerWorldModel
from ..states.encoder import StateVectorizer


class WorldModelPredictor:
    def __init__(self, model: TemporalTransformerWorldModel, vectorizer: StateVectorizer,
                 *, horizon: int = 3, model_version: str = "1.0.0", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vectorizer = vectorizer
        self.horizon = int(horizon)
        self.model_version = str(model_version)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, artifacts_dir: str | Path, device: str = "auto") -> "WorldModelPredictor":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        d = Path(artifacts_dir)
        bundle_path = d / "world_model.pt"
        if not bundle_path.exists():
            raise FileNotFoundError(f"world_model.pt not found in {d} - train first")
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

        cfg = bundle["config"]
        vectorizer = StateVectorizer.from_artifacts(d)
        model = TemporalTransformerWorldModel(
            input_dim=vectorizer.input_dim,
            sequence_length=int(cfg["sequence_length"]),
            d_model=int(cfg["d_model"]), n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]), d_ff=int(cfg["d_ff"]),
            dropout=float(cfg["dropout"]),
            backbone=str(cfg.get("backbone", "temporal_transformer")),
        )
        model.load_state_dict(bundle["model_state_dict"])

        version = str(bundle.get("model_version", "1.0.0"))
        meta_p = d / "model_metadata.json"
        if meta_p.exists():
            try:
                version = str(json.loads(meta_p.read_text(encoding="utf-8")).get("model_version", version))
            except json.JSONDecodeError:
                pass
        return cls(model, vectorizer,
                   horizon=int(cfg.get("prediction_horizon", 3)),
                   model_version=version, device=device)

    # ------------------------------------------------------------- internals
    def _tensor(self, sequence: NetworkStateSequence) -> torch.Tensor:
        if sequence is None:
            raise ValueError("INVALID_SEQUENCE: sequence is None")
        if not sequence.states:
            raise ValueError("INVALID_SEQUENCE: no states")
        arr = self.vectorizer.sequence_tensor(sequence)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def _forward_heads(self, x: torch.Tensor) -> dict:
        out = self.model(x)
        sig = lambda t: float(torch.sigmoid(t)[0])  # noqa: E731
        return {"risk": sig(out["risk_logit"]), "malicious": sig(out["malicious_logit"]),
                "confidence": sig(out["confidence_logit"]),
                "next_state_pred": out["next_state_pred"][0].detach().float().cpu().numpy()}

    # ------------------------------------------------------------ public API
    @torch.no_grad()
    def predict_next_state(self, sequence: NetworkStateSequence) -> dict[str, float]:
        """S(t+1) as denormalized named features."""
        heads = self._forward_heads(self._tensor(sequence))
        return self.vectorizer.denormalize(heads["next_state_pred"])

    @torch.no_grad()
    def rollout(self, sequence: NetworkStateSequence,
                horizon: int | None = None) -> list[FutureStateEntry]:
        """K-step autoregressive forward simulation.

        The model's own predicted (normalized) state is fed back as input:
            S(t+1) = f(S(t-3)..S(t));  S(t+2) = f(S(t-2)..S(t+1));  ...
        Per-step confidence comes from the confidence head at each simulated step.
        """
        h = self.horizon if horizon is None else int(horizon)
        if h < 1:
            raise ValueError("horizon must be >= 1")
        window = self._tensor(sequence)

        step_seconds = sequence.window_seconds or 10
        last_ts = sequence.states[-1].timestamp_end
        entries: list[FutureStateEntry] = []

        for k in range(1, h + 1):
            out = self.model(window)
            next_norm = out["next_state_pred"][0]
            conf = float(torch.sigmoid(out["confidence_logit"][0]))
            features = self.vectorizer.denormalize(next_norm.detach().float().cpu().numpy())
            entries.append(FutureStateEntry(
                step=k,
                timestamp=(last_ts + timedelta(seconds=step_seconds * k)) if last_ts else None,
                features={n: round(v, 6) for n, v in features.items()},
                confidence=round(conf, 4),
            ))
            window = torch.cat([window[:, 1:, :], next_norm.detach().view(1, 1, -1).float()], dim=1)
        return entries

    def predict(self, sequence: NetworkStateSequence,
                *, explain: bool = False) -> PredictionResult:
        """Full PredictionResult (CONTRACT.md section 7): risk now + K-step timeline."""
        heads = self._forward_heads(self._tensor(sequence))
        contributions = self.attribute_features(sequence) if explain else []
        return PredictionResult(
            prediction_id=f"pred_{uuid.uuid4().hex[:12]}",
            risk_score=round(heads["risk"], 4),
            malicious_probability=round(heads["malicious"], 4),
            confidence=round(heads["confidence"], 4),
            future_states=self.rollout(sequence),
            feature_contributions=[
                FeatureContribution(feature=n, contribution=float(c)) for n, c in contributions
            ],
            model=ModelInfo(name="threatcast-world-model", version=self.model_version),
        )

    # ------------------------------------------------- explainability hooks
    @torch.no_grad()
    def get_attentions(self, sequence: NetworkStateSequence) -> list[np.ndarray]:
        """Per-layer averaged attention maps [L, L] for Phase 5 consumption."""
        x = self._tensor(sequence)
        out = self.model(x, need_attention=True)
        return [w[0].detach().float().cpu().numpy() for w in out["attentions"]]

    def attribute_features(self, sequence: NetworkStateSequence) -> list[tuple[str, float]]:
        """Input-gradient saliency for the malicious head.

        contribution(f) = mean_t | d(logit_malicious)/d(x_t,f) * x_t,f |
        Deterministic, cheap; SHAP integration deferred to Phase 5.
        """
        x = self._tensor(sequence).requires_grad_(True)
        logit = self.model(x)["malicious_logit"][0]
        grads = torch.autograd.grad(logit, x)[0][0]  # [L, F]
        scores = (grads.abs() * x.detach()[0].abs()).mean(dim=0).cpu().numpy()
        ranked = sorted(zip(self.vectorizer.feature_names, scores.tolist()),
                        key=lambda kv: kv[1], reverse=True)
        return ranked


__all__ = ["WorldModelPredictor"]
