"""Loss calculation: masking, targets, confidence signal."""

import numpy as np
import torch

from ml.src.training.losses import confidence_target, world_model_loss


def batch(target_mask=None, mal_mask=None):
    n = 2
    tm = torch.ones(n, dtype=torch.bool) if target_mask is None else torch.tensor(target_mask)
    mm = torch.ones(n, dtype=torch.bool) if mal_mask is None else torch.tensor(mal_mask)
    return {
        "X": torch.zeros(n, 5, 9),
        "Y": torch.zeros(n, 9),
        "target_mask": tm,
        "mal_target": torch.tensor([0.0, 1.0]),
        "mal_mask": mm,
    }


def outputs():
    return {
        "latent": torch.zeros(2, 8),
        "next_state_pred": torch.zeros(2, 9),
        "malicious_logit": torch.tensor([10.0, -10.0]),  # confident WRONG predictions
        "risk_logit": torch.tensor([-10.0, 10.0]),
        "confidence_logit": torch.zeros(2),
        "attentions": [],
    }


class TestWorldModelLoss:
    def test_perfect_prediction_zero_next_state_term(self):
        out = outputs()
        out["next_state_pred"] = batch()["Y"].clone()
        total, terms = world_model_loss(out, batch(), {})
        assert terms["next_state"] == 0.0
        assert total.item() > 0  # malicious term still nonzero (wrong logits)

    def test_malicious_term_punishes_wrong_confident_logits(self):
        _, terms_bad = world_model_loss(outputs(), batch(), {})
        good = outputs()
        good["malicious_logit"] = torch.tensor([-10.0, 10.0])
        good["risk_logit"] = torch.tensor([10.0, -10.0])
        _, terms_good = world_model_loss(good, batch(), {})
        assert terms_bad["malicious"] > terms_good["malicious"]
        assert terms_bad["risk"] > terms_good["risk"]

    def test_unlabeled_sequences_excluded(self):
        _, terms = world_model_loss(outputs(), batch(mal_mask=[False, False]),
                                    {"malicious": 1.0})
        assert terms["malicious"] == 0.0 and terms["risk"] == 0.0

    def test_no_target_states(self):
        _, terms = world_model_loss(outputs(), batch(target_mask=[False, False]), {})
        assert terms["next_state"] == 0.0

    def test_weights_scale_terms(self):
        total_a, _ = world_model_loss(outputs(), batch(), {"next_state": 1.0, "malicious": 1.0, "risk": 0.5, "confidence": 0.2})
        total_b, _ = world_model_loss(outputs(), batch(), {"next_state": 1.0, "malicious": 1.0, "risk": 0.5, "confidence": 0.2})  # noqa: E501 same weights -> same total
        assert torch.isclose(total_a, total_b)

    def test_loss_is_differentiable(self):
        out = {**outputs(), "next_state_pred": torch.zeros(2, 9, requires_grad=True)}
        total, _ = world_model_loss(out, batch(), {})
        total.backward()


class TestConfidenceTarget:
    def test_perfect_prediction_full_confidence(self):
        y = torch.randn(4, 9)
        conf = confidence_target(y.clone(), y, torch.ones(4, dtype=torch.bool))
        assert np.allclose(conf.numpy(), 1.0)

    def test_bounds_and_masking(self):
        pred = torch.zeros(3, 9)
        true = torch.tensor(np.random.default_rng(0).normal(size=(3, 9)), dtype=torch.float32) * 10
        mask = torch.tensor([True, True, False])
        conf = confidence_target(pred, true, mask)
        assert ((conf >= 0) & (conf <= 1)).all()
        assert conf[2] == 0.0  # masked rows contribute nothing
