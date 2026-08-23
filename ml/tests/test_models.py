"""Model initialization, tensor shapes, forward pass, attention exposure."""

import pytest
import torch

from ml.src.models.world_model import TemporalTransformerWorldModel, build_model


def make_model(**kw) -> TemporalTransformerWorldModel:
    return TemporalTransformerWorldModel(input_dim=9, sequence_length=5, **kw)


class TestInitialization:
    def test_default_init(self):
        m = make_model()
        assert m.input_dim == 9 and m.sequence_length == 5
        assert len(m.blocks) == 4
        assert m.next_state_head.out_features == 9

    def test_invalid_head_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            make_model(d_model=100, n_heads=6)

    def test_unknown_backbone(self):
        with pytest.raises(ValueError, match="backbone"):
            make_model(backbone="gru")

    def test_build_model_from_config(self):
        from ml.tests.conftest import make_tiny_config

        cfg = make_tiny_config()
        cfg.sequence.length = 5
        m = build_model(input_dim=9, cfg=cfg)
        assert isinstance(m, TemporalTransformerWorldModel)
        assert m.d_model == 32


class TestShapes:
    def test_forward_output_shapes(self):
        m = make_model(d_model=32, n_layers=2, n_heads=4)
        x = torch.randn(3, 5, 9)
        out = m(x)
        assert out["latent"].shape == (3, 32)
        assert out["next_state_pred"].shape == (3, 9)
        for head in ("malicious_logit", "risk_logit", "confidence_logit"):
            assert out[head].shape == (3,)
        assert torch.isfinite(out["next_state_pred"]).all()

    def test_wrong_sequence_length_rejected(self):
        m = make_model()
        with pytest.raises(ValueError, match="sequence length"):
            m(torch.randn(1, 7, 9))

    def test_attention_maps(self):
        m = make_model(d_model=16, n_layers=2, n_heads=4).eval()
        with torch.no_grad():
            out = m(torch.randn(1, 5, 9), need_attention=True)
        assert len(out["attentions"]) == 2
        for w in out["attentions"]:
            assert w.shape == (1, 5, 5)
            row_sums = w.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_lstm_baseline_interface(self):
        m = make_model(backbone="lstm", d_model=16, n_layers=1, n_heads=4)
        out = m(torch.randn(2, 5, 9))
        assert out["latent"].shape == (2, 16)
        assert out["attentions"] == []


class TestForwardPass:
    def test_deterministic_in_eval(self):
        m = make_model(d_model=16, n_layers=1).eval()
        x = torch.randn(1, 5, 9)
        with torch.no_grad():
            a = m(x)["risk_logit"]
            b = m(x)["risk_logit"]
        assert torch.equal(a, b)

    def test_gradients_flow(self):
        m = make_model(d_model=16, n_layers=1)
        loss = m(torch.randn(2, 5, 9))["next_state_pred"].sum()
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert len(grads) > 0
