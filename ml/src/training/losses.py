"""Loss design for the multi-head world model.

    L = w_ns * L_next_state + w_mal * L_malicious + w_risk * L_risk + w_conf * L_confidence

  - L_next_state : masked MSE between predicted and true normalized S(t+K)
                   features. Only sequences whose target state exists
                   (target_mask) contribute. This is the *world model* term —
                   it teaches temporal dynamics, not classification.
  - L_malicious  : BCEWithLogits of the malicious head vs binary label
                   (1 = attack-family label, 0 = benign). Sequences without a
                   verified label (mal_mask False) are excluded.
  - L_risk       : BCEWithLogits on a separate risk head using the same target.
                   Kept distinct from the malicious head so risk calibration
                   can later be tuned independently (e.g. ordinal risk bands).
  - L_confidence : MSE of the confidence head vs exp(-relative next-state
                   error) computed from the actual prediction error — a
                   self-consistency signal: the model learns to predict how
                   wrong its own forecast is.

All terms are individually masked; empty masks yield zero contribution.
"""

import torch
import torch.nn.functional as F


def confidence_target(next_state_pred: torch.Tensor, next_state_true: torch.Tensor,
                      target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """exp(-rel_mse): 1.0 when prediction is perfect, ->0 as error grows."""
    if target_mask.sum() == 0:
        return torch.zeros_like(target_mask, dtype=torch.float32)
    sq_err = ((next_state_pred - next_state_true) ** 2).mean(dim=1)
    var = (next_state_true**2).mean(dim=1) + eps
    rel = sq_err / var
    conf = torch.exp(-rel)
    return torch.where(target_mask, conf.float(), torch.zeros_like(conf))


def world_model_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                     weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (total_loss, per-term diagnostics)."""
    terms: dict[str, torch.Tensor] = {}

    ns_mask = batch["target_mask"]
    if ns_mask.any():
        terms["next_state"] = F.mse_loss(
            outputs["next_state_pred"][ns_mask], batch["Y"][ns_mask]
        )
    else:
        terms["next_state"] = torch.zeros((), device=outputs["next_state_pred"].device)

    mal_mask = batch["mal_mask"]
    if mal_mask.any():
        targets = batch["mal_target"][mal_mask]
        terms["malicious"] = F.binary_cross_entropy_with_logits(
            outputs["malicious_logit"][mal_mask], targets
        )
        terms["risk"] = F.binary_cross_entropy_with_logits(
            outputs["risk_logit"][mal_mask], targets
        )
    else:
        zero = torch.zeros((), device=outputs["malicious_logit"].device)
        terms["malicious"] = zero
        terms["risk"] = zero

    # confidence only supervised where we know the true future state
    conf_t = confidence_target(
        outputs["next_state_pred"].detach(), batch["Y"], ns_mask
    ).to(outputs["confidence_logit"].device)
    if ns_mask.any():
        terms["confidence"] = F.mse_loss(torch.sigmoid(outputs["confidence_logit"][ns_mask]), conf_t[ns_mask])
    else:
        terms["confidence"] = torch.zeros((), device=outputs["confidence_logit"].device)

    total = (
        weights.get("next_state", 1.0) * terms["next_state"]
        + weights.get("malicious", 1.0) * terms["malicious"]
        + weights.get("risk", 0.5) * terms["risk"]
        + weights.get("confidence", 0.2) * terms["confidence"]
    )

    diagnostics = {
        "next_state": float(terms["next_state"].detach()),
        "malicious": float(terms["malicious"].detach()),
        "risk": float(terms["risk"].detach()),
        "confidence": float(terms["confidence"].detach()),
        "total": float(total.detach()),
    }
    return total, diagnostics


__all__ = ["world_model_loss", "confidence_target"]
