"""Temporal Transformer World Model — multi-head architecture.

Architecture (per ml/MODEL.md):

    input [B, L, F]
      -> feature projection Linear(F, d_model)
      -> learned positional embedding
      -> N x TemporalEncoderBlock (self-attention + FFN, causal masking)
      -> latent state z(t) = hidden at last position [B, d_model]

    heads:
      next_state_head  Linear(d_model, F)   predicts normalized S(t+K) features
      malicious_head   Linear(d_model, 1)   logit P(malicious activity)
      risk_head        Linear(d_model, 1)   logit risk score (0..1 after sigmoid)
      confidence_head  Linear(d_model, 1)   self-consistency estimate of its own
                                            next-state prediction quality

The world model learns *temporal dynamics* via future-state prediction;
classification heads are auxiliary consumers of the same latent dynamics.
"""

import math

import torch
import torch.nn as nn

from ..utils.seeding import count_parameters


class TemporalEncoderBlock(nn.Module):
    """Pre-norm transformer block that can expose attention weights."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, need_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        # MultiheadAttention returns (out, averaged attn weights) when average_attn_weights=True
        attn_out, weights = self.attn(
            h, h, h, attn_mask=attn_mask, need_weights=need_attention, average_attn_weights=True
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, weights


class TemporalTransformerWorldModel(nn.Module):
    """Multi-head temporal world model over NetworkState sequences."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        *,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        backbone: str = "temporal_transformer",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if backbone not in ("temporal_transformer", "lstm"):
            raise ValueError(f"unknown backbone: {backbone}")
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.backbone = backbone
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        if backbone == "temporal_transformer":
            self.blocks = nn.ModuleList(
                [
                    TemporalEncoderBlock(d_model, n_heads, d_ff, dropout)
                    for _ in range(n_layers)
                ]
            )
            self.final_norm = nn.LayerNorm(d_model)
        else:  # lstm baseline sharing the same head interface
            self.lstm = nn.LSTM(
                d_model, d_model // 2, num_layers=n_layers, batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0, bidirectional=True,
            )

        # ---- multi-head outputs ----
        self.next_state_head = nn.Linear(d_model, input_dim)
        self.malicious_head = nn.Linear(d_model, 1)
        self.risk_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)

    def encode(
        self, x: torch.Tensor, need_attention: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """x: [B, L, F] -> latent [B, d_model] (+ per-layer avg attention [B, L, L])."""
        B, L, _ = x.shape
        if L != self.sequence_length:
            raise ValueError(f"expected sequence length {self.sequence_length}, got {L}")
        h = self.input_proj(x) + self.pos_embed[:, :L, :]
        attentions: list[torch.Tensor] = []
        if self.backbone == "temporal_transformer":
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
            )
            for block in self.blocks:
                h, w = block(h, attn_mask=causal, need_attention=need_attention)
                if need_attention:
                    attentions.append(w)
            h = self.final_norm(h)
        else:
            h, _ = self.lstm(h)
        return h[:, -1, :], attentions

    def forward(self, x: torch.Tensor, need_attention: bool = False) -> dict[str, torch.Tensor]:
        latent, attentions = self.encode(x, need_attention=need_attention)
        return {
            "latent": latent,
            "next_state_pred": self.next_state_head(latent),
            "malicious_logit": self.malicious_head(latent).squeeze(-1),
            "risk_logit": self.risk_head(latent).squeeze(-1),
            "confidence_logit": self.confidence_head(latent).squeeze(-1),
            "attentions": attentions,
        }

    def num_parameters(self) -> dict[str, int]:
        return count_parameters(self)


def build_model(input_dim: int, cfg) -> TemporalTransformerWorldModel:
    """cfg: WorldModelConfig."""
    return TemporalTransformerWorldModel(
        input_dim=input_dim,
        sequence_length=cfg.sequence.length,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        backbone=cfg.model.backbone,
    )


__all__ = ["TemporalTransformerWorldModel", "TemporalEncoderBlock", "build_model"]
