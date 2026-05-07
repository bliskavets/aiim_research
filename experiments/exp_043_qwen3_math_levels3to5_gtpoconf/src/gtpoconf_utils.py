"""
gtpoconf_utils.py
-----------------
Vectorized GTPO-Conf: column-wise group-relative confidence weighting.

Algorithm (from exp_005, vectorized):
  C_i,t = -mean_{top-k}(log π(v | context))   [higher = more uncertain]

  For O+ sequences:
    C_comp = log(1 + C)  [log-compressed confidence]
    Bonus at column t: (C_comp / sum_col(C_comp)) * d_t   where d_t = #O+ seqs at t
    shaped_pos = α₁ * mask + α₂ * bonus

  For O- sequences:
    C_inv = log(1 + 1/C)  [inverse: small C → large penalty]
    Penalty at col t: (C_inv / sum_col(C_inv)) * h_t   where h_t = #O- seqs at t
    shaped_neg = -(α₁ * mask + α₂ * penalty)

  Both are z-normalized separately, then summed.

Key difference from exp_039/042 EMA-flipped:
  - No EMA time dependency; normalization is across sequences at each timestep
  - O+ bonus rewards exploration on correct paths (uncertain → larger bonus)
  - O- penalty punishes confident mistakes (confident → larger penalty via 1/C)
"""

import torch

EPS = 1e-8


def compute_gtpo_conf_advantages(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.1,
    reward_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Compute GTPO-Conf per-token advantages (vectorized, no Python loop).

    Args:
        rewards:          (B,) sequence rewards
        confidence:       (B, T) per-token C = -mean_topk(log π), higher = more uncertain
        completion_mask:  (B, T) 1 for valid tokens
        alpha1, alpha2:   shaping weights (base + confidence bonus)
        reward_threshold: O+/O- split boundary

    Returns:
        token_advantages: (B, T)
    """
    is_pos = (rewards > reward_threshold)       # (B,)
    is_neg = ~is_pos

    C_comp = torch.log1p(confidence)                     # (B, T), log(1+C)
    C_inv  = torch.log1p(1.0 / (confidence + EPS))      # (B, T), log(1+1/C)

    mask_pos = completion_mask * is_pos.float().unsqueeze(1)   # (B, T)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)   # (B, T)

    # ── O+ shaped rewards (column-wise normalization) ─────────────────────────
    d_col  = mask_pos.sum(0, keepdim=True).clamp(min=EPS)     # (1, T)
    C_pos  = C_comp * mask_pos                                  # (B, T)
    sum_C  = C_pos.sum(0, keepdim=True) + EPS                  # (1, T)
    bonus  = (C_pos / sum_C) * d_col                           # (B, T)
    shaped_pos = alpha1 * mask_pos + alpha2 * bonus            # (B, T)

    # ── O- shaped rewards (column-wise normalization) ─────────────────────────
    h_col  = mask_neg.sum(0, keepdim=True).clamp(min=EPS)     # (1, T)
    I_neg  = C_inv * mask_neg                                   # (B, T)
    sum_I  = I_neg.sum(0, keepdim=True) + EPS                  # (1, T)
    penalty = (I_neg / sum_I) * h_col                          # (B, T)
    shaped_neg = -(alpha1 * mask_neg + alpha2 * penalty)       # (B, T)

    # ── Z-normalize O+ and O- separately ─────────────────────────────────────
    adv_pos = _znorm(shaped_pos, mask_pos)
    adv_neg = _znorm(shaped_neg, mask_neg)

    return adv_pos + adv_neg


def _znorm(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Z-normalize x over active (masked) positions only."""
    active = x[mask.bool()]
    if active.numel() <= 1:
        return x * mask
    mean = active.mean()
    std  = active.std() + EPS
    return ((x - mean) / std) * mask
