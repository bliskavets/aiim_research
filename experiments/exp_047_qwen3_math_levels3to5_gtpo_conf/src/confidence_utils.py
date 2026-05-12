"""
confidence_utils.py
-------------------
Confidence-based reward shaping for GTPO-Conf (exp_047).

Ported from exp_005 with vectorised loops (no Python for-t-in-range(T) —
all per-position ops are fused into batch matrix operations).

Confidence metric:
    C_i,t = -mean_{v ∈ top-k}( log π(v | context_i,t) )
    Large C → uncertain (spread probability); Small C → peaked/certain.

For O+ (correct):  bonus ∝ log(1+C)     → reward uncertain/exploratory tokens
For O- (wrong):    penalty ∝ log(1+1/C) → punish confident mistakes
"""

import torch

EPS = 1e-8


def confidence_from_log_probs(log_probs: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """
    Per-token confidence from log-softmax output.

    C_i,t = -mean_{v ∈ top-k}( log π(v | ctx) )

    Args:
        log_probs: (B, T, V) log-softmax tensor (float32)
        top_k:     number of top tokens

    Returns:
        confidence: (B, T), C ≥ 0
    """
    k = min(top_k, log_probs.shape[-1])
    topk_lp, topk_ids = torch.topk(log_probs, k, dim=-1)   # (B, T, k)
    return -topk_lp.mean(dim=-1), topk_lp, topk_ids         # (B, T), (B,T,k), (B,T,k)


def compute_gtpo_conf_rewards(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.1,
    reward_threshold: float = 0.0,
):
    """
    GTPO-Conf: vectorised per-token reward shaping (no Python for-t loop).

    O+ bonus    ∝ log(1 + C)      high C (uncertain) → large bonus
    O- penalty  ∝ log(1 + 1/C)   small C (confident) → large penalty

    Args:
        rewards:         (B,) GRPO z-normalised advantages
        confidence:      (B, T) per-token confidence C, detached
        completion_mask: (B, T)
        alpha1, alpha2:  shaping hyperparams (ideally α₁+α₂=1)
        reward_threshold: O+/O- split (applied on z-norm GRPO advantages)

    Returns:
        adv_pos: (B, T) z-normalised O+ advantages (0 for O- sequences)
        adv_neg: (B, T) z-normalised O- advantages (0 for O+ sequences)
    """
    B, T = confidence.shape
    device = confidence.device

    is_pos = (rewards > reward_threshold)    # (B,)
    is_neg = ~is_pos

    C_comp = torch.log1p(confidence)                     # (B, T) log(1+C)
    C_inv  = torch.log1p(1.0 / (confidence + EPS))      # (B, T) log(1+1/C)

    mask_pos = completion_mask * is_pos.float().unsqueeze(1)   # (B, T)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)   # (B, T)

    # ── O+: vectorised over T ────────────────────────────────────────────────
    d          = mask_pos.sum(0)                                       # (T,)
    C_pos      = C_comp * mask_pos                                     # (B, T)
    sum_C_pos  = C_pos.sum(0).clamp(min=EPS)                          # (T,)
    bonus_pos  = C_pos / sum_C_pos.unsqueeze(0) * d.unsqueeze(0)      # (B, T)
    shaped_pos = alpha1 * mask_pos + alpha2 * bonus_pos               # (B, T)

    # ── O-: vectorised over T ────────────────────────────────────────────────
    h          = mask_neg.sum(0)                                       # (T,)
    I_neg      = C_inv * mask_neg                                      # (B, T)
    sum_I_neg  = I_neg.sum(0).clamp(min=EPS)                          # (T,)
    penalty_neg = I_neg / sum_I_neg.unsqueeze(0) * h.unsqueeze(0)     # (B, T)
    shaped_neg = -(alpha1 * mask_neg + alpha2 * penalty_neg)          # (B, T)

    # ── Z-normalise over active tokens ───────────────────────────────────────
    adv_pos = torch.zeros(B, T, device=device)
    pos_tok = shaped_pos[mask_pos.bool()]
    if pos_tok.numel() > 1:
        adv_pos[mask_pos.bool()] = (pos_tok - pos_tok.mean()) / (pos_tok.std() + EPS)
    elif pos_tok.numel() == 1:
        adv_pos = shaped_pos * mask_pos

    adv_neg = torch.zeros(B, T, device=device)
    neg_tok = shaped_neg[mask_neg.bool()]
    if neg_tok.numel() > 1:
        adv_neg[mask_neg.bool()] = (neg_tok - neg_tok.mean()) / (neg_tok.std() + EPS)
    elif neg_tok.numel() == 1:
        adv_neg = shaped_neg * mask_neg

    return adv_pos, adv_neg
