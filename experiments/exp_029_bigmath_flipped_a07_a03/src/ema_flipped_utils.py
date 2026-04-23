"""
ema_flipped_utils.py  (exp_028 — Big-Math int-2000 variant)
-----------------------------------------------------------
Same reward-shaping logic as exp_026/src/ema_flipped_utils.py — the pure-
proof GTPO-EMA with O+/O- signal roles swapped — but the O+/O- partition
is now driven by an external boolean mask (from reward_cache) instead of
by seq_advantages > threshold.

Reason: on Big-Math the raw answer_exact reward is a much cleaner
"did-the-model-get-it-right" signal than the z-scored group advantage,
because an entire group can still contain a mix of format-perfect vs
wrong-number-extracted sequences.

O+ at step t:
    bonus_{i,t} = (1/EMA(C)_{i,t} / Σ_{k∈O⁺_t} 1/EMA(C)_{k,t}) · d_t
O- at step t:
    penalty_{j,t} = (EMA(C)_{j,t} / Σ_{k∈O⁻_t} EMA(C)_{k,t}) · h_t

Conservation (Σ r̃⁺ = d_t, Σ r̃⁻ = -h_t) and per-group z-norm unchanged.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    B, T, V = logits.shape
    k = min(top_k, V)
    log_probs = F.log_softmax(logits, dim=-1)
    topk_log_probs, _ = torch.topk(log_probs, k, dim=-1)
    return -topk_log_probs.mean(dim=-1)


def compute_ema_vectorized(
    confidence: torch.Tensor, mask: torch.Tensor, lam: float = 0.9
) -> torch.Tensor:
    B, T = confidence.shape
    ema = torch.zeros_like(confidence)
    ema[:, 0] = confidence[:, 0] * mask[:, 0]
    for t in range(1, T):
        valid = mask[:, t].bool()
        new_ema = lam * ema[:, t - 1] + (1.0 - lam) * confidence[:, t]
        ema[:, t] = torch.where(valid, new_ema, ema[:, t - 1])
    return ema


def compute_gtpo_ema_flipped_advantages(
    is_pos: torch.Tensor,          # (B,) bool — O+ membership from reward_cache
    confidence: torch.Tensor,       # (B, T)
    completion_mask: torch.Tensor,  # (B, T)
    alpha1: float = 0.9,
    alpha2: float = 0.1,
    lam: float = 0.9,
) -> torch.Tensor:
    """
    Pure-proof GTPO-EMA with flipped O+/O- signals, O+ set determined by
    an explicit boolean mask `is_pos` (from reward_cache).
    """
    B, T = confidence.shape
    device = confidence.device

    is_pos = is_pos.to(device=device)
    is_neg = ~is_pos

    ema     = compute_ema_vectorized(confidence, completion_mask, lam=lam)
    ema_inv = 1.0 / (ema + EPS)

    mask_pos = completion_mask * is_pos.float().unsqueeze(1)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)

    shaped_pos = torch.zeros(B, T, device=device)
    shaped_neg = torch.zeros(B, T, device=device)

    # O+: weight = 1/EMA (flipped)
    for t in range(T):
        active = mask_pos[:, t]
        d_t = active.sum()
        if d_t.item() == 0:
            continue
        w_t = ema_inv[:, t] * active
        sum_w = w_t.sum() + EPS
        bonus_t = (w_t / sum_w) * d_t
        shaped_pos[:, t] = (alpha1 * 1.0 + alpha2 * bonus_t) * active

    # O-: weight = EMA (flipped)
    for t in range(T):
        active = mask_neg[:, t]
        h_t = active.sum()
        if h_t.item() == 0:
            continue
        w_t = ema[:, t] * active
        sum_w = w_t.sum() + EPS
        penalty_t = (w_t / sum_w) * h_t
        shaped_neg[:, t] = -(alpha1 * 1.0 + alpha2 * penalty_t) * active

    adv_pos = _znorm_over_active(shaped_pos, mask_pos)
    adv_neg = _znorm_over_active(shaped_neg, mask_neg)

    return adv_pos + adv_neg


def _znorm_over_active(shaped: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    m = active_mask.bool()
    if not m.any():
        return torch.zeros_like(shaped)
    vals = shaped[m]
    if vals.numel() == 1:
        return shaped * active_mask
    mean = vals.mean()
    std  = vals.std() + EPS
    out  = (shaped - mean) / std
    return out * active_mask
