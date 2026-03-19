"""
confidence_utils.py
-------------------
Confidence-based reward shaping for GTPO-Conf and GRPO-S-Conf (exp_005).

Confidence metric (from "Deep Think with Confidence", Meta, arXiv:2508.15260):
    C_i,t = -mean_{v ∈ top-k}( log π(v | context_i,t) )

Interpretation:
    - Small C → model is focused/certain (probability peaks on few tokens)
    - Large C → model is uncertain (probability spread across top-k)

Key difference from entropy:
    - Entropy uses ALL vocab tokens: H = -Σ_v π(v) log π(v)
    - Confidence uses only top-k tokens: C = -mean_{top-k} log π(v)
    - Confidence is cheaper and more interpretable at inference time

Normalization: log(1 + C) then group-relative division.
This squashes large values while preserving relative ranking.

For O+ sequences: high C (uncertain) tokens get bonus → reward exploration in correct paths
For O- sequences: low C (confident) tokens get penalty → punish confident mistakes
  → penalty ∝ 1 / log(1 + C) (inverse: small C → large penalty)
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Core: compute confidence from logits
# ─────────────────────────────────────────────────────────────────────────────

def confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """
    Compute per-token confidence score C_i,t from raw logits.

    C_i,t = -mean_{v ∈ top-k}( log π(v | context) )
           = -mean_{v ∈ top-k}( log_softmax(logits)[v] )

    Args:
        logits: (B, T, V) float tensor
        top_k:  number of top tokens to consider (paper uses k=20)

    Returns:
        confidence: (B, T) float tensor, C ≥ 0
    """
    B, T, V = logits.shape
    k = min(top_k, V)

    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)

    # Get top-k log probs at each position
    topk_log_probs, _ = torch.topk(log_probs, k, dim=-1)  # (B, T, k)

    # C = -mean(top-k log probs)  [log probs are negative, so C > 0]
    confidence = -topk_log_probs.mean(dim=-1)           # (B, T)

    return confidence  # always ≥ 0 since log_probs ≤ 0


def compress_confidence(c: torch.Tensor) -> torch.Tensor:
    """
    Log-compress confidence values to reduce scale and squash outliers:
        c_compressed = log(1 + C)

    This maps C ∈ [0, ∞) → [0, ∞) monotonically but more slowly.
    Example: C=1 → 0.69, C=10 → 2.40, C=100 → 4.61
    """
    return torch.log1p(c)  # log(1 + c), numerically stable


# ─────────────────────────────────────────────────────────────────────────────
# GTPO-Conf: token-level confidence-weighted reward shaping
# ─────────────────────────────────────────────────────────────────────────────

def compute_gtpo_conf_rewards(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.1,
    top_k: int = 20,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GTPO-Conf: token-level reward shaping using confidence instead of entropy.

    For O+ tokens (correct sequences):
        r̃_i,t = α₁·1 + α₂·(C̃_i,t / Σ_k C̃_k,t) · d_t
        where C̃ = log(1 + C)  [compressed confidence]
        High confidence (small C, small C̃) → smaller bonus
        Low confidence (large C, large C̃) → larger bonus (reward exploration)

    For O- tokens (incorrect sequences):
        r̃_j,t = -(α₁·1 + α₂·(Ĩ_j,t / Σ_k Ĩ_k,t) · h_t)
        where Ĩ = log(1 + 1/(C+ε))  [inverse-then-compressed]
        High confidence (small C) → large 1/C → large penalty (punish confident mistakes)
        Low confidence (large C) → small 1/C → small penalty

    Args:
        rewards:         (B,) sequence rewards (continuous)
        confidence:      (B, T) per-token confidence from model logits
        completion_mask: (B, T) 1 for valid tokens
        alpha1, alpha2:  shaping hyperparams (α₁+α₂ ≈ 1)
        top_k:           kept for signature compat (used upstream in confidence_from_logits)
        reward_threshold: O+/O- split threshold

    Returns:
        adv_pos: (B, T) advantage for O+ tokens (0 for O- sequences)
        adv_neg: (B, T) advantage for O- tokens (0 for O+ sequences)
    """
    B, T = confidence.shape
    device = confidence.device

    is_pos = (rewards > reward_threshold)   # (B,)
    is_neg = ~is_pos

    # Compressed confidence for O+ and inverse-compressed for O-
    C_comp = compress_confidence(confidence)                     # (B, T) log(1+C)
    C_inv  = torch.log1p(1.0 / (confidence + EPS))              # (B, T) log(1+1/C)

    mask_pos = completion_mask * is_pos.float().unsqueeze(1)     # (B, T)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)     # (B, T)

    # ── O+ shaped rewards ────────────────────────────────────────────────────
    shaped_pos = torch.zeros(B, T, device=device)
    for t in range(T):
        active = mask_pos[:, t]            # (B,)
        d_t = active.sum()
        if d_t == 0:
            continue
        C_t = C_comp[:, t] * active        # zero out inactive
        sum_C_t = C_t.sum() + EPS
        bonus = (C_t / sum_C_t) * d_t     # group-relative, sums to d_t
        shaped_pos[:, t] = alpha1 * active + alpha2 * bonus

    # ── O- shaped rewards ────────────────────────────────────────────────────
    shaped_neg = torch.zeros(B, T, device=device)
    for t in range(T):
        active = mask_neg[:, t]
        h_t = active.sum()
        if h_t == 0:
            continue
        I_t = C_inv[:, t] * active         # inverse confidence, masked
        sum_I_t = I_t.sum() + EPS
        penalty = (I_t / sum_I_t) * h_t   # group-relative
        shaped_neg[:, t] = -(alpha1 * active + alpha2 * penalty)

    # ── Normalize to advantages ───────────────────────────────────────────────
    adv_pos = torch.zeros(B, T, device=device)
    adv_neg = torch.zeros(B, T, device=device)

    pos_tokens = shaped_pos[mask_pos.bool()]
    if pos_tokens.numel() > 1:
        adv_pos = (shaped_pos - pos_tokens.mean()) / (pos_tokens.std() + EPS) * mask_pos
    elif pos_tokens.numel() == 1:
        adv_pos = shaped_pos * mask_pos

    neg_tokens = shaped_neg[mask_neg.bool()]
    if neg_tokens.numel() > 1:
        adv_neg = (shaped_neg - neg_tokens.mean()) / (neg_tokens.std() + EPS) * mask_neg
    elif neg_tokens.numel() == 1:
        adv_neg = shaped_neg * mask_neg

    return adv_pos, adv_neg


# ─────────────────────────────────────────────────────────────────────────────
# GRPO-S-Conf: sequence-level confidence-weighted reward shaping
# ─────────────────────────────────────────────────────────────────────────────

def compute_grpo_s_conf_rewards(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    beta1: float = 1.0,
    beta2: float = 0.1,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO-S-Conf: sequence-level reward shaping using confidence.

    Sequence-level confidence:
        Ĉ_i = (1/|o_i|) Σ_t C_i,t   [mean confidence per sequence]

    For O+ sequences:
        r̂_i = β₁·1 + β₂·(log(1+Ĉ_i) / Σ_k log(1+Ĉ_k)) · n
        High mean confidence → smaller bonus (model was sure = less exploration value)
        Low mean confidence  → larger bonus  (model explored = reward it)

    For O- sequences:
        r̂_j = -(β₁·1 + β₂·(log(1+1/Ĉ_j) / Σ_k log(1+1/Ĉ_k)) · m)
        High mean confidence (small Ĉ) → large 1/Ĉ → large penalty
        Low mean confidence  (large Ĉ) → small 1/Ĉ → small penalty

    Args:
        rewards:         (B,) sequence rewards
        confidence:      (B, T) per-token confidence
        completion_mask: (B, T)
        beta1, beta2:    shaping hyperparams
        reward_threshold: O+/O- split threshold

    Returns:
        shaped_rewards:  (B,) shaped sequence rewards
        seq_avg_conf:    (B,) mean confidence per sequence (for logging)
    """
    device = confidence.device
    B = rewards.shape[0]

    is_pos = (rewards > reward_threshold)
    is_neg = ~is_pos

    n = is_pos.sum().item()
    m = is_neg.sum().item()

    # Sequence-level mean confidence Ĉ_i
    seq_lengths = completion_mask.sum(dim=1).clamp(min=1)
    C_avg = (confidence * completion_mask).sum(dim=1) / seq_lengths  # (B,)

    # Compressed variants
    C_avg_comp = compress_confidence(C_avg)               # log(1+Ĉ)
    C_avg_inv  = torch.log1p(1.0 / (C_avg + EPS))        # log(1+1/Ĉ)

    shaped_rewards = torch.zeros(B, device=device)

    # O+ reward shaping
    if n > 0:
        sum_comp_pos = C_avg_comp[is_pos].sum() + EPS
        bonus = beta2 * (C_avg_comp / sum_comp_pos) * n
        shaped_rewards[is_pos] = beta1 + bonus[is_pos]

    # O- reward shaping
    if m > 0:
        sum_inv_neg = C_avg_inv[is_neg].sum() + EPS
        penalty = beta2 * (C_avg_inv / sum_inv_neg) * m
        shaped_rewards[is_neg] = -(beta1 + penalty[is_neg])

    return shaped_rewards, C_avg
