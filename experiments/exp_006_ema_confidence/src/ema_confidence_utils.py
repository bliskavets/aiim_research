"""
ema_confidence_utils.py
-----------------------
EMA-smoothed confidence-based reward shaping for exp_006.

Key difference from exp_005:
- Instead of raw per-token confidence C_i,t, we use its EMA:
    EMA_i,t = λ · EMA_i,t-1 + (1-λ) · C_i,t    [λ=0.9]

- For GTPO-EMA: uses EMA_i,t at each token position
- For GRPO-S-EMA: uses the LAST EMA value per sequence EMA_i,T
  (captures the confidence trend at the END of generation)

Rationale:
- EMA smooths out spiky per-token confidence noise
- High EMA at the end → model was confidently generating → useful signal for GRPO-S
- For O+: reward high EMA (model was exploring/uncertain in good paths)
- For O-: penalize low EMA (model was confidently wrong)
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """
    C_i,t = -mean_{v ∈ top-k}( log π(v | context) )
    Args:
        logits: (B, T, V)
    Returns:
        confidence: (B, T), values ≥ 0
    """
    B, T, V = logits.shape
    k = min(top_k, V)
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
    topk_log_probs, _ = torch.topk(log_probs, k, dim=-1)  # (B, T, k)
    return -topk_log_probs.mean(dim=-1)                # (B, T)


def compute_ema(confidence: torch.Tensor, mask: torch.Tensor, lam: float = 0.9) -> torch.Tensor:
    """
    Compute EMA of confidence along the time dimension (left to right).

    EMA_i,0 = C_i,0
    EMA_i,t = λ · EMA_i,t-1 + (1-λ) · C_i,t    for valid tokens
    EMA_i,t = EMA_i,t-1                           for padding tokens (mask=0)

    Args:
        confidence: (B, T) per-token confidence
        mask:       (B, T) 1 for valid tokens, 0 for padding
        lam:        EMA decay factor (default 0.9)
    Returns:
        ema: (B, T) EMA-smoothed confidence
    """
    B, T = confidence.shape
    ema = torch.zeros_like(confidence)
    ema[:, 0] = confidence[:, 0] * mask[:, 0]

    for t in range(1, T):
        valid = mask[:, t]  # (B,)
        # Update EMA only where mask=1; keep previous value where mask=0
        new_ema = lam * ema[:, t-1] + (1 - lam) * confidence[:, t]
        ema[:, t] = torch.where(valid.bool(), new_ema, ema[:, t-1])

    return ema  # (B, T)


def get_last_ema(ema: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Get the EMA value at the last valid token for each sequence.
    Used for GRPO-S-EMA sequence-level signal.

    Args:
        ema:  (B, T)
        mask: (B, T)
    Returns:
        last_ema: (B,) EMA value at last valid position
    """
    B, T = ema.shape
    # Last valid index for each sequence
    seq_lengths = mask.sum(dim=1).long().clamp(min=1)  # (B,)
    last_idx = (seq_lengths - 1).clamp(min=0)          # (B,)
    return ema[torch.arange(B), last_idx]              # (B,)


def compress(c: torch.Tensor) -> torch.Tensor:
    """log(1+c) compression to reduce scale."""
    return torch.log1p(c)


# ─────────────────────────────────────────────────────────────────────────────
# GTPO-EMA: token-level EMA-smoothed confidence reward shaping
# ─────────────────────────────────────────────────────────────────────────────

def compute_gtpo_ema_rewards(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.1,
    lam: float = 0.9,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GTPO with EMA-smoothed per-token confidence bonus.

    For O+ tokens:
        r̃_i,t = α₁ + α₂ · (compress(EMA_i,t) / Σ_k compress(EMA_k,t)) · d_t

    For O- tokens:
        r̃_j,t = -(α₁ + α₂ · (compress(1/EMA_j,t) / Σ_k compress(1/EMA_k,t)) · h_t)

    Returns:
        adv_pos: (B, T)
        adv_neg: (B, T)
    """
    B, T = confidence.shape
    device = confidence.device

    # Compute EMA
    ema = compute_ema(confidence, completion_mask, lam=lam)   # (B, T)
    ema_comp     = compress(ema)                               # log(1+EMA)
    ema_inv_comp = compress(1.0 / (ema + EPS))                 # log(1+1/EMA)

    is_pos = (rewards > reward_threshold)
    is_neg = ~is_pos
    mask_pos = completion_mask * is_pos.float().unsqueeze(1)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)

    shaped_pos = torch.zeros(B, T, device=device)
    shaped_neg = torch.zeros(B, T, device=device)

    # O+ shaping
    for t in range(T):
        active = mask_pos[:, t]
        d_t = active.sum()
        if d_t == 0: continue
        E_t = ema_comp[:, t] * active
        sum_E = E_t.sum() + EPS
        bonus = (E_t / sum_E) * d_t
        shaped_pos[:, t] = alpha1 * active + alpha2 * bonus

    # O- shaping
    for t in range(T):
        active = mask_neg[:, t]
        h_t = active.sum()
        if h_t == 0: continue
        I_t = ema_inv_comp[:, t] * active
        sum_I = I_t.sum() + EPS
        penalty = (I_t / sum_I) * h_t
        shaped_neg[:, t] = -(alpha1 * active + alpha2 * penalty)

    # Normalize to advantages
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
# GRPO-S-EMA: sequence-level using LAST EMA value
# ─────────────────────────────────────────────────────────────────────────────

def compute_grpo_s_ema_rewards(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    beta1: float = 1.0,
    beta2: float = 0.1,
    lam: float = 0.9,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO-S with last-EMA sequence-level confidence bonus.

    Uses EMA_i,T (last valid EMA value) as the sequence-level signal.
    This captures the confidence TREND at the end of generation.

    For O+:
        r̂_i = β₁ + β₂ · (compress(EMA_i,T) / Σ_k compress(EMA_k,T)) · n
    For O-:
        r̂_j = -(β₁ + β₂ · (compress(1/EMA_j,T) / Σ_k compress(1/EMA_k,T)) · m)

    Returns:
        shaped_rewards: (B,)
        last_ema:       (B,) last EMA value per sequence (for logging)
    """
    device = confidence.device
    B = rewards.shape[0]

    ema      = compute_ema(confidence, completion_mask, lam=lam)   # (B, T)
    last_ema = get_last_ema(ema, completion_mask)                   # (B,)

    last_ema_comp     = compress(last_ema)              # log(1+EMA_T)
    last_ema_inv_comp = compress(1.0 / (last_ema + EPS))# log(1+1/EMA_T)

    is_pos = (rewards > reward_threshold)
    is_neg = ~is_pos
    n = is_pos.sum().item()
    m = is_neg.sum().item()

    shaped_rewards = torch.zeros(B, device=device)

    if n > 0:
        sum_comp_pos = last_ema_comp[is_pos].sum() + EPS
        bonus = beta2 * (last_ema_comp / sum_comp_pos) * n
        shaped_rewards[is_pos] = beta1 + bonus[is_pos]

    if m > 0:
        sum_inv_neg = last_ema_inv_comp[is_neg].sum() + EPS
        penalty = beta2 * (last_ema_inv_comp / sum_inv_neg) * m
        shaped_rewards[is_neg] = -(beta1 + penalty[is_neg])

    return shaped_rewards, last_ema
