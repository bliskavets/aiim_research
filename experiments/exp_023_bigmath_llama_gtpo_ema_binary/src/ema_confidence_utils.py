"""
ema_confidence_utils_v2.py
--------------------------
Fixed EMA-smoothed confidence reward shaping for exp_010.

Key fixes vs exp_006:
1. NO z-score normalization of shaped advantages — it was destroying the EMA signal.
   Instead: group-relative scaling (divide by group mean absolute value).
2. Work directly with raw rewards, not pre-normalized seq_advantages.
3. Token-loop replaced with vectorized EMA computation.
4. Cleaner separation of O+/O- shaping.

GTPO-EMA formula (per token t, sequence i):
    O+: token_adv_i,t = α₁ · (seq_reward_i - μ_group) / σ_group
                       + α₂ · normalize_within_group(EMA_i,t)
    O-: token_adv_i,t = α₁ · (seq_reward_i - μ_group) / σ_group
                       - α₂ · normalize_within_group(1 / EMA_i,t)

GRPO-S-EMA formula (sequence i):
    O+: seq_adv_i = β₁ · (seq_reward_i - μ_group) / σ_group
                  + β₂ · normalize_within_group(EMA_i,T)
    O-: seq_adv_i = β₁ · (seq_reward_i - μ_group) / σ_group
                  - β₂ · normalize_within_group(1 / EMA_i,T)
    → broadcast to all tokens in sequence
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────────────

def confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """
    C_i,t = -mean_{v ∈ top-k}(log π(v | context))
    Higher = less confident.
    Args:
        logits: (B, T, V)
    Returns:
        confidence: (B, T) >= 0
    """
    B, T, V = logits.shape
    k = min(top_k, V)
    log_probs = F.log_softmax(logits, dim=-1)
    topk_log_probs, _ = torch.topk(log_probs, k, dim=-1)
    return -topk_log_probs.mean(dim=-1)


def compute_ema_vectorized(confidence: torch.Tensor, mask: torch.Tensor, lam: float = 0.9) -> torch.Tensor:
    """
    Vectorized EMA along time dimension.
    EMA_i,0 = C_i,0
    EMA_i,t = λ·EMA_i,t-1 + (1-λ)·C_i,t  for valid tokens
            = EMA_i,t-1                    for padding

    Args:
        confidence: (B, T)
        mask:       (B, T) 1=valid, 0=padding
        lam:        decay factor
    Returns:
        ema: (B, T)
    """
    B, T = confidence.shape
    ema = torch.zeros_like(confidence)
    ema[:, 0] = confidence[:, 0] * mask[:, 0]
    for t in range(1, T):
        valid = mask[:, t].bool()
        new_ema = lam * ema[:, t-1] + (1.0 - lam) * confidence[:, t]
        ema[:, t] = torch.where(valid, new_ema, ema[:, t-1])
    return ema


def get_last_ema(ema: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Last valid EMA value per sequence. Returns (B,)."""
    lengths = mask.sum(dim=1).long().clamp(min=1)
    last_idx = (lengths - 1).clamp(min=0)
    return ema[torch.arange(ema.size(0), device=ema.device), last_idx]


def group_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize within-group (mean=0, std=1), safe when all values equal.
    x: (N,) values to normalize.
    Returns (N,) normalized.
    """
    if x.numel() <= 1:
        return torch.zeros_like(x)
    return (x - x.mean()) / (x.std() + EPS)


def compress(c: torch.Tensor) -> torch.Tensor:
    """log(1+c) compression."""
    return torch.log1p(c)


# ─────────────────────────────────────────────────────────────────────────────
# GTPO-EMA: token-level advantage
# ─────────────────────────────────────────────────────────────────────────────

def compute_gtpo_ema_advantages(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.1,
    lam: float = 0.9,
    reward_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Token-level advantages for GTPO-EMA.

    Base advantage (GRPO-style, per sequence):
        A_i = (r_i - μ) / σ   [group-relative]

    EMA bonus per token:
        For O+ seqs: bonus_i,t = normalize_within_group(compress(EMA_i,t))
        For O- seqs: bonus_i,t = normalize_within_group(compress(1/EMA_i,t))

    Final token advantage:
        adv_i,t = α₁ · A_i + α₂ · bonus_i,t  (O+)
        adv_i,t = α₁ · A_i - α₂ · bonus_i,t  (O-)

    Args:
        rewards:         (B,) raw sequence rewards
        confidence:      (B, T) per-token confidence
        completion_mask: (B, T)
        alpha1:          weight for base GRPO advantage
        alpha2:          weight for EMA confidence bonus
        lam:             EMA decay
        reward_threshold: O+/O- split

    Returns:
        token_advantages: (B, T)
    """
    B, T = confidence.shape
    device = confidence.device

    # Group-relative base advantage per sequence
    r_mean = rewards.mean()
    r_std  = rewards.std() + EPS
    base_adv = (rewards - r_mean) / r_std  # (B,)

    # EMA
    ema = compute_ema_vectorized(confidence, completion_mask, lam=lam)  # (B, T)

    is_pos = rewards > reward_threshold  # (B,)
    is_neg = ~is_pos

    token_advantages = torch.zeros(B, T, device=device)

    # Process each time step — normalize EMA within O+ / O- groups
    for t in range(T):
        valid = completion_mask[:, t].bool()
        if not valid.any():
            continue

        ema_t = ema[:, t]  # (B,)
        bonus = torch.zeros(B, device=device)

        # O+ bonus: high EMA → model was exploring → reward it
        pos_valid = is_pos & valid
        if pos_valid.any():
            ema_pos = compress(ema_t[pos_valid])
            norm_pos = group_normalize(ema_pos)
            bonus[pos_valid] = norm_pos

        # O- bonus: high 1/EMA → model was overconfident but wrong → penalize
        neg_valid = is_neg & valid
        if neg_valid.any():
            ema_neg = compress(1.0 / (ema_t[neg_valid] + EPS))
            norm_neg = group_normalize(ema_neg)
            bonus[neg_valid] = -norm_neg  # negative sign: penalize

        # Broadcast base_adv to tokens
        base_t = base_adv * valid.float()  # (B,) zeroed for padding
        token_advantages[:, t] = alpha1 * base_t + alpha2 * bonus * valid.float()

    return token_advantages


# ─────────────────────────────────────────────────────────────────────────────
# GRPO-S-EMA: sequence-level advantage → broadcast to tokens
# ─────────────────────────────────────────────────────────────────────────────

def compute_grpo_s_ema_advantages(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    beta1: float = 1.0,
    beta2: float = 0.1,
    lam: float = 0.9,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequence-level advantages for GRPO-S-EMA, broadcast to tokens.

    seq_adv_i = β₁ · (r_i - μ)/σ  ±  β₂ · normalize_within_group(EMA_signal_i)

    + for O+, - for O-. Signal = last EMA value EMA_i,T.

    Returns:
        token_advantages: (B, T) — seq advantage broadcast to each token
        last_ema:         (B,)   — last EMA per sequence (for logging)
    """
    B, T = confidence.shape
    device = confidence.device

    r_mean = rewards.mean()
    r_std  = rewards.std() + EPS
    base_adv = (rewards - r_mean) / r_std  # (B,)

    ema      = compute_ema_vectorized(confidence, completion_mask, lam=lam)
    last_ema = get_last_ema(ema, completion_mask)  # (B,)

    is_pos = rewards > reward_threshold
    is_neg = ~is_pos

    ema_signal = torch.zeros(B, device=device)

    if is_pos.any():
        lec_pos = compress(last_ema[is_pos])
        norm_pos = group_normalize(lec_pos)
        ema_signal[is_pos] = norm_pos

    if is_neg.any():
        lec_neg = compress(1.0 / (last_ema[is_neg] + EPS))
        norm_neg = group_normalize(lec_neg)
        ema_signal[is_neg] = -norm_neg

    seq_advantages = beta1 * base_adv + beta2 * ema_signal  # (B,)

    # Broadcast to tokens
    token_advantages = seq_advantages.unsqueeze(1) * completion_mask  # (B, T)

    return token_advantages, last_ema
