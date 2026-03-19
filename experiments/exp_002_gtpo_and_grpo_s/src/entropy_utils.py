"""
entropy_utils.py
----------------
Shared utilities for entropy-weighted reward shaping (GTPO and GRPO-S).

References:
  - GTPO:   Eq. (3), (5), (6)  in the paper
  - GRPO-S: Eq. (8), (9)       in the paper
"""

import torch
import torch.nn.functional as F


EPS = 1e-8  # numerical stability constant (paper Remark 2.1)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy helpers
# ─────────────────────────────────────────────────────────────────────────────

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute token-level Shannon entropy from raw logits.
    H = -Σ_v softmax(logits) * log_softmax(logits)

    Args:
        logits: (B, T, V) float tensor

    Returns:
        entropies: (B, T) float tensor, H ≥ 0
    """
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
    probs = torch.exp(log_probs)                        # (B, T, V)
    return -(probs * log_probs).sum(dim=-1)             # (B, T)


def clip_entropies(entropies: torch.Tensor, eps_low: float = 0.2, eps_high: float = 0.28) -> torch.Tensor:
    """
    Clip entropy values to [eps_low, eps_high] to avoid extreme rewards.
    Paper experimental setup: eps_low=0.2, eps_high=0.28
    """
    return entropies.clamp(min=eps_low, max=eps_high)


# ─────────────────────────────────────────────────────────────────────────────
# GTPO: token-level reward shaping  (Eq. 3, 5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gtpo_rewards(
    rewards: torch.Tensor,
    entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.1,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GTPO token-level shaped advantages.

    Args:
        rewards:         (B,) sequence-level rewards (continuous, our multi-reward sum)
        entropies:       (B, T) per-token entropy from old policy, masked where completion_mask==0
        completion_mask: (B, T) 1 for valid completion tokens, 0 for padding
        alpha1, alpha2:  reward shaping hyperparameters (α₁ + α₂ ≈ 1)
        eps_low, eps_high: entropy clipping range
        reward_threshold: sequences with reward > threshold → O+, else O-

    Returns:
        advantages_pos: (B, T) advantage for O+ tokens (0 for O- sequences)
        advantages_neg: (B, T) advantage for O- tokens (0 for O+ sequences)

    Both tensors must be added: total_advantage = advantages_pos + advantages_neg
    """
    B, T = entropies.shape
    device = entropies.device

    # Split into O+ / O-
    is_pos = (rewards > reward_threshold)  # (B,)
    is_neg = ~is_pos                       # (B,)

    # Clip entropies for numerical stability
    H = clip_entropies(entropies, eps_low, eps_high)  # (B, T)

    # Mask: 1 where token exists AND sequence is in O+/O-
    mask_pos = completion_mask * is_pos.float().unsqueeze(1)  # (B, T)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)  # (B, T)

    # ── O+ reward shaping (Eq. 3) ──────────────────────────────────────────
    # r̃_i,t = α₁·r_i + α₂·(H_i,t / Σ_k H_k,t) · d_t
    # d_t = number of O+ sequences with valid token at position t
    # Σ_k H_k,t = sum of entropies of O+ sequences at position t

    shaped_pos = torch.zeros(B, T, device=device)
    for t in range(T):
        # Which O+ seqs are active at this timestep?
        active = mask_pos[:, t]  # (B,) binary
        d_t = active.sum()
        if d_t == 0:
            continue
        H_t = H[:, t] * active                        # zero out inactive seqs
        sum_H_t = H_t.sum() + EPS
        entropy_bonus = (H_t / sum_H_t) * d_t         # (B,) normalized bonus
        # r_i ≈ 1 for O+ (we normalize as if r_i=1 per paper; our rewards are scaled)
        shaped_pos[:, t] = alpha1 * active + alpha2 * entropy_bonus

    # ── O- reward shaping (Eq. 5) ──────────────────────────────────────────
    # r̃_j,t = α₁·(-1) + α₂·(1/H_j,t / Σ_k 1/H_{n+k,t}) · h_t · (-1)
    # h_t = number of O- sequences with valid token at position t

    shaped_neg = torch.zeros(B, T, device=device)
    for t in range(T):
        active = mask_neg[:, t]  # (B,) binary
        h_t = active.sum()
        if h_t == 0:
            continue
        inv_H_t = (1.0 / (H[:, t] + EPS)) * active   # (B,) inv entropy, masked
        sum_inv_H_t = inv_H_t.sum() + EPS
        inv_entropy_bonus = (inv_H_t / sum_inv_H_t) * h_t  # (B,)
        # Negative reward: penalize more for low entropy (confident) tokens
        shaped_neg[:, t] = -(alpha1 * active + alpha2 * inv_entropy_bonus)

    # ── Compute advantages with separate normalization (Eq. 6) ────────────
    # Ã+_i,t = (r̃+_i,t - mean(R̃+)) / std(R̃+)  over all O+ tokens in batch
    # Ã-_j,t = (r̃-_j,t - mean(R̃-)) / std(R̃-)  over all O- tokens in batch

    adv_pos = torch.zeros(B, T, device=device)
    adv_neg = torch.zeros(B, T, device=device)

    pos_tokens = shaped_pos[mask_pos.bool()]
    if pos_tokens.numel() > 1:
        mean_pos = pos_tokens.mean()
        std_pos  = pos_tokens.std().clamp(min=EPS)
        adv_pos  = (shaped_pos - mean_pos) / std_pos * mask_pos
    elif pos_tokens.numel() == 1:
        adv_pos = shaped_pos * mask_pos  # can't normalize single token

    neg_tokens = shaped_neg[mask_neg.bool()]
    if neg_tokens.numel() > 1:
        mean_neg = neg_tokens.mean()
        std_neg  = neg_tokens.std().clamp(min=EPS)
        adv_neg  = (shaped_neg - mean_neg) / std_neg * mask_neg
    elif neg_tokens.numel() == 1:
        adv_neg = shaped_neg * mask_neg

    return adv_pos, adv_neg


# ─────────────────────────────────────────────────────────────────────────────
# GRPO-S: sequence-level reward shaping  (Eq. 8, 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_grpo_s_rewards(
    rewards: torch.Tensor,
    entropies: torch.Tensor,
    completion_mask: torch.Tensor,
    beta1: float = 1.0,
    beta2: float = 0.1,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    reward_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO-S sequence-level shaped advantages.

    Args:
        rewards:         (B,) sequence-level rewards
        entropies:       (B, T) per-token entropy from old policy
        completion_mask: (B, T) 1 for valid tokens
        beta1, beta2:    reward shaping hyperparameters
        eps_low, eps_high: entropy clipping range
        reward_threshold: O+/O- split threshold

    Returns:
        shaped_rewards: (B,) shaped sequence rewards (mix of O+ and O- terms)
        seq_avg_entropy: (B,) average entropy per sequence (for logging)
    """
    device = entropies.device
    B = rewards.shape[0]

    is_pos = (rewards > reward_threshold)  # (B,)
    is_neg = ~is_pos

    # Clip entropies
    H = clip_entropies(entropies, eps_low, eps_high)  # (B, T)

    # Ĥ_k = (1/|o_k|) Σ_t H_k,t  (Eq. 8)
    seq_lengths = completion_mask.sum(dim=1).clamp(min=1)  # (B,)
    H_avg = (H * completion_mask).sum(dim=1) / seq_lengths  # (B,)

    n = is_pos.sum().item()  # number of O+ sequences
    m = is_neg.sum().item()  # number of O- sequences

    shaped_rewards = torch.zeros(B, device=device)

    # ── O+ reward shaping (Eq. 9 top) ──────────────────────────────────────
    # r̂_i = β₁·r_i + β₂·(Ĥ_i / Σ_k Ĥ_k) · n
    if n > 0:
        sum_H_pos = H_avg[is_pos].sum() + EPS
        bonus_pos = beta2 * (H_avg / sum_H_pos) * n  # (B,)
        shaped_rewards[is_pos] = beta1 + bonus_pos[is_pos]  # r_i=1 for O+

    # ── O- reward shaping (Eq. 9 bottom) ────────────────────────────────────
    # r̂_j = β₁·(-1) + β₂·(1/Ĥ_j / Σ_k 1/Ĥ_{n+k}) · m · (-1)
    if m > 0:
        inv_H_neg = 1.0 / (H_avg[is_neg] + EPS)
        sum_inv_H_neg = inv_H_neg.sum() + EPS
        inv_H_all = 1.0 / (H_avg + EPS)
        penalty_neg = beta2 * (inv_H_all / sum_inv_H_neg) * m  # (B,)
        shaped_rewards[is_neg] = -(beta1 + penalty_neg[is_neg])  # r_j=-1 for O-

    return shaped_rewards, H_avg
