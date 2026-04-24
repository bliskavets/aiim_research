"""
ema_proof_utils.py
------------------
Pure-proof implementation of GTPO-EMA reward shaping per
experiments/proof/GTPO-EMA-full.txt (Definitions 1.1–1.5).

Differences from exp_005 GTPOConf and exp_010 GTPO-EMA v2:
  - Uses EMA of top-k confidence, not raw confidence (matches Def 1.2).
  - Per-token bonus is (EMA_{i,t} / Σ_{k∈O⁺_t} EMA_{k,t}) · d_t,
    NOT z-normalized and NOT log-compressed. This keeps the
    conservation property: Σ_{i∈O⁺_t} r̃⁺_{i,t} = (α₁+α₂)·d_t
    (Prop 2.3 in the proof), so with α₁+α₂=1 the total reward mass
    at each timestep equals d_t.
  - For O-, penalty uses raw 1/(EMA+ε) (Def 1.4), not log(1+1/C).
  - Final z-norm is applied separately to R̃⁺ and R̃⁻ per Def 1.5,
    on the flattened set of active-token values (not per-seq).
  - d_t / h_t = |O⁺_t| / |O⁻_t| are the number of *active* sequences
    at position t (those with |o_i| ≥ t), per Def 1.3. Tokens past the
    end of a sequence contribute r̃ = 0 and are skipped in the mean/std.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """
    C_{i,t} = -mean_{v ∈ top-k}( log π_θ(v | context_{i,t}) )   [Def 1.1]
    Returns (B, T) ≥ 0.
    """
    B, T, V = logits.shape
    k = min(top_k, V)
    log_probs = F.log_softmax(logits, dim=-1)
    topk_log_probs, _ = torch.topk(log_probs, k, dim=-1)
    return -topk_log_probs.mean(dim=-1)


def compute_ema_vectorized(
    confidence: torch.Tensor, mask: torch.Tensor, lam: float = 0.9
) -> torch.Tensor:
    """
    EMA_{i,0} = C_{i,0}
    EMA_{i,t} = λ·EMA_{i,t-1} + (1-λ)·C_{i,t}   for valid tokens       [Def 1.2]
              = EMA_{i,t-1}                     past end-of-sequence

    Args:
        confidence: (B, T)
        mask:       (B, T) 1 for valid tokens
        lam:        λ ∈ (0, 1); paper default 0.9
    Returns:
        ema: (B, T)
    """
    B, T = confidence.shape
    ema = torch.zeros_like(confidence)
    ema[:, 0] = confidence[:, 0] * mask[:, 0]
    for t in range(1, T):
        valid = mask[:, t].bool()
        new_ema = lam * ema[:, t - 1] + (1.0 - lam) * confidence[:, t]
        ema[:, t] = torch.where(valid, new_ema, ema[:, t - 1])
    return ema


def compute_gtpo_ema_proof_advantages(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 0.9,
    alpha2: float = 0.1,
    lam: float = 0.9,
    reward_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Pure-proof GTPO-EMA shaped advantages.

    For o_i ∈ O+ at position t (with o_i still active, i.e. |o_i| ≥ t):
        r̃⁺_{i,t} = α₁ · r_i
                 + α₂ · (EMA_{i,t} / Σ_{k ∈ O⁺_t} EMA_{k,t}) · d_t            [Def 1.4, O+]
    For o_j ∈ O- at position t (with o_j still active):
        r̃⁻_{j,t} = α₁ · (-r_i baseline = -1)
                 + α₂ · (EMA⁻¹_{j,t} / Σ_{k ∈ O⁻_t} EMA⁻¹_{k,t}) · h_t · (-1) [Def 1.4, O-]
        EMA⁻¹ = 1/(EMA + ε).
    r̃ = 0 for tokens past sequence end (same as r̃ = 0 when not in O⁺_t/O⁻_t).

    Final advantages (Def 1.5):
        Ã⁺_{i,t} = (r̃⁺_{i,t} - mean(R̃⁺)) / std(R̃⁺)   over active O+ tokens
        Ã⁻_{j,t} = (r̃⁻_{j,t} - mean(R̃⁻)) / std(R̃⁻)   over active O- tokens

    Args:
        rewards:         (B,) sequence rewards
        confidence:      (B, T) raw top-k confidence (from confidence_from_logits)
        completion_mask: (B, T) 1 if valid token
        alpha1, alpha2:  (paper default 0.9 and 0.1; α₁+α₂=1 for Prop 2.3 conservation)
        lam:             EMA decay
        reward_threshold: O+/O- split; sequences with reward > threshold are in O+

    Returns:
        token_advantages: (B, T) — Ã⁺ + Ã⁻ combined (non-overlapping by is_pos / is_neg)
    """
    B, T = confidence.shape
    device = confidence.device

    # Use r_i = +1 / -1 as the "base reward" in the shaping formula, per proof Def 1.4.
    # The split uses the raw (continuous) reward against the threshold.
    is_pos = rewards > reward_threshold   # (B,)
    is_neg = ~is_pos

    ema = compute_ema_vectorized(confidence, completion_mask, lam=lam)   # (B, T)

    mask_pos = completion_mask * is_pos.float().unsqueeze(1)             # (B, T), 1 iff i∈O⁺ and t<|o_i|
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)

    shaped_pos = torch.zeros(B, T, device=device)
    shaped_neg = torch.zeros(B, T, device=device)

    ema_inv = 1.0 / (ema + EPS)                                          # for O- weighting

    # ── O+ shaping: per-timestep bonus with Σ-conservation ──────────────────
    for t in range(T):
        active = mask_pos[:, t]
        d_t = active.sum()
        if d_t.item() == 0:
            continue
        ema_t = ema[:, t] * active
        sum_ema = ema_t.sum() + EPS
        bonus_t = (ema_t / sum_ema) * d_t                                # Σ_{i∈O⁺_t} bonus = d_t
        shaped_pos[:, t] = (alpha1 * 1.0 + alpha2 * bonus_t) * active    # 0 for inactive

    # ── O- shaping: per-timestep penalty, total = -h_t at step t ────────────
    for t in range(T):
        active = mask_neg[:, t]
        h_t = active.sum()
        if h_t.item() == 0:
            continue
        w_t = ema_inv[:, t] * active
        sum_w = w_t.sum() + EPS
        penalty_t = (w_t / sum_w) * h_t
        shaped_neg[:, t] = -(alpha1 * 1.0 + alpha2 * penalty_t) * active  # negative contribution

    # ── Final advantage via z-normalization on each group (Def 1.5) ─────────
    adv_pos = _znorm_over_active(shaped_pos, mask_pos)
    adv_neg = _znorm_over_active(shaped_neg, mask_neg)

    # Non-overlapping by construction: at most one of mask_pos[i], mask_neg[i] is 1
    return adv_pos + adv_neg


def _znorm_over_active(shaped: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Z-normalize over all values of `shaped` where `active_mask` is 1; zero elsewhere."""
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
