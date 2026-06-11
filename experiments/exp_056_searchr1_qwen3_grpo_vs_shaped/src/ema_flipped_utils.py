"""
ema_flipped_utils.py
--------------------
Variant C of the pure-proof GTPO-EMA: swap the EMA-vs-inverse-EMA roles
between O+ and O-.

Background. The top-k confidence metric C = -mean_{v∈top-k} log π(v) used in
exp_005/006/010/025 increases with peakedness, NOT with entropy. A one-hot-ish
distribution gives a LARGE C (top-2..k log-probs are very negative, pulling
the mean down); a flat-uniform distribution gives a smaller C. See
tests/test_ema_flipped_utils.py::test_confidence_peaked_gt_flat.

The Def 1.4 formula in exp_025 is therefore rewarding "peaked/decisive"
tokens in O+ (bonus ∝ EMA(C), larger for peaked) and penalizing "flat"
tokens in O- (penalty ∝ 1/EMA(C), larger for flat). This is opposite to
the docstring intuition that calls C "uncertainty".

This file keeps the entire skeleton of exp_025 — per-timestep active
accounting (d_t / h_t), Σ-conservation (α₁+α₂=1), separate z-norm over
O+/O- active tokens — and SWAPS the two signals between groups:

    O+: bonus_{i,t}   = (1/EMA(C)_{i,t} / Σ_{k∈O⁺_t} 1/EMA(C)_{k,t}) · d_t
    O-: penalty_{j,t} = (EMA(C)_{j,t}   / Σ_{k∈O⁻_t} EMA(C)_{k,t})   · h_t

Interpretation after swap:
  - High EMA(C) (peaked/decisive) in O+ → small 1/EMA(C) → small bonus
  - Low  EMA(C) (flat/hesitant) in O+  → large 1/EMA(C) → large bonus
      → reward exploration on correct reasoning paths
  - High EMA(C) (peaked/decisive) in O- → large penalty
      → punish confident mistakes
  - Low  EMA(C) (flat/hesitant) in O-   → small penalty

Conservation and mass invariants (Prop 2.3) are preserved because they
only require the weights to be positive and the normalization (Lemma 2.2);
they hold for any positive signal, including the swapped one.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def confidence_from_logits(logits: torch.Tensor, top_k: int = 20) -> torch.Tensor:
    """C_{i,t} = -mean_{v ∈ top-k}(log π_θ(v | ctx))  — (B, T), ≥ 0."""
    B, T, V = logits.shape
    k = min(top_k, V)
    log_probs = F.log_softmax(logits, dim=-1)
    topk_log_probs, _ = torch.topk(log_probs, k, dim=-1)
    return -topk_log_probs.mean(dim=-1)


@torch.no_grad()
def confidence_from_model_chunked(model, input_ids, attention_mask, logits_to_keep,
                                  top_k: int = 20, pass_logits_to_keep: bool = False,
                                  micro_bs: int = 2) -> torch.Tensor:
    """Memory-safe confidence: run ``model`` forward in micro-batches over the
    batch dim so the full (B, L, V) fp32 logits tensor over Qwen3's ~152k vocab
    is never materialized at once (it OOMs the backward on long Search-R1
    rollouts). Mathematically identical to one forward + confidence_from_logits;
    only the peak memory differs. Returns (B, T)."""
    B = input_ids.size(0)
    chunks = []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        chunks.append(confidence_from_logits(logits, top_k=top_k))
        del logits
    return torch.cat(chunks, dim=0)


def compute_ema_vectorized(
    confidence: torch.Tensor, mask: torch.Tensor, lam: float = 0.9
) -> torch.Tensor:
    """
    EMA_{i,0} = C_{i,0}
    EMA_{i,t} = λ·EMA_{i,t-1} + (1-λ)·C_{i,t}   for valid tokens
              = EMA_{i,t-1}                     past end-of-sequence
    """
    B, T = confidence.shape
    ema = torch.zeros_like(confidence)
    ema[:, 0] = confidence[:, 0] * mask[:, 0]
    for t in range(1, T):
        valid = mask[:, t].bool()
        new_ema = lam * ema[:, t - 1] + (1.0 - lam) * confidence[:, t]
        ema[:, t] = torch.where(valid, new_ema, ema[:, t - 1])
    return ema


def compute_gtpo_ema_flipped_advantages(
    rewards: torch.Tensor,
    confidence: torch.Tensor,
    completion_mask: torch.Tensor,
    alpha1: float = 0.9,
    alpha2: float = 0.1,
    lam: float = 0.9,
    reward_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Pure-proof GTPO-EMA with the signal roles swapped between O+ and O-.

    For o_i ∈ O+ at position t (|o_i| ≥ t):
        r̃⁺_{i,t} = α₁ · 1 + α₂ · (1/EMA_{i,t} / Σ_{k∈O⁺_t} 1/EMA_{k,t}) · d_t
    For o_j ∈ O- at position t (|o_j| ≥ t):
        r̃⁻_{j,t} = -α₁ + α₂ · (EMA_{j,t} / Σ_{k∈O⁻_t} EMA_{k,t}) · h_t · (-1)
    r̃ = 0 for tokens outside the active set.

    Final Ã⁺ and Ã⁻ are z-normalized separately over their active tokens
    (Def 1.5). Conservation holds: Σ_{i∈O⁺_t} r̃⁺ = d_t; Σ_{j∈O⁻_t} r̃⁻ = -h_t
    when α₁+α₂=1.
    """
    B, T = confidence.shape
    device = confidence.device

    is_pos = rewards > reward_threshold
    is_neg = ~is_pos

    ema     = compute_ema_vectorized(confidence, completion_mask, lam=lam)
    ema_inv = 1.0 / (ema + EPS)

    mask_pos = completion_mask * is_pos.float().unsqueeze(1)
    mask_neg = completion_mask * is_neg.float().unsqueeze(1)

    shaped_pos = torch.zeros(B, T, device=device)
    shaped_neg = torch.zeros(B, T, device=device)

    # ── O+ (FLIPPED: now uses 1/EMA as the weight) ────────────────────────
    for t in range(T):
        active = mask_pos[:, t]
        d_t = active.sum()
        if d_t.item() == 0:
            continue
        w_t = ema_inv[:, t] * active          # <-- swap: was ema[:, t]
        sum_w = w_t.sum() + EPS
        bonus_t = (w_t / sum_w) * d_t
        shaped_pos[:, t] = (alpha1 * 1.0 + alpha2 * bonus_t) * active

    # ── O- (FLIPPED: now uses raw EMA as the weight) ──────────────────────
    for t in range(T):
        active = mask_neg[:, t]
        h_t = active.sum()
        if h_t.item() == 0:
            continue
        w_t = ema[:, t] * active              # <-- swap: was ema_inv[:, t]
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
