"""
roadmap_shaping.py — pure functions for the exp_071–075 roadmap setups
(see analysis/exp055-070_deep_analysis.md §5).

All functions are PURE (tensors in, tensors out), evaluated on the FULL group in
_generate_and_score (the FIXED pattern), and CPU-unit-tested.

Setups:
  071 zero-variance gate   — group_has_signal(); skip shaping when std(R)=0
  072 branching signal     — head_branching_* : h = H(renormalized top-k head)/log k ∈ [0,1]
                             branch_advantages: O+ bonus ∝ h, O− penalty ∝ (1−h)
                             (bounded by construction — no reciprocal, no ε-blowup)
  073 bonus budget         — budget=True: per-rollout Σ_t bonus is length-invariant
                             (equal to the polarity's mean active length)
  074 surprisal credit     — surprisal_advantages: additive ±α₂·z_polarity(−log p(o_t))
  075 final combo          — gate + branch signal + budget (composed in the trainer)
"""
import math
import torch
from .ema_flipped_utils import EPS, _znorm_over_active


def group_has_signal(seq_adv, eps=1e-8):
    """True iff the group carries terminal signal. TRL sets all advantages to 0
    when the group's rewards have zero variance — in that case any polarity split
    is pure noise (exp analysis §3) and shaping must be skipped."""
    return bool(seq_adv.abs().max().item() > eps)


def head_branching_from_sorted(sorted_logprobs, k=None):
    """h ∈ [0,1]: normalized entropy of the RENORMALIZED top-k head.
    sorted_logprobs: (..., N) top-N logprobs sorted desc (N ≥ k ≥ 2).
    p̃ = softmax over the head → H = −Σ p̃ log p̃ → h = H / log k.
    Peaked head → h≈0; uniform head → h=1. Bounded both sides by construction."""
    lp = sorted_logprobs.float()
    if k is None:
        k = lp.shape[-1]
    assert k >= 2, "head_branching requires k >= 2 (log k = 0 at k=1)"
    lp = lp[..., :k]
    p = torch.softmax(lp, dim=-1)                       # renormalized head
    H = -(p * torch.log(p + EPS)).sum(dim=-1)
    return (H / math.log(k)).clamp(0.0, 1.0)


@torch.no_grad()
def head_branching_from_model_chunked(model, input_ids, attention_mask, logits_to_keep,
                                      top_k=5, pass_logits_to_keep=False, micro_bs=1,
                                      seq_chunk=512):
    """(G, T) branching h from a memory-safe forward (micro-batch over G, chunk over T
    for the 152k-vocab log_softmax — same pattern as confidence_from_model_chunked)."""
    B = input_ids.size(0)
    outs = []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        b, T, V = logits.shape
        h = torch.empty(b, T, device=logits.device, dtype=torch.float32)
        for i in range(0, T, seq_chunk):
            lp = torch.log_softmax(logits[:, i:i + seq_chunk, :].float(), dim=-1)
            top_lp = lp.topk(min(top_k, V), dim=-1).values
            h[:, i:i + seq_chunk] = head_branching_from_sorted(top_lp)
            del lp, top_lp
        outs.append(h)
        del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)


def _polarity_shaped(seq_adv, w_pos, w_neg, mask, alpha1=0.9, alpha2=0.1,
                     reward_threshold=0.0, bonus_mult=None, budget=False,
                     return_parts=False):
    """Generic engine for the flipped-family shaping.
    w_pos/w_neg: (G,T) NON-NEGATIVE per-token weights for the O+ bonus and O− penalty.
    Per position t: bonus = (w/Σw over active)·d_t  (group-relative, mean 1 over active).
    bonus_mult (G,T) multiplies ONLY the α₂ part (position discount).
    budget=True: rescale each rollout's bonus/penalty row so Σ_t equals the
    polarity's MEAN active length — total harvestable shaped credit becomes
    length-invariant (a long rollout no longer collects more bonus than a short one)."""
    B, T = mask.shape
    device = mask.device
    is_pos = seq_adv > reward_threshold
    mask_pos = mask * is_pos.float().unsqueeze(1)
    mask_neg = mask * (~is_pos).float().unsqueeze(1)
    if bonus_mult is None:
        bonus_mult = torch.ones_like(mask, dtype=torch.float32)

    def per_position(w, m):
        out = torch.zeros(B, T, device=device)
        for t in range(T):
            a = m[:, t]; d = a.sum()
            if d.item() > 0:
                wt = w[:, t] * a
                out[:, t] = (wt / (wt.sum() + EPS)) * d
        return out

    bonus = per_position(w_pos, mask_pos)
    pen = per_position(w_neg, mask_neg)

    if budget:
        def rescale(rows, m):
            act = m.sum(dim=1)                                  # (G,) active len per row
            tgt = act[act > 0].mean() if (act > 0).any() else act.new_tensor(0.0)
            s = rows.sum(dim=1)                                 # (G,) current row sums
            scale = torch.where(s > EPS, tgt / (s + EPS), torch.ones_like(s))
            return rows * scale.unsqueeze(1)
        bonus = rescale(bonus, mask_pos)
        pen = rescale(pen, mask_neg)

    shaped_pos = (alpha1 + alpha2 * bonus_mult * bonus) * mask_pos
    shaped_neg = -(alpha1 + alpha2 * bonus_mult * pen) * mask_neg
    result = _znorm_over_active(shaped_pos, mask_pos) + _znorm_over_active(shaped_neg, mask_neg)
    if return_parts:
        return result, bonus, pen
    return result


def branch_advantages(seq_adv, h_ema, mask, alpha1=0.9, alpha2=0.1,
                      reward_threshold=0.0, bonus_mult=None, budget=False,
                      return_parts=False):
    """exp_072/075 core: O+ bonus ∝ h (reward branch points on correct rollouts),
    O− penalty ∝ (1−h) (blame peaked tokens on wrong rollouts). h ∈ [0,1] → both
    weights bounded; no reciprocal anywhere."""
    return _polarity_shaped(seq_adv, h_ema, 1.0 - h_ema, mask, alpha1, alpha2,
                            reward_threshold, bonus_mult, budget, return_parts)


def flipped_budget_advantages(seq_adv, signal, mask, alpha1=0.9, alpha2=0.1,
                              reward_threshold=0.0, return_parts=False):
    """exp_073: the CURRENT best signal/weights (O+ ∝ 1/EMA(C), O− ∝ EMA(C)) but with
    the length-invariant budget instead of the position discount."""
    return _polarity_shaped(seq_adv, 1.0 / (signal + EPS), signal, mask, alpha1, alpha2,
                            reward_threshold, bonus_mult=None, budget=True,
                            return_parts=return_parts)


def surprisal_advantages(seq_adv, s, mask, alpha2=0.1, reward_threshold=0.0,
                         bonus_mult=None):
    """exp_074: additive surprisal credit on top of the GRPO scalar.
        Ã_{i,t} = A_i + α₂·g(t)·z_polarity(s_{i,t})       s = −log p(o_t)
    O+: surprising tokens (high s) get extra credit; O−: confident tokens (low s →
    negative z) get extra punishment, exploratory ones are forgiven. Additive on the
    GRPO base → no cold-start dead signal (same trick as refdelta_advantages)."""
    is_pos = seq_adv > reward_threshold
    mask_pos = mask * is_pos.float().unsqueeze(1)
    mask_neg = mask * (~is_pos).float().unsqueeze(1)
    if bonus_mult is None:
        bonus_mult = torch.ones_like(mask, dtype=torch.float32)
    zp = _znorm_over_active(s, mask_pos)
    zn = _znorm_over_active(s, mask_neg)
    base = seq_adv.unsqueeze(1) * mask
    return base + alpha2 * bonus_mult * (zp * mask_pos + zn * mask_neg)
