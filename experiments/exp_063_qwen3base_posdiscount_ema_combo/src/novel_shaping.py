"""
novel_shaping.py — non-entropy per-token credit-assignment variants (exp_062).

All functions are PURE (tensors in, tensor out) so they can be unit-tested on CPU
with synthetic full-group inputs — this is the guard against the B=1 degeneracy
that broke the original gtpo_ema_flipped. They are always evaluated on the FULL
group (G = num_generations completions) inside _generate_and_score (see
novel_trainers.py), never recomputed in the B=1 compute_loss.

Signals (NO entropy):
  - confidence C_{i,t} = -mean_topk log p  (peakedness) — used raw or EMA-smoothed
  - reference-relative log-delta  δ = logπ_θ(o_t) - logπ_ref(o_t)

Candidates:
  flipped_advantages(...)    — the FIXED gtpo_ema_flipped core, generalized to an
                               arbitrary positive signal + optional per-token bonus
                               multiplier (=> raw_c, pos_discount reuse it).
  apply_sign_gate(...)       — keep shaped advantage only where its sign agrees with
                               the GRPO advantage; else revert to the GRPO scalar.
  refdelta_advantages(...)   — credit ∝ per-polarity z-scored deviation-from-prior.
  position_discount(...)     — gentle g(t)=tau/(tau+t) bonus multiplier.
"""
import torch
from .ema_flipped_utils import EPS, _znorm_over_active, compute_ema_vectorized


def position_discount(T, tau, device, dtype=torch.float32):
    """Gentle decay g(t)=tau/(tau+t), (T,). Much softer than 1/sqrt(t):
    at t=tau -> 0.5; 1/sqrt(t) at t=tau(=1024) -> 0.031."""
    t = torch.arange(T, device=device, dtype=dtype)
    return tau / (tau + t)


def flipped_advantages(seq_adv, signal, mask, alpha1=0.9, alpha2=0.1,
                       reward_threshold=0.0, bonus_mult=None):
    """Generalized FIXED gtpo_ema_flipped core on the FULL group.
    seq_adv (G,), signal (G,T) positive, mask (G,T). O+ (seq_adv>thr) weights tokens
    by 1/signal (reward low-signal/exploratory), O- weights by signal (penalize
    high-signal/decisive mistakes); per-position group normalization + per-polarity
    z-norm. bonus_mult (G,T) scales ONLY the alpha2 bonus (e.g. position discount)."""
    B, T = signal.shape
    device = signal.device
    is_pos = seq_adv > reward_threshold
    mask_pos = mask * is_pos.float().unsqueeze(1)
    mask_neg = mask * (~is_pos).float().unsqueeze(1)
    if bonus_mult is None:
        bonus_mult = torch.ones_like(signal)
    sig_inv = 1.0 / (signal + EPS)
    shaped_pos = torch.zeros(B, T, device=device)
    shaped_neg = torch.zeros(B, T, device=device)
    for t in range(T):
        ap = mask_pos[:, t]; d = ap.sum()
        if d.item() > 0:
            w = sig_inv[:, t] * ap; sw = w.sum() + EPS
            bonus = (w / sw) * d
            shaped_pos[:, t] = (alpha1 + alpha2 * bonus_mult[:, t] * bonus) * ap
        an = mask_neg[:, t]; h = an.sum()
        if h.item() > 0:
            w = signal[:, t] * an; sw = w.sum() + EPS
            pen = (w / sw) * h
            shaped_neg[:, t] = -(alpha1 + alpha2 * bonus_mult[:, t] * pen) * an
    return _znorm_over_active(shaped_pos, mask_pos) + _znorm_over_active(shaped_neg, mask_neg)


def apply_sign_gate(shaped_adv, seq_adv, mask):
    """Keep shaped_adv only where sign(shaped) == sign(GRPO advantage); elsewhere
    revert to the GRPO scalar (so shaping never fights the reward direction).
    seq_adv (G,), shaped_adv/mask (G,T)."""
    base = seq_adv.unsqueeze(1) * mask
    agree = torch.sign(shaped_adv) == torch.sign(base)
    return torch.where(agree, shaped_adv, base) * mask


def refdelta_advantages(seq_adv, delta, mask, alpha1=0.9, alpha2=0.1, reward_threshold=0.0):
    """Credit proportional to deviation-from-prior, ADDED on top of the GRPO
    advantage. delta (G,T) = logπ_θ - logπ_ref.

        Ã_{i,t} = A_grpo_i + α₂·( z(δ|O+)·1[O+]  −  z(δ|O−)·1[O−] )

    So a high-δ token (policy confidently deviated from the base) is reinforced on
    CORRECT sequences and penalized harder on WRONG ones. Crucially this is
    additive on the GRPO scalar, so at cold start (δ=0, LoRA≈0) it reduces to plain
    GRPO and still learns — NO z-norm-of-constant dead start (the failure mode that
    a pure per-polarity z-norm would hit). alpha1 is unused (kept for signature
    symmetry)."""
    is_pos = seq_adv > reward_threshold
    mask_pos = mask * is_pos.float().unsqueeze(1)
    mask_neg = mask * (~is_pos).float().unsqueeze(1)
    zp = _znorm_over_active(delta, mask_pos)
    zn = _znorm_over_active(delta, mask_neg)
    base = seq_adv.unsqueeze(1) * mask
    return base + alpha2 * (zp * mask_pos - zn * mask_neg)


@torch.no_grad()
def token_logprobs_chunked(model, input_ids, attention_mask, logits_to_keep, target_ids,
                           pass_logits_to_keep=False, micro_bs=1):
    """Per-token logprob of the realized `target_ids` (G, logits_to_keep), chunked
    over the batch dim to bound memory (same pattern as confidence_from_model_chunked)."""
    B = input_ids.size(0)
    outs = []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = target_ids[s:e].unsqueeze(-1)
        outs.append(lp.gather(-1, tgt).squeeze(-1))
        del logits, lp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)
