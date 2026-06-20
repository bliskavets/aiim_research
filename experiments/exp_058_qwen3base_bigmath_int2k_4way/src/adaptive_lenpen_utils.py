"""
adaptive_lenpen_utils.py
------------------------
exp_058 — ADAPTIVE length penalty whose knee L is derived from each group's own
length distribution (no fixed L hyperparameter), per the operator's spec:

    L_min, L_max, L_mean = min/max/mean(len(rollout) for rollout in group)
    L = max((L_min + L_max) / 2, L_mean)

    length_penalty(len) =  0                       if len <= L
                           -0.5 * (len - L) / L    if L < len < 2L     (linear ramp)
                           -0.5                     if len >= 2L         (saturated)

This is a bounded penalty in [-0.5, 0]. We return its MAGNITUDE pen = -length_penalty
in [0, 0.5] and group-CENTER it (relative within the group of num_generations),
matching the working exp_058 plumbing where compute_loss does
`token_advantages -= pen_rel * completion_mask` (shorter-than-group rollouts get
boosted, longer ones penalised). The chosen application point is the shaped
advantage, group-relative (a reward-level add is ~no-op for gtpo_ema_flipped,
whose shaping uses only the sign of the group-relative reward).
"""
import torch


def adaptive_length_penalty(lengths, ng):
    """lengths: (B_gen,) float tensor of completion token counts.
    Returns (pen_rel (B_gen,), L_per (B_gen,)):
      pen_rel  — group-centered penalty magnitude (subtract from shaped advantage)
      L_per    — the adaptive knee L for each rollout's group (for logging)."""
    n = lengths.numel()
    if n % ng == 0 and n >= ng:
        Lg = lengths.view(-1, ng)
    else:                                    # safety: treat the whole batch as one group
        Lg = lengths.view(1, -1)
    Lmin = Lg.min(dim=1, keepdim=True).values
    Lmax = Lg.max(dim=1, keepdim=True).values
    Lmean = Lg.mean(dim=1, keepdim=True)
    L = torch.maximum((Lmin + Lmax) * 0.5, Lmean).clamp(min=1.0)      # (n_groups, 1)

    ratio = (Lg - L) / L
    pen = torch.zeros_like(Lg)
    mid = (Lg > L) & (Lg < 2.0 * L)
    high = Lg >= 2.0 * L
    pen = torch.where(mid, 0.5 * ratio, pen)                          # linear ramp 0 -> 0.5
    pen = torch.where(high, torch.full_like(Lg, 0.5), pen)            # saturate at 0.5

    pen_rel = pen - pen.mean(dim=1, keepdim=True)                     # group-center
    L_per = L.expand_as(Lg)
    return pen_rel.reshape(-1), L_per.reshape(-1)
