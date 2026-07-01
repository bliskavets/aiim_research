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


def _subgroup_penalty(sub_lens):
    """Adaptive piecewise penalty MAGNITUDE in [0,0.5] for one subgroup's lengths,
    centered within the subgroup. L = max((Lmin+Lmax)/2, Lmean)."""
    Lmin = sub_lens.min(); Lmax = sub_lens.max(); Lmean = sub_lens.mean()
    L = torch.maximum((Lmin + Lmax) * 0.5, Lmean).clamp(min=1.0)
    ratio = (sub_lens - L) / L
    p = torch.zeros_like(sub_lens)
    mid = (sub_lens > L) & (sub_lens < 2.0 * L)
    high = sub_lens >= 2.0 * L
    p = torch.where(mid, 0.5 * ratio, p)
    p = torch.where(high, torch.full_like(sub_lens, 0.5), p)
    return p - p.mean(), L                                            # center within subgroup


def adaptive_length_penalty_polarity(lengths, advantages, ng):
    """Adaptive length penalty computed SEPARATELY within the O+ / O- subgroups of
    each group of ng rollouts (split by seq-advantage sign: O+ = adv>0 'correct',
    O- = adv<0 'incorrect'). Each polarity gets its OWN knee
    L_+/L_- = max((Lmin+Lmax)/2, Lmean) over that polarity's lengths, the piecewise
    penalty in [0,0.5], and centering WITHIN that polarity. adv==0 rollouts and
    singleton subgroups (after centering) get 0. Returns (pen_rel (B,), L_own (B,))
    where L_own carries each rollout's own-polarity knee (NaN where N/A, for logging).

    Group-normalised advantages sum to ~0 per group, so both polarities are almost
    always non-empty. Rationale: gtpo_ema_flipped already z-norms its confidence
    weighting per polarity, so a per-polarity length ranking is the consistent
    place to add "short beats long" — among the correct rollouts AND among the
    incorrect rollouts independently, rather than mixing the two length regimes."""
    n = lengths.numel()
    device = lengths.device
    pen_rel = torch.zeros(n, device=device)
    L_own = torch.full((n,), float("nan"), device=device)
    if not (n % ng == 0 and n >= ng):
        return pen_rel, L_own                          # safety: no-op
    L_idx = lengths.view(-1, ng)
    A_idx = advantages.view(-1, ng)
    for g in range(L_idx.shape[0]):
        lens_g, adv_g, base = L_idx[g], A_idx[g], g * ng
        for positive in (True, False):
            mask = (adv_g > 0) if positive else (adv_g < 0)
            if mask.any():
                p, L = _subgroup_penalty(lens_g[mask])
                idx = mask.nonzero(as_tuple=True)[0]
                pen_rel[base + idx] = p
                L_own[base + idx] = L
    return pen_rel, L_own


def _knee(ls):
    return torch.maximum((ls.min() + ls.max()) * 0.5, ls.mean()).clamp(min=1.0)


def _overlong_mag(ls, L):
    """Piecewise overlong penalty MAGNITUDE in [0,0.5] (Appendix D):
    0 if |o|<L ; 0.5·(|o|−L)/L if L≤|o|<2L ; 0.5 if |o|≥2L."""
    r = (ls - L) / L
    p = torch.zeros_like(ls)
    mid = (ls >= L) & (ls < 2.0 * L)
    high = ls >= 2.0 * L
    p = torch.where(mid, 0.5 * r, p)
    p = torch.where(high, torch.full_like(ls, 0.5), p)
    return p


def group_relative_overlong_punishment(lengths, correct, ng, gamma1=0.75):
    """Group Relative Overlong Punishment — Appendix D of arXiv:2508.04349.

    lengths (B,) float; correct (B,) bool (terminal-reward correctness). Per group
    of ng responses with n correct / m incorrect, frac = n/(n+m):
      - EASY   (frac >= gamma1, n>0): penalize CORRECT responses; knee L+ over the
        CORRECT lengths.
      - HARD   (frac <= 1-gamma1): no penalty (preserve ability to solve).
      - MEDIUM (otherwise): knee L- over ALL G lengths; if n>m penalize CORRECT,
        else penalize INCORRECT.
      knee L = max((min+max)/2, mean) over the relevant subset.
    Returns (pen (B,) >=0 magnitude = -R(i), regime (B,) int: 0 hard/none, 1 easy,
    2 medium). The penalty is the paper's reward term R(i) <= 0; the caller decides
    the injection point (we subtract it from the shaped advantage)."""
    n_tot = lengths.numel()
    pen = torch.zeros(n_tot, device=lengths.device)
    regime = torch.zeros(n_tot, dtype=torch.long, device=lengths.device)
    if not (n_tot % ng == 0 and n_tot >= ng):
        return pen, regime
    g2 = 1.0 - gamma1
    for gi in range(n_tot // ng):
        b = gi * ng
        ls = lengths[b:b + ng]
        cr = correct[b:b + ng].bool()
        n = int(cr.sum().item()); m = ng - n
        frac = n / ng
        if frac >= gamma1 and n > 0:                       # EASY -> penalize correct, L+ over correct
            regime[b:b + ng] = 1
            L = _knee(ls[cr])
            idx = cr.nonzero(as_tuple=True)[0]
            pen[b + idx] = _overlong_mag(ls[cr], L)
        elif frac <= g2:                                   # HARD -> no penalty
            regime[b:b + ng] = 0
        else:                                              # MEDIUM -> L- over all G
            regime[b:b + ng] = 2
            L = _knee(ls)
            sub = cr if n > m else (~cr)
            idx = sub.nonzero(as_tuple=True)[0]
            pen[b + idx] = _overlong_mag(ls[sub], L)
    return pen, regime
