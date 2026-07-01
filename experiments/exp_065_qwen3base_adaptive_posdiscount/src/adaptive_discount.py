"""
adaptive_discount.py — pure functions for the exp_065 adaptive pos_discount g_{i,t}
(a multiplier in (0,1] — or up to g_max>1 for PC2 — on the α₂ exploration bonus of
the FIXED λ=0.7 flipped shaping). Batch-relative stats (m,sd,C_ref,s_ref) are passed
in by the trainer (computed over valid tokens) so these stay pure/testable.
"""
import torch
EPS = 1e-8


def g_p1(T, tau, floor, device):
    """Position-only hyperbolic with floor: g = floor + (1-floor)*tau/(tau+t). (T,)"""
    t = torch.arange(T, device=device, dtype=torch.float32)
    return floor + (1.0 - floor) * tau / (tau + t)


def g_pc1(ema_C, m, sd, tau, floor):
    """Position × prefix-decisiveness: gentle positional decay * σ((EMA(C)-m)/sd).
    ema_C (G,T); m,sd batch stats of EMA(C). Returns (G,T) in (0,1]."""
    G, T = ema_C.shape
    pos = g_p1(T, tau, floor, ema_C.device).unsqueeze(0)
    return pos * torch.sigmoid((ema_C - m) / (sd + EPS))


def g_c1(surprisal, s_ref, g_min):
    """Surprisal weight: g = clip(s/s_ref, g_min, 1). s=-logp (G,T), s_ref batch mean."""
    return torch.clip(surprisal / (s_ref + EPS), g_min, 1.0)


def g_pc2(C, C_ref, tau, g_min, g_max):
    """Early-decisiveness (allows boost>1): g = clip((C/C_ref)*(tau/(tau+t)), g_min, g_max).
    C (G,T), C_ref batch mean of C."""
    G, T = C.shape
    pos = (tau / (tau + torch.arange(T, device=C.device, dtype=torch.float32))).unsqueeze(0)
    return torch.clip((C / (C_ref + EPS)) * pos, g_min, g_max)
