"""CPU unit tests for the exp_062 non-entropy shaping candidates.
Run: python -m pytest tests/test_candidates.py -q   (or: python tests/test_candidates.py)
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.novel_shaping import (
    flipped_advantages, apply_sign_gate, refdelta_advantages,
    position_discount,
)


def _active_mean(x, m):
    mm = m.bool()
    return x[mm].mean().item() if mm.any() else float("nan")


def test_flipped_nondegenerate_and_polarity():
    torch.manual_seed(0)
    G, T = 4, 5
    mask = torch.ones(G, T)
    seq_adv = torch.tensor([1.0, 0.5, -0.5, -1.0])          # O+={0,1}, O-={2,3}
    signal = torch.rand(G, T) + 0.5
    adv = flipped_advantages(seq_adv, signal, mask, 0.9, 0.1)
    assert adv.shape == (G, T)
    assert torch.isfinite(adv).all()
    # NOT degenerate (the B=1 bug produced a constant): real spread
    assert adv.std().item() > 1e-3
    # per-polarity z-norm => mean ~0 over each polarity's active tokens
    mp = mask.clone(); mp[2:] = 0
    mn = mask.clone(); mn[:2] = 0
    assert abs(_active_mean(adv, mp)) < 1e-4
    assert abs(_active_mean(adv, mn)) < 1e-4


def test_flipped_signal_direction():
    G, T = 4, 1
    mask = torch.ones(G, T)
    seq_adv = torch.tensor([1.0, 1.0, -1.0, -1.0])          # O+={0,1}, O-={2,3}
    signal = torch.tensor([[1.0], [4.0], [1.0], [4.0]])
    adv = flipped_advantages(seq_adv, signal, mask, 0.9, 0.1)
    # O+: LOWER signal (row0) -> larger 1/signal bonus -> higher advantage
    assert adv[0, 0] > adv[1, 0]
    # O-: HIGHER signal (row3) -> larger penalty -> more negative advantage
    assert adv[3, 0] < adv[2, 0]


def test_position_discount_shape():
    g = position_discount(2048, tau=1024.0, device=torch.device("cpu"))
    assert abs(g[0].item() - 1.0) < 1e-6
    assert abs(g[1024].item() - 0.5) < 1e-3
    assert (g[1:] < g[:-1]).all()                            # strictly decreasing
    # gentler than 1/sqrt(t): at t=1024 ours 0.5 >> 1/sqrt(1024)=0.031
    assert g[1024].item() > 0.3


def test_position_discount_softens_late_bonus():
    G, T = 4, 6
    mask = torch.ones(G, T)
    seq_adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    signal = torch.rand(G, T) + 0.5
    base = flipped_advantages(seq_adv, signal, mask, 0.9, 0.1)
    bm = position_discount(T, tau=2.0, device=mask.device).unsqueeze(0).expand(G, T)
    disc = flipped_advantages(seq_adv, signal, mask, 0.9, 0.1, bonus_mult=bm)
    assert torch.isfinite(disc).all()
    assert not torch.allclose(base, disc)                    # discount changes the result


def test_sign_gate_reverts_on_disagreement():
    seq_adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3)
    shaped = torch.tensor([[2.0, -0.7, 0.3], [-0.4, 0.9, -1.1]])
    out = apply_sign_gate(shaped, seq_adv, mask)
    exp = torch.tensor([[2.0, 1.0, 0.3],      # row0 base=+1: keep>0, revert<0 to +1
                        [-0.4, -1.0, -1.1]])  # row1 base=-1: keep<0, revert>0 to -1
    assert torch.allclose(out, exp, atol=1e-5)


def test_sign_gate_zero_variance_group():
    # seq_adv all 0 (zero-variance group) -> base sign 0 -> everything reverts to 0
    seq_adv = torch.zeros(2)
    mask = torch.ones(2, 3)
    shaped = torch.randn(2, 3)
    out = apply_sign_gate(shaped, seq_adv, mask)
    assert torch.allclose(out, torch.zeros(2, 3))


def test_refdelta_direction():
    G, T = 4, 1
    mask = torch.ones(G, T)
    seq_adv = torch.tensor([1.0, 1.0, -1.0, -1.0])          # O+={0,1}, O-={2,3}
    delta = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])    # row0/2 high deviation
    adv = refdelta_advantages(seq_adv, delta, mask, 0.9, 0.1)
    assert torch.isfinite(adv).all()
    # O+: higher delta (row0) -> higher advantage
    assert adv[0, 0] > adv[1, 0]
    # O-: higher delta (row2) -> more negative (penalize confident wrong deviation)
    assert adv[2, 0] < adv[3, 0]


def test_refdelta_coldstart_is_grpo_not_dead():
    # delta=0 (cold start: LoRA~0 => policy==ref) must reduce to plain GRPO, NOT 0
    G, T = 4, 4
    mask = torch.ones(G, T)
    seq_adv = torch.tensor([1.0, 0.5, -0.5, -1.0])
    delta = torch.zeros(G, T)
    adv = refdelta_advantages(seq_adv, delta, mask, 0.9, 0.1)
    base = seq_adv.unsqueeze(1) * mask
    assert torch.allclose(adv, base, atol=1e-6)      # == GRPO advantage
    assert adv.abs().sum().item() > 1e-3             # NOT a dead/zero signal


def test_refdelta_nondegenerate():
    torch.manual_seed(1)
    G, T = 4, 5
    mask = torch.ones(G, T)
    seq_adv = torch.tensor([1.0, 0.5, -0.5, -1.0])
    delta = torch.randn(G, T)
    adv = refdelta_advantages(seq_adv, delta, mask, 0.9, 0.1)
    assert torch.isfinite(adv).all()
    assert adv.std().item() > 1e-3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")
