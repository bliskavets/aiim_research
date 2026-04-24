"""
Unit tests for ema_flipped_utils.py (pure-proof GTPO-EMA with O+/O- swap).
Focus on the swap-specific invariants: conservation still holds with
flipped weights, and the bonus ranking in O+ actually inverts vs exp_025
(small EMA → large bonus, instead of vice-versa).
"""
import math
import pytest
import torch

from src.ema_flipped_utils import (
    confidence_from_logits,
    compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages,
    EPS,
)


# ── Confidence (sanity) ─────────────────────────────────────────────────────

def test_confidence_nonnegative():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 100)
    c = confidence_from_logits(logits, top_k=20)
    assert c.shape == (3, 5) and (c >= 0).all()


def test_confidence_peaked_gt_flat():
    """Key motivating fact: peaked distribution gives LARGER C than flat."""
    logits_peaked = torch.zeros(1, 1, 100); logits_peaked[..., 0] = 10.0
    logits_flat   = torch.zeros(1, 1, 100)
    c_peaked = confidence_from_logits(logits_peaked, top_k=20)
    c_flat   = confidence_from_logits(logits_flat,   top_k=20)
    assert c_peaked.item() > c_flat.item()


# ── EMA (unchanged) ─────────────────────────────────────────────────────────

def test_ema_recurrence():
    C = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0]])
    ema = compute_ema_vectorized(C, mask, lam=0.9)
    assert math.isclose(ema[0, 1].item(), 1.1, rel_tol=1e-6)


def test_ema_past_end_is_frozen():
    C = torch.tensor([[1.0, 1.0, 999.0, 999.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    ema = compute_ema_vectorized(C, mask, lam=0.9)
    assert math.isclose(ema[0, 1].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(ema[0, 3].item(), 1.0, rel_tol=1e-6)


# ── Flipped conservation (O+ uses 1/EMA, O- uses EMA) ───────────────────────

def test_conservation_positive_flipped():
    """
    With α₁+α₂=1 the total shaped_pos over active O+ tokens at each t
    should equal d_t, regardless of whether we weigh by EMA or by 1/EMA.
    """
    torch.manual_seed(0)
    B, T = 4, 5
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    alpha1, alpha2 = 0.9, 0.1
    ema = compute_ema_vectorized(confidence, mask, lam=0.9)
    ema_inv = 1.0 / (ema + EPS)
    for t in range(T):
        active = mask[:, t]
        d_t = active.sum().item()
        w_t = ema_inv[:, t] * active
        sum_w = w_t.sum() + EPS
        bonus = (w_t / sum_w) * d_t
        shaped = (alpha1 + alpha2 * bonus) * active
        assert math.isclose(shaped.sum().item(), (alpha1 + alpha2) * d_t, rel_tol=1e-4)


def test_conservation_negative_flipped():
    """Σ shaped_neg over active O- tokens at each t should equal -h_t."""
    torch.manual_seed(1)
    B, T = 4, 5
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    alpha1, alpha2 = 0.9, 0.1
    ema = compute_ema_vectorized(confidence, mask, lam=0.9)
    for t in range(T):
        active = mask[:, t]
        h_t = active.sum().item()
        w_t = ema[:, t] * active
        sum_w = w_t.sum() + EPS
        penalty = (w_t / sum_w) * h_t
        shaped = -(alpha1 + alpha2 * penalty) * active
        assert math.isclose(shaped.sum().item(), -(alpha1 + alpha2) * h_t, rel_tol=1e-4)


# ── Flip test: bonus ranking in O+ inverts vs pure-proof ────────────────────

def test_flipped_bonus_ranking_inverts():
    """
    In pure-proof (exp_025) the token with the LARGEST EMA gets the LARGEST
    bonus. Here we expect the token with the SMALLEST EMA to get the
    LARGEST bonus in O+, because the weight is now 1/EMA.
    """
    # Construct two O+ sequences with constant but different EMA levels
    B, T = 2, 3
    confidence = torch.tensor([[1.0, 1.0, 1.0],   # seq 0: small C (low)
                               [9.0, 9.0, 9.0]])  # seq 1: big   C (high)
    mask = torch.ones(B, T)
    rewards = torch.tensor([+1.0, +1.0])
    adv = compute_gtpo_ema_flipped_advantages(
        rewards, confidence, mask, alpha1=0.9, alpha2=0.1, lam=0.9,
    )
    # After z-norm, shaped_pos for the small-EMA sequence should be POSITIVE
    # (it gets the larger raw bonus), and negative for the big-EMA sequence.
    assert (adv[0] > adv[1]).all(), "small-EMA seq should now get larger adv"


# ── Flipped z-norm properties (Def 1.5) ─────────────────────────────────────

def test_znorm_separate_groups_mean_zero_std_one():
    B, T = 8, 10
    torch.manual_seed(7)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = torch.tensor([+1., +1., +1., +1., -1., -1., -1., -1.])
    adv = compute_gtpo_ema_flipped_advantages(rewards, confidence, mask)
    is_pos = (rewards > 0).float().unsqueeze(1) * mask
    pos_vals = adv[is_pos.bool()]
    assert abs(pos_vals.mean().item()) < 1e-4
    assert abs(pos_vals.std().item() - 1.0) < 1e-2
    is_neg = (rewards <= 0).float().unsqueeze(1) * mask
    neg_vals = adv[is_neg.bool()]
    assert abs(neg_vals.mean().item()) < 1e-4
    assert abs(neg_vals.std().item() - 1.0) < 1e-2


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_padding_tokens_zero():
    B, T = 3, 5
    torch.manual_seed(0)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.tensor([[1., 1., 1., 0., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 1.]])
    rewards = torch.tensor([+1.0, -1.0, +1.0])
    adv = compute_gtpo_ema_flipped_advantages(rewards, confidence, mask)
    assert (adv * (1 - mask)).abs().max().item() == 0.0


def test_all_positive_group():
    B, T = 4, 3
    torch.manual_seed(2)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = torch.ones(B)
    adv = compute_gtpo_ema_flipped_advantages(rewards, confidence, mask)
    assert adv.shape == (B, T) and adv.isfinite().all()


def test_all_negative_group():
    B, T = 4, 3
    torch.manual_seed(3)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = -torch.ones(B)
    adv = compute_gtpo_ema_flipped_advantages(rewards, confidence, mask)
    assert adv.shape == (B, T) and adv.isfinite().all()


def test_ema_inv_small_ema_bounded():
    """Verify EPS keeps 1/EMA bounded when some C/EMA is tiny."""
    B, T = 2, 3
    confidence = torch.tensor([[1e-9, 1e-9, 1e-9],
                               [1.0,  1.0,  1.0]])
    mask = torch.ones(B, T)
    rewards = torch.tensor([+1.0, +1.0])
    adv = compute_gtpo_ema_flipped_advantages(rewards, confidence, mask)
    assert adv.isfinite().all()
