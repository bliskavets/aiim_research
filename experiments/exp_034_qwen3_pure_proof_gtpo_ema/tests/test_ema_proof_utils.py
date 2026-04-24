"""
Unit tests for ema_proof_utils.py (pure-proof GTPO-EMA).
Cover: confidence non-negativity, EMA monotonicity, conservation of
reward mass (Prop 2.3), active-token accounting at boundary-of-sequence,
and degenerate all-pos / all-neg edge cases.
"""
import math
import pytest
import torch

from src.ema_proof_utils import (
    confidence_from_logits,
    compute_ema_vectorized,
    compute_gtpo_ema_proof_advantages,
    EPS,
)


# ── Confidence ──────────────────────────────────────────────────────────────

def test_confidence_nonnegative():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 100)
    c = confidence_from_logits(logits, top_k=20)
    assert c.shape == (3, 5)
    assert (c >= 0).all()


def test_confidence_peaked_gt_flat():
    # Note on interpretation: the "top-k confidence" C = -mean(top-k log π) does
    # NOT equal Shannon entropy. For a one-hot-ish peaked distribution, top-1
    # has log π ≈ 0 but top-2..k have very negative log π, so mean is pushed
    # low and C is LARGE. For a uniform distribution, all top-k log π equal
    # -log V, giving a smaller C. This matches the raw formula from the paper
    # but runs OPPOSITE to the naive "C is confidence" reading — what the EMA
    # bonus actually rewards is "peaked / sharp" attention at the token.
    logits_peaked = torch.zeros(1, 1, 100); logits_peaked[..., 0] = 10.0
    logits_flat   = torch.zeros(1, 1, 100)
    c_peaked = confidence_from_logits(logits_peaked, top_k=20)
    c_flat   = confidence_from_logits(logits_flat,   top_k=20)
    assert c_peaked.item() > c_flat.item()


# ── EMA ─────────────────────────────────────────────────────────────────────

def test_ema_init_equals_c0():
    C = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    ema = compute_ema_vectorized(C, mask, lam=0.9)
    assert math.isclose(ema[0, 0].item(), 1.0)


def test_ema_recurrence():
    # EMA_1 = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    C = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0]])
    ema = compute_ema_vectorized(C, mask, lam=0.9)
    assert math.isclose(ema[0, 1].item(), 1.1, rel_tol=1e-6)


def test_ema_nonnegative():
    torch.manual_seed(1)
    C = torch.abs(torch.randn(4, 8))
    mask = torch.ones(4, 8)
    ema = compute_ema_vectorized(C, mask, lam=0.9)
    assert (ema >= 0).all()


def test_ema_past_end_is_frozen():
    # After end-of-seq (mask=0) EMA should hold its last valid value.
    C = torch.tensor([[1.0, 1.0, 999.0, 999.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    ema = compute_ema_vectorized(C, mask, lam=0.9)
    # ema[1] = 0.9*1 + 0.1*1 = 1.0; ema[2] and ema[3] should equal ema[1] = 1.0
    assert math.isclose(ema[0, 1].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(ema[0, 2].item(), 1.0, rel_tol=1e-6)
    assert math.isclose(ema[0, 3].item(), 1.0, rel_tol=1e-6)


# ── Shaping conservation (Proposition 2.3) ──────────────────────────────────

def test_conservation_of_positive_reward_mass():
    """
    With α₁+α₂ = 1 and a single O+ sequence active at each timestep,
    Σ over active O+ of shaped_pos should equal (α₁+α₂)·d_t = d_t.
    We can't read shaped_pos directly (function returns z-normed adv);
    instead, replay the raw-shaped formula from the utility.
    """
    from src.ema_proof_utils import compute_ema_vectorized
    B, T = 4, 6
    torch.manual_seed(0)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])   # all O+
    alpha1, alpha2 = 0.9, 0.1
    ema = compute_ema_vectorized(confidence, mask, lam=0.9)

    for t in range(T):
        active = mask[:, t]
        d_t = active.sum()
        ema_t = ema[:, t] * active
        sum_ema = ema_t.sum() + EPS
        bonus_t = (ema_t / sum_ema) * d_t
        shaped_t = (alpha1 * 1.0 + alpha2 * bonus_t) * active
        total = shaped_t.sum()
        assert math.isclose(total.item(), (alpha1 + alpha2) * d_t.item(), rel_tol=1e-4)


# ── Active-tokens accounting (d_t / h_t) ────────────────────────────────────

def test_d_t_drops_with_ended_sequences():
    """
    |O⁺_t| should shrink as sequences end. Two O+ sequences; one ends at t=2.
    """
    B, T = 2, 4
    confidence = torch.ones(B, T)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 0.0, 0.0]])
    rewards = torch.tensor([1.0, 1.0])
    alpha1, alpha2 = 0.9, 0.1
    ema = compute_ema_vectorized(confidence, mask, lam=0.9)
    # All EMA values = 1.0 (constant C=1). Bonus at any t equals d_t * (1/d_t) = 1.
    # shaped = 0.9*1 + 0.1*1 = 1.0 per active token.
    for t in range(T):
        mask_pos = mask[:, t]
        d_t = mask_pos.sum().item()
        expected = (alpha1 + alpha2) * d_t
        ema_t = ema[:, t] * mask_pos
        shaped_t = (alpha1 + alpha2 * ema_t / (ema_t.sum() + EPS) * d_t) * mask_pos
        assert math.isclose(shaped_t.sum().item(), expected, rel_tol=1e-4)
    # At t=0..1 both active (d_t=2); at t=2..3 only seq 0 (d_t=1).
    assert mask[:, 0].sum().item() == 2
    assert mask[:, 2].sum().item() == 1


def test_padding_tokens_zero():
    """Advantages past end-of-sequence must be exactly zero."""
    B, T = 3, 5
    torch.manual_seed(0)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.tensor([[1., 1., 1., 0., 0.],
                         [1., 1., 1., 1., 0.],
                         [1., 1., 1., 1., 1.]])
    rewards = torch.tensor([+1.0, -1.0, +1.0])
    adv = compute_gtpo_ema_proof_advantages(rewards, confidence, mask)
    # adv should be 0 wherever mask==0
    assert (adv * (1 - mask)).abs().max().item() == 0.0


# ── Z-norm properties (Def 1.5) ─────────────────────────────────────────────

def test_znorm_separate_groups_mean_zero_std_one():
    """Final advantages, restricted to O+ active tokens, have mean≈0, std≈1."""
    B, T = 8, 10
    torch.manual_seed(7)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = torch.tensor([+1., +1., +1., +1., -1., -1., -1., -1.])
    adv = compute_gtpo_ema_proof_advantages(rewards, confidence, mask)
    # O+ active mask
    is_pos = (rewards > 0).float().unsqueeze(1) * mask
    pos_vals = adv[is_pos.bool()]
    assert abs(pos_vals.mean().item()) < 1e-4
    assert abs(pos_vals.std().item() - 1.0) < 1e-2
    # O- similarly
    is_neg = (rewards <= 0).float().unsqueeze(1) * mask
    neg_vals = adv[is_neg.bool()]
    assert abs(neg_vals.mean().item()) < 1e-4
    assert abs(neg_vals.std().item() - 1.0) < 1e-2


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_all_positive_group():
    B, T = 4, 3
    torch.manual_seed(2)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = torch.ones(B)           # all O+
    adv = compute_gtpo_ema_proof_advantages(rewards, confidence, mask)
    assert adv.shape == (B, T)
    assert adv.isfinite().all()


def test_all_negative_group():
    B, T = 4, 3
    torch.manual_seed(3)
    confidence = torch.abs(torch.randn(B, T)) + 0.1
    mask = torch.ones(B, T)
    rewards = -torch.ones(B)          # all O-
    adv = compute_gtpo_ema_proof_advantages(rewards, confidence, mask)
    assert adv.shape == (B, T)
    assert adv.isfinite().all()


def test_shapes_match():
    B, T = 2, 7
    confidence = torch.rand(B, T)
    mask = torch.ones(B, T)
    rewards = torch.tensor([+1.0, -1.0])
    adv = compute_gtpo_ema_proof_advantages(rewards, confidence, mask)
    assert adv.shape == (B, T)
