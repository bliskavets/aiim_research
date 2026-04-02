"""
Tests for ema_confidence_utils_v2.py (exp_010)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.ema_confidence_utils_v2 import (
    confidence_from_logits,
    compute_ema_vectorized,
    get_last_ema,
    group_normalize,
    compress,
    compute_gtpo_ema_advantages,
    compute_grpo_s_ema_advantages,
    EPS,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_batch(B=4, T=8, V=100, seed=42):
    torch.manual_seed(seed)
    logits = torch.randn(B, T, V)
    mask   = torch.ones(B, T)
    mask[0, 6:] = 0   # seq 0 is shorter
    mask[2, 4:] = 0   # seq 2 is shorter
    rewards = torch.tensor([-1.0, 2.0, -0.5, 3.0])
    return logits, mask, rewards

# ─── confidence_from_logits ───────────────────────────────────────────────────

def test_confidence_shape():
    logits, mask, _ = make_batch()
    conf = confidence_from_logits(logits, top_k=20)
    assert conf.shape == (4, 8), f"Expected (4,8), got {conf.shape}"
    print("  [PASS] confidence shape")

def test_confidence_nonneg():
    logits, mask, _ = make_batch()
    conf = confidence_from_logits(logits, top_k=20)
    assert (conf >= 0).all(), "confidence should be >= 0"
    print("  [PASS] confidence non-negative")

def test_confidence_top_k_clamp():
    """top_k > vocab should not crash"""
    logits = torch.randn(2, 5, 10)
    conf = confidence_from_logits(logits, top_k=50)
    assert conf.shape == (2, 5)
    print("  [PASS] confidence top_k clamp")

# ─── compute_ema_vectorized ────────────────────────────────────────────────────

def test_ema_shape():
    logits, mask, _ = make_batch()
    conf = confidence_from_logits(logits, top_k=20)
    ema = compute_ema_vectorized(conf, mask, lam=0.9)
    assert ema.shape == conf.shape
    print("  [PASS] EMA shape")

def test_ema_first_token():
    """EMA at t=0 should equal confidence at t=0 (for valid tokens)."""
    conf = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    mask = torch.ones(2, 3)
    ema  = compute_ema_vectorized(conf, mask, lam=0.9)
    assert torch.allclose(ema[:, 0], conf[:, 0]), "EMA[0] != conf[0]"
    print("  [PASS] EMA first token equals confidence")

def test_ema_smoothing():
    """EMA should be smoother than raw confidence."""
    torch.manual_seed(0)
    conf = torch.rand(1, 20) * 5
    mask = torch.ones(1, 20)
    ema  = compute_ema_vectorized(conf, mask, lam=0.9)
    raw_var = conf.var().item()
    ema_var = ema.var().item()
    assert ema_var < raw_var, f"EMA var {ema_var:.4f} should < raw var {raw_var:.4f}"
    print(f"  [PASS] EMA smoother: raw_var={raw_var:.4f}, ema_var={ema_var:.4f}")

def test_ema_padding_stable():
    """EMA should not change after padding starts."""
    conf = torch.tensor([[1.0, 2.0, 3.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    ema  = compute_ema_vectorized(conf, mask, lam=0.9)
    assert ema[0, 3] == ema[0, 2], "EMA should be stable after padding"
    print("  [PASS] EMA stable at padding")

# ─── get_last_ema ─────────────────────────────────────────────────────────────

def test_last_ema_full_seq():
    conf = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    mask = torch.ones(2, 3)
    ema  = compute_ema_vectorized(conf, mask, lam=0.9)
    last = get_last_ema(ema, mask)
    assert last.shape == (2,)
    assert torch.allclose(last[0], ema[0, 2]) and torch.allclose(last[1], ema[1, 2])
    print("  [PASS] last EMA full sequence")

def test_last_ema_padded():
    conf = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    ema  = compute_ema_vectorized(conf, mask, lam=0.9)
    last = get_last_ema(ema, mask)
    assert torch.allclose(last[0], ema[0, 1]), f"Expected ema[0,1]={ema[0,1]}, got {last[0]}"
    print("  [PASS] last EMA padded sequence")

# ─── group_normalize ──────────────────────────────────────────────────────────

def test_group_normalize_basic():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n = group_normalize(x)
    assert abs(n.mean().item()) < 1e-5
    assert abs(n.std().item() - 1.0) < 1e-4
    print("  [PASS] group_normalize mean=0, std=1")

def test_group_normalize_single():
    x = torch.tensor([5.0])
    n = group_normalize(x)
    assert n[0] == 0.0
    print("  [PASS] group_normalize single element = 0")

# ─── compute_gtpo_ema_advantages ──────────────────────────────────────────────

def test_gtpo_advantages_shape():
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv  = compute_gtpo_ema_advantages(rewards, conf, mask)
    assert adv.shape == (4, 8), f"Expected (4,8), got {adv.shape}"
    print("  [PASS] GTPO advantages shape")

def test_gtpo_advantages_masked():
    """Padded positions should have zero advantage."""
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv  = compute_gtpo_ema_advantages(rewards, conf, mask)
    pad_adv = adv[mask == 0]
    assert (pad_adv == 0).all(), "Padded positions should be zero"
    print("  [PASS] GTPO advantages zero at padding")

def test_gtpo_ema_signal_preserved():
    """
    Key test: with alpha2 > 0, advantages should differ from pure GRPO (alpha2=0).
    This tests that the EMA signal is NOT washed out.
    """
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv_with_ema  = compute_gtpo_ema_advantages(rewards, conf, mask, alpha1=1.0, alpha2=0.5)
    adv_grpo_only = compute_gtpo_ema_advantages(rewards, conf, mask, alpha1=1.0, alpha2=0.0)
    diff = (adv_with_ema - adv_grpo_only).abs().sum().item()
    assert diff > 0.01, f"EMA signal should affect advantages, diff={diff:.6f}"
    print(f"  [PASS] EMA signal preserved: adv diff = {diff:.4f}")

# ─── compute_grpo_s_ema_advantages ────────────────────────────────────────────

def test_grpo_s_advantages_shape():
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv, last_ema = compute_grpo_s_ema_advantages(rewards, conf, mask)
    assert adv.shape == (4, 8)
    assert last_ema.shape == (4,)
    print("  [PASS] GRPO-S advantages shape")

def test_grpo_s_advantages_masked():
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv, _ = compute_grpo_s_ema_advantages(rewards, conf, mask)
    pad_adv = adv[mask == 0]
    assert (pad_adv == 0).all(), "Padded positions should be zero"
    print("  [PASS] GRPO-S advantages zero at padding")

def test_grpo_s_ema_signal_preserved():
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv_with, _ = compute_grpo_s_ema_advantages(rewards, conf, mask, beta1=1.0, beta2=0.5)
    adv_base, _ = compute_grpo_s_ema_advantages(rewards, conf, mask, beta1=1.0, beta2=0.0)
    diff = (adv_with - adv_base).abs().sum().item()
    assert diff > 0.01, f"EMA signal should affect advantages, diff={diff:.6f}"
    print(f"  [PASS] GRPO-S EMA signal preserved: adv diff = {diff:.4f}")

def test_grpo_s_constant_within_sequence():
    """
    GRPO-S: all valid tokens in a sequence should have the same advantage value.
    """
    logits, mask, rewards = make_batch()
    conf = confidence_from_logits(logits)
    adv, _ = compute_grpo_s_ema_advantages(rewards, conf, mask)
    for i in range(4):
        valid_advs = adv[i][mask[i] == 1]
        if valid_advs.numel() > 1:
            assert valid_advs.std() < 1e-5, f"Seq {i}: advantages not constant: {valid_advs}"
    print("  [PASS] GRPO-S advantages constant within sequence")

# ─── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_confidence_shape,
        test_confidence_nonneg,
        test_confidence_top_k_clamp,
        test_ema_shape,
        test_ema_first_token,
        test_ema_smoothing,
        test_ema_padding_stable,
        test_last_ema_full_seq,
        test_last_ema_padded,
        test_group_normalize_basic,
        test_group_normalize_single,
        test_gtpo_advantages_shape,
        test_gtpo_advantages_masked,
        test_gtpo_ema_signal_preserved,
        test_grpo_s_advantages_shape,
        test_grpo_s_advantages_masked,
        test_grpo_s_ema_signal_preserved,
        test_grpo_s_constant_within_sequence,
    ]
    passed = failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"[TEST] {name}")
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        exit(1)
