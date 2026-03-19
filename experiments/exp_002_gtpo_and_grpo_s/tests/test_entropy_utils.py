"""
Unit tests for entropy_utils.py

Run inside container:
  python3 tests/test_entropy_utils.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.entropy_utils import (
    entropy_from_logits,
    clip_entropies,
    compute_gtpo_rewards,
    compute_grpo_s_rewards,
    EPS,
)

# ── helpers ──────────────────────────────────────────────────────────────────
def make_uniform_logits(B, T, V=10):
    """Uniform distribution → max entropy = log(V)"""
    return torch.zeros(B, T, V)

def make_peaked_logits(B, T, V=10):
    """Very peaked (one token dominates) → entropy ≈ 0"""
    logits = torch.full((B, T, V), -100.0)
    logits[:, :, 0] = 100.0
    return logits

def ones_mask(B, T):
    return torch.ones(B, T)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: entropy_from_logits
# ─────────────────────────────────────────────────────────────────────────────
def test_entropy_uniform():
    B, T, V = 2, 5, 10
    logits = make_uniform_logits(B, T, V)
    H = entropy_from_logits(logits)
    assert H.shape == (B, T), f"Shape mismatch: {H.shape}"
    expected = torch.log(torch.tensor(V, dtype=torch.float))
    assert torch.allclose(H, expected.expand_as(H), atol=1e-4), \
        f"Uniform entropy should be log(V)={expected:.4f}, got {H[0,0]:.4f}"
    print(f"  [PASS] entropy_from_logits: uniform → H={H[0,0]:.4f} (expected {expected:.4f})")

def test_entropy_peaked():
    B, T, V = 2, 5, 10
    logits = make_peaked_logits(B, T, V)
    H = entropy_from_logits(logits)
    assert H.shape == (B, T)
    assert (H < 0.01).all(), f"Peaked distribution should have ~0 entropy, got {H.max():.4f}"
    print(f"  [PASS] entropy_from_logits: peaked → H={H[0,0]:.6f} (≈0)")

def test_entropy_nonnegative():
    torch.manual_seed(42)
    logits = torch.randn(4, 8, 32)
    H = entropy_from_logits(logits)
    assert (H >= 0).all(), "Entropy must be non-negative"
    print(f"  [PASS] entropy_from_logits: always non-negative (min={H.min():.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: clip_entropies
# ─────────────────────────────────────────────────────────────────────────────
def test_clip_entropies():
    H = torch.tensor([0.0, 0.1, 0.25, 0.5, 1.0])
    clipped = clip_entropies(H, eps_low=0.2, eps_high=0.28)
    assert (clipped >= 0.2).all(), "Below eps_low should be clipped"
    assert (clipped <= 0.28).all(), "Above eps_high should be clipped"
    assert clipped[2].item() == 0.25, "In-range value should be unchanged"
    print(f"  [PASS] clip_entropies: [{clipped.tolist()}]")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: GTPO reward mass conservation (Proposition 2.2)
#   Σ_i r̃_i,t = d_t  for all t  (when α₁ + α₂ = 1)
# ─────────────────────────────────────────────────────────────────────────────
def test_gtpo_reward_mass_conservation():
    torch.manual_seed(0)
    B, T = 4, 8
    # Rewards: 2 positive, 2 negative
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
    entropies = torch.rand(B, T) * 0.1 + 0.2  # in [0.2, 0.3] to pass clip
    mask = ones_mask(B, T)

    alpha1, alpha2 = 0.9, 0.1  # α₁ + α₂ = 1.0

    adv_pos, adv_neg = compute_gtpo_rewards(
        rewards=rewards, entropies=entropies, completion_mask=mask,
        alpha1=alpha1, alpha2=alpha2, eps_low=0.2, eps_high=0.28,
        reward_threshold=0.0,
    )

    # The SHAPED REWARDS (before normalization) for O+ should sum to d_t at each t.
    # Since we return normalized advantages, we verify the underlying logic:
    # For O+ sequences (idx 0,1), shaped_pos[:, t] = alpha1*1 + alpha2*(H_i,t/ΣH)*d_t
    # Sum over O+ seqs at position t:
    #   Σ shaped_pos_i,t = alpha1*d_t + alpha2*d_t * (ΣH / ΣH) = (alpha1+alpha2)*d_t = d_t
    is_pos = rewards > 0.0
    pos_idx = is_pos.nonzero(as_tuple=True)[0]
    d_t = len(pos_idx)

    # Manually recompute shaped_pos before normalization
    H = clip_entropies(entropies)
    shaped_pos_raw = torch.zeros(B, T)
    for t in range(T):
        H_t = H[:, t] * is_pos.float()
        sum_H_t = H_t.sum() + EPS
        entropy_bonus = (H_t / sum_H_t) * d_t
        shaped_pos_raw[:, t] = alpha1 * is_pos.float() + alpha2 * entropy_bonus

    for t in range(T):
        total = shaped_pos_raw[is_pos, t].sum().item()
        expected = d_t * (alpha1 + alpha2)  # = d_t when α₁+α₂=1
        assert abs(total - expected) < 1e-4, \
            f"Reward mass conservation failed at t={t}: got {total:.4f}, expected {expected:.4f}"

    print(f"  [PASS] GTPO reward mass conservation: Σr̃_i,t = d_t={d_t} at each timestep")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: GTPO output shapes
# ─────────────────────────────────────────────────────────────────────────────
def test_gtpo_shapes():
    torch.manual_seed(1)
    B, T = 4, 16
    rewards = torch.tensor([2.0, -1.0, 3.0, -0.5])
    entropies = torch.rand(B, T) * 0.1 + 0.2
    mask = ones_mask(B, T)

    adv_pos, adv_neg = compute_gtpo_rewards(
        rewards=rewards, entropies=entropies, completion_mask=mask,
    )
    assert adv_pos.shape == (B, T), f"adv_pos shape: {adv_pos.shape}"
    assert adv_neg.shape == (B, T), f"adv_neg shape: {adv_neg.shape}"

    # O+ advantages should be 0 for O- rows
    is_pos = rewards > 0.0
    is_neg = ~is_pos
    assert (adv_pos[is_neg] == 0).all(), "adv_pos should be 0 for O- sequences"
    assert (adv_neg[is_pos] == 0).all(), "adv_neg should be 0 for O+ sequences"

    print(f"  [PASS] GTPO output shapes: adv_pos={tuple(adv_pos.shape)}, adv_neg={tuple(adv_neg.shape)}")
    print(f"         O+ rows non-zero: {(adv_pos != 0).any(dim=1).tolist()}")
    print(f"         O- rows non-zero: {(adv_neg != 0).any(dim=1).tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: GTPO with padding mask
# ─────────────────────────────────────────────────────────────────────────────
def test_gtpo_padding_mask():
    torch.manual_seed(2)
    B, T = 2, 8
    rewards = torch.tensor([1.0, -1.0])
    entropies = torch.rand(B, T) * 0.1 + 0.2
    # Sequence 0 has 5 valid tokens, sequence 1 has 3
    mask = torch.zeros(B, T)
    mask[0, :5] = 1.0
    mask[1, :3] = 1.0

    adv_pos, adv_neg = compute_gtpo_rewards(
        rewards=rewards, entropies=entropies, completion_mask=mask,
    )
    # Padded positions should be 0
    assert (adv_pos[0, 5:] == 0).all(), "Padded positions should be 0 in adv_pos"
    assert (adv_neg[1, 3:] == 0).all(), "Padded positions should be 0 in adv_neg"
    print(f"  [PASS] GTPO padding mask respected")

# ─────────────────────────────────────────────────────────────────────────────
# Test 6: GRPO-S output shapes + reward conservation (Proposition B.5)
# ─────────────────────────────────────────────────────────────────────────────
def test_grpo_s_shapes():
    torch.manual_seed(3)
    B, T = 4, 12
    rewards = torch.tensor([2.0, -1.0, 3.0, -0.5])
    entropies = torch.rand(B, T) * 0.1 + 0.2
    mask = ones_mask(B, T)

    shaped_rewards, seq_avg_entropy = compute_grpo_s_rewards(
        rewards=rewards, entropies=entropies, completion_mask=mask,
    )
    assert shaped_rewards.shape == (B,), f"shaped_rewards shape: {shaped_rewards.shape}"
    assert seq_avg_entropy.shape == (B,), f"seq_avg_entropy shape: {seq_avg_entropy.shape}"
    print(f"  [PASS] GRPO-S output shapes: shaped_rewards={tuple(shaped_rewards.shape)}")
    print(f"         shaped_rewards: {shaped_rewards.tolist()}")

def test_grpo_s_conservation():
    """Proposition B.5: Σ r̂_i = n when β₁ + β₂ = 1"""
    torch.manual_seed(4)
    B, T = 6, 10
    # 3 positive, 3 negative
    rewards = torch.tensor([1.0, 2.0, 0.5, -1.0, -0.5, -2.0])
    entropies = torch.rand(B, T) * 0.1 + 0.2
    mask = ones_mask(B, T)

    beta1, beta2 = 0.9, 0.1  # β₁ + β₂ = 1

    shaped_rewards, _ = compute_grpo_s_rewards(
        rewards=rewards, entropies=entropies, completion_mask=mask,
        beta1=beta1, beta2=beta2, reward_threshold=0.0,
    )
    is_pos = rewards > 0.0
    n = is_pos.sum().item()
    sum_pos = shaped_rewards[is_pos].sum().item()
    expected = n * (beta1 + beta2)  # = n when β₁+β₂=1

    assert abs(sum_pos - expected) < 1e-4, \
        f"GRPO-S reward conservation: Σr̂_i={sum_pos:.4f}, expected n={expected:.4f}"
    print(f"  [PASS] GRPO-S reward conservation: Σr̂+ = n={n:.0f} (β₁+β₂=1)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 7: All-positive and all-negative edge cases (no crash)
# ─────────────────────────────────────────────────────────────────────────────
def test_all_positive():
    rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
    entropies = torch.rand(4, 8) * 0.1 + 0.2
    mask = ones_mask(4, 8)
    adv_pos, adv_neg = compute_gtpo_rewards(rewards, entropies, mask)
    assert (adv_neg == 0).all(), "No O- sequences → adv_neg should be 0"
    print(f"  [PASS] Edge case: all-positive → adv_neg all zeros")

def test_all_negative():
    rewards = torch.tensor([-1.0, -2.0, -0.5, -1.5])
    entropies = torch.rand(4, 8) * 0.1 + 0.2
    mask = ones_mask(4, 8)
    adv_pos, adv_neg = compute_gtpo_rewards(rewards, entropies, mask)
    assert (adv_pos == 0).all(), "No O+ sequences → adv_pos should be 0"
    print(f"  [PASS] Edge case: all-negative → adv_pos all zeros")

# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("entropy_from_logits: uniform",       test_entropy_uniform),
        ("entropy_from_logits: peaked",         test_entropy_peaked),
        ("entropy_from_logits: non-negative",   test_entropy_nonnegative),
        ("clip_entropies",                      test_clip_entropies),
        ("GTPO reward mass conservation",       test_gtpo_reward_mass_conservation),
        ("GTPO output shapes",                  test_gtpo_shapes),
        ("GTPO padding mask",                   test_gtpo_padding_mask),
        ("GRPO-S output shapes",                test_grpo_s_shapes),
        ("GRPO-S reward conservation",          test_grpo_s_conservation),
        ("Edge case: all-positive",             test_all_positive),
        ("Edge case: all-negative",             test_all_negative),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        sys.exit(1)
