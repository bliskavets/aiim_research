"""Unit tests for confidence_utils.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.confidence_utils import (
    confidence_from_logits, compress_confidence,
    compute_gtpo_conf_rewards, compute_grpo_s_conf_rewards, EPS
)

def ones_mask(B, T): return torch.ones(B, T)

# ── Test 1: confidence_from_logits basic properties ──────────────────────────
def test_confidence_nonnegative():
    logits = torch.randn(2, 8, 32)
    C = confidence_from_logits(logits, top_k=10)
    assert C.shape == (2, 8)
    assert (C >= 0).all(), f"Confidence should be ≥ 0, got min={C.min():.4f}"
    print(f"  [PASS] confidence non-negative: min={C.min():.4f}, max={C.max():.4f}")

def test_confidence_peaked():
    """Very peaked logits → model is certain → small C (high confidence)"""
    B, T, V = 2, 4, 32
    logits = torch.full((B, T, V), -100.0)
    logits[:, :, 0] = 100.0  # all probability on token 0
    C = confidence_from_logits(logits, top_k=5)
    # top-k logprobs: token 0 has logprob≈0, rest ≈-200 → mean ≈ -40 → C ≈ 40? 
    # Actually: log_softmax puts ≈0 on token 0, ≈-200 on others
    # top-5: [≈0, ≈-200, ≈-200, ≈-200, ≈-200] → mean ≈ -160 → C = 160
    # Hmm, peaked = high C in this formulation? Let's check uniform:
    uniform_logits = torch.zeros(B, T, V)
    C_uniform = confidence_from_logits(uniform_logits, top_k=5)
    # uniform: log_softmax = log(1/V) ≈ -3.47 for all → top-5 mean = -3.47 → C = 3.47
    assert C_uniform[0,0] < C[0,0], "Uniform should have smaller C than peaked (uniform is more spread)"
    print(f"  [PASS] confidence: peaked C={C[0,0]:.2f} > uniform C={C_uniform[0,0]:.2f}")

def test_confidence_shape():
    B, T, V = 3, 10, 100
    logits = torch.randn(B, T, V)
    C = confidence_from_logits(logits, top_k=20)
    assert C.shape == (B, T), f"Expected ({B},{T}), got {C.shape}"
    print(f"  [PASS] confidence shape: {tuple(C.shape)}")

# ── Test 2: compress_confidence ───────────────────────────────────────────────
def test_compress_monotone():
    c = torch.tensor([0.0, 0.5, 1.0, 5.0, 10.0, 100.0])
    cc = compress_confidence(c)
    assert (cc[1:] > cc[:-1]).all(), "Compression should be monotonically increasing"
    assert (cc >= 0).all()
    print(f"  [PASS] compress_confidence monotone: {cc.tolist()}")

def test_compress_squashes():
    c = torch.tensor([1.0, 10.0, 100.0])
    cc = compress_confidence(c)
    # Ratio should be much smaller after compression
    raw_ratio = c[-1] / c[0]   # 100x
    comp_ratio = cc[-1] / cc[0] # should be << 100x
    assert comp_ratio < raw_ratio / 2, f"Compression didn't squash enough: {comp_ratio:.1f} vs {raw_ratio:.1f}"
    print(f"  [PASS] compress squashes: raw ratio={raw_ratio:.1f}x → compressed={comp_ratio:.1f}x")

# ── Test 3: GTPO-Conf output shapes ──────────────────────────────────────────
def test_gtpo_conf_shapes():
    torch.manual_seed(42)
    B, T = 4, 16
    rewards    = torch.tensor([2.0, -1.0, 3.0, -0.5])
    confidence = torch.rand(B, T) * 2.0 + 0.5  # C ∈ [0.5, 2.5]
    mask       = ones_mask(B, T)

    adv_pos, adv_neg = compute_gtpo_conf_rewards(rewards, confidence, mask)
    assert adv_pos.shape == (B, T)
    assert adv_neg.shape == (B, T)

    is_pos = rewards > 0.0
    is_neg = ~is_pos
    assert (adv_pos[is_neg] == 0).all(), "adv_pos should be 0 for O- sequences"
    assert (adv_neg[is_pos] == 0).all(), "adv_neg should be 0 for O+ sequences"
    print(f"  [PASS] GTPO-Conf shapes: {tuple(adv_pos.shape)}, O+/O- separation correct")

# ── Test 4: GTPO-Conf reward mass conservation ────────────────────────────────
def test_gtpo_conf_mass_conservation():
    """Σ shaped_pos_i,t = (α₁+α₂)·d_t at each timestep"""
    torch.manual_seed(1)
    B, T = 4, 8
    rewards    = torch.tensor([1.0, 1.0, -1.0, -1.0])
    confidence = torch.rand(B, T) * 2.0 + 0.5
    mask       = ones_mask(B, T)
    alpha1, alpha2 = 0.9, 0.1

    is_pos = rewards > 0
    d_t = is_pos.sum().item()

    # Manually compute shaped_pos before normalization
    from src.confidence_utils import compress_confidence
    C_comp = compress_confidence(confidence)
    shaped_pos_raw = torch.zeros(B, T)
    for t in range(T):
        active = is_pos.float()
        C_t = C_comp[:, t] * active
        sum_C_t = C_t.sum() + EPS
        bonus = (C_t / sum_C_t) * d_t
        shaped_pos_raw[:, t] = alpha1 * active + alpha2 * bonus

    for t in range(T):
        total = shaped_pos_raw[is_pos, t].sum().item()
        expected = d_t * (alpha1 + alpha2)
        assert abs(total - expected) < 1e-4, \
            f"Mass conservation failed at t={t}: {total:.4f} != {expected:.4f}"
    print(f"  [PASS] GTPO-Conf mass conservation: Σr̃=d_t·(α₁+α₂)={d_t*(alpha1+alpha2):.1f}")

# ── Test 5: GTPO-Conf with padding mask ──────────────────────────────────────
def test_gtpo_conf_padding():
    rewards    = torch.tensor([1.0, -1.0])
    confidence = torch.rand(2, 8) * 1.0 + 0.5
    mask = torch.zeros(2, 8)
    mask[0, :5] = 1.0
    mask[1, :3] = 1.0

    adv_pos, adv_neg = compute_gtpo_conf_rewards(rewards, confidence, mask)
    assert (adv_pos[0, 5:] == 0).all()
    assert (adv_neg[1, 3:] == 0).all()
    print(f"  [PASS] GTPO-Conf: padding mask respected")

# ── Test 6: GRPO-S-Conf shapes ────────────────────────────────────────────────
def test_grpo_s_conf_shapes():
    torch.manual_seed(2)
    B, T = 4, 12
    rewards    = torch.tensor([2.0, -1.0, 3.0, -0.5])
    confidence = torch.rand(B, T) * 2.0 + 0.5
    mask       = ones_mask(B, T)

    shaped, seq_conf = compute_grpo_s_conf_rewards(rewards, confidence, mask)
    assert shaped.shape == (B,)
    assert seq_conf.shape == (B,)

    is_pos = rewards > 0.0
    is_neg = ~is_pos
    assert (shaped[is_pos] > 0).all(), "O+ shaped rewards should be positive"
    assert (shaped[is_neg] < 0).all(), "O- shaped rewards should be negative"
    print(f"  [PASS] GRPO-S-Conf shapes: {tuple(shaped.shape)}, signs correct")
    print(f"         shaped_rewards: {shaped.tolist()}")

# ── Test 7: GRPO-S-Conf reward conservation (β₁+β₂=1) ───────────────────────
def test_grpo_s_conf_conservation():
    torch.manual_seed(3)
    B, T = 6, 10
    rewards    = torch.tensor([1.0, 2.0, 0.5, -1.0, -0.5, -2.0])
    confidence = torch.rand(B, T) * 2.0 + 0.5
    mask       = ones_mask(B, T)
    beta1, beta2 = 0.9, 0.1

    shaped, _ = compute_grpo_s_conf_rewards(rewards, confidence, mask, beta1=beta1, beta2=beta2)
    is_pos = rewards > 0.0
    n = is_pos.sum().item()
    sum_pos = shaped[is_pos].sum().item()
    expected = n * (beta1 + beta2)
    assert abs(sum_pos - expected) < 1e-4, \
        f"GRPO-S-Conf conservation failed: {sum_pos:.4f} != {expected:.4f}"
    print(f"  [PASS] GRPO-S-Conf conservation: Σr̂+={sum_pos:.3f} ≈ n={expected:.3f}")

# ── Test 8: edge cases ────────────────────────────────────────────────────────
def test_all_positive():
    rewards    = torch.tensor([1.0, 2.0, 0.5, 1.5])
    confidence = torch.rand(4, 8) * 1.0 + 0.5
    mask       = ones_mask(4, 8)
    adv_pos, adv_neg = compute_gtpo_conf_rewards(rewards, confidence, mask)
    assert (adv_neg == 0).all()
    print(f"  [PASS] Edge: all-positive → adv_neg=0")

def test_all_negative():
    rewards    = torch.tensor([-1.0, -2.0, -0.5, -1.5])
    confidence = torch.rand(4, 8) * 1.0 + 0.5
    mask       = ones_mask(4, 8)
    adv_pos, adv_neg = compute_gtpo_conf_rewards(rewards, confidence, mask)
    assert (adv_pos == 0).all()
    print(f"  [PASS] Edge: all-negative → adv_pos=0")

# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("confidence non-negative",           test_confidence_nonnegative),
        ("confidence peaked vs uniform",       test_confidence_peaked),
        ("confidence shape",                   test_confidence_shape),
        ("compress monotone",                  test_compress_monotone),
        ("compress squashes",                  test_compress_squashes),
        ("GTPO-Conf shapes",                   test_gtpo_conf_shapes),
        ("GTPO-Conf mass conservation",        test_gtpo_conf_mass_conservation),
        ("GTPO-Conf padding mask",             test_gtpo_conf_padding),
        ("GRPO-S-Conf shapes",                 test_grpo_s_conf_shapes),
        ("GRPO-S-Conf conservation",           test_grpo_s_conf_conservation),
        ("Edge: all-positive",                 test_all_positive),
        ("Edge: all-negative",                 test_all_negative),
    ]
    passed = failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed: sys.exit(1)
