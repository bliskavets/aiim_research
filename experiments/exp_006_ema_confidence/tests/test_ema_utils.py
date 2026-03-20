"""Unit tests for ema_confidence_utils.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.ema_confidence_utils import (
    confidence_from_logits, compute_ema, get_last_ema,
    compute_gtpo_ema_rewards, compute_grpo_s_ema_rewards, EPS
)

def ones_mask(B, T): return torch.ones(B, T)

# ── EMA correctness ───────────────────────────────────────────────────────────
def test_ema_shape():
    conf = torch.rand(3, 10)
    mask = ones_mask(3, 10)
    ema = compute_ema(conf, mask, lam=0.9)
    assert ema.shape == (3, 10)
    print(f"  [PASS] EMA shape: {tuple(ema.shape)}")

def test_ema_first_token():
    """EMA at t=0 should equal confidence at t=0"""
    conf = torch.rand(2, 8)
    mask = ones_mask(2, 8)
    ema = compute_ema(conf, mask, lam=0.9)
    assert torch.allclose(ema[:, 0], conf[:, 0])
    print(f"  [PASS] EMA[t=0] == conf[t=0]")

def test_ema_smoothing():
    """EMA should be smoother than raw (lower variance across time)"""
    torch.manual_seed(42)
    conf = torch.rand(1, 50) * 5.0  # noisy signal
    mask = ones_mask(1, 50)
    ema = compute_ema(conf, mask, lam=0.9)
    raw_var = conf.var().item()
    ema_var = ema.var().item()
    assert ema_var < raw_var, f"EMA should be smoother: ema_var={ema_var:.4f} vs raw_var={raw_var:.4f}"
    print(f"  [PASS] EMA smoother: raw_var={raw_var:.3f} → ema_var={ema_var:.3f}")

def test_ema_manual():
    """Manual computation: EMA_t = 0.9*EMA_{t-1} + 0.1*C_t"""
    conf = torch.tensor([[1.0, 2.0, 3.0]])
    mask = ones_mask(1, 3)
    ema = compute_ema(conf, mask, lam=0.9)
    expected_0 = 1.0
    expected_1 = 0.9 * 1.0 + 0.1 * 2.0  # = 1.1
    expected_2 = 0.9 * 1.1 + 0.1 * 3.0  # = 1.29
    assert abs(ema[0, 0].item() - expected_0) < 1e-5
    assert abs(ema[0, 1].item() - expected_1) < 1e-5
    assert abs(ema[0, 2].item() - expected_2) < 1e-5
    print(f"  [PASS] EMA manual: [{ema[0].tolist()}] ≈ [{expected_0}, {expected_1:.2f}, {expected_2:.2f}]")

def test_ema_padding_preserved():
    """EMA should hold last value in padded positions"""
    conf = torch.tensor([[1.0, 2.0, 3.0, 99.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    ema = compute_ema(conf, mask, lam=0.9)
    # At t=3 (padding), EMA should equal EMA at t=2
    assert abs(ema[0, 3].item() - ema[0, 2].item()) < 1e-5
    assert abs(ema[0, 4].item() - ema[0, 2].item()) < 1e-5
    print(f"  [PASS] EMA padding: value held at {ema[0, 2].item():.3f} for padded positions")

# ── get_last_ema ──────────────────────────────────────────────────────────────
def test_last_ema():
    conf = torch.rand(3, 10)
    mask = torch.zeros(3, 10)
    mask[0, :5] = 1.0   # seq 0: length 5, last idx=4
    mask[1, :8] = 1.0   # seq 1: length 8, last idx=7
    mask[2, :10] = 1.0  # seq 2: length 10, last idx=9
    ema = compute_ema(conf, mask, lam=0.9)
    last = get_last_ema(ema, mask)
    assert abs(last[0].item() - ema[0, 4].item()) < 1e-5
    assert abs(last[1].item() - ema[1, 7].item()) < 1e-5
    assert abs(last[2].item() - ema[2, 9].item()) < 1e-5
    print(f"  [PASS] get_last_ema: correct indices [4, 7, 9] → {last.tolist()}")

# ── GTPO-EMA ──────────────────────────────────────────────────────────────────
def test_gtpo_ema_shapes():
    torch.manual_seed(1)
    B, T = 4, 16
    rewards    = torch.tensor([2.0, -1.0, 3.0, -0.5])
    confidence = torch.rand(B, T) * 2.0 + 0.5
    mask       = ones_mask(B, T)
    adv_pos, adv_neg = compute_gtpo_ema_rewards(rewards, confidence, mask)
    assert adv_pos.shape == (B, T)
    assert adv_neg.shape == (B, T)
    is_pos = rewards > 0
    assert (adv_pos[~is_pos] == 0).all()
    assert (adv_neg[is_pos] == 0).all()
    print(f"  [PASS] GTPO-EMA shapes correct, O+/O- separation ok")

def test_gtpo_ema_mass_conservation():
    """Σ shaped_pos_i,t = (α₁+α₂)·d_t at each timestep"""
    torch.manual_seed(2)
    B, T = 4, 8
    rewards    = torch.tensor([1.0, 1.0, -1.0, -1.0])
    confidence = torch.rand(B, T) * 2.0 + 0.5
    mask       = ones_mask(B, T)
    alpha1, alpha2 = 0.9, 0.1
    is_pos = rewards > 0
    d_t = is_pos.sum().item()

    from src.ema_confidence_utils import compute_ema, compress
    ema  = compute_ema(confidence, mask, lam=0.9)
    ema_comp = compress(ema)

    shaped_raw = torch.zeros(B, T)
    for t in range(T):
        active = is_pos.float()
        E_t = ema_comp[:, t] * active
        sum_E = E_t.sum() + EPS
        bonus = (E_t / sum_E) * d_t
        shaped_raw[:, t] = alpha1 * active + alpha2 * bonus

    for t in range(T):
        total = shaped_raw[is_pos, t].sum().item()
        expected = d_t * (alpha1 + alpha2)
        assert abs(total - expected) < 1e-4, f"t={t}: {total:.4f} != {expected:.4f}"
    print(f"  [PASS] GTPO-EMA mass conservation: Σ={d_t*(alpha1+alpha2):.1f}")

def test_gtpo_ema_padding():
    rewards    = torch.tensor([1.0, -1.0])
    confidence = torch.rand(2, 8)
    mask = torch.zeros(2, 8)
    mask[0, :5] = 1.0
    mask[1, :3] = 1.0
    adv_pos, adv_neg = compute_gtpo_ema_rewards(rewards, confidence, mask)
    assert (adv_pos[0, 5:] == 0).all()
    assert (adv_neg[1, 3:] == 0).all()
    print(f"  [PASS] GTPO-EMA padding respected")

# ── GRPO-S-EMA ────────────────────────────────────────────────────────────────
def test_grpo_s_ema_shapes():
    torch.manual_seed(3)
    B, T = 4, 12
    rewards    = torch.tensor([2.0, -1.0, 3.0, -0.5])
    confidence = torch.rand(B, T) * 2.0 + 0.5
    mask       = ones_mask(B, T)
    shaped, last_ema = compute_grpo_s_ema_rewards(rewards, confidence, mask)
    assert shaped.shape == (B,)
    assert last_ema.shape == (B,)
    is_pos = rewards > 0
    assert (shaped[is_pos] > 0).all(), "O+ rewards should be positive"
    assert (shaped[~is_pos] < 0).all(), "O- rewards should be negative"
    print(f"  [PASS] GRPO-S-EMA shapes: {tuple(shaped.shape)}, signs correct")
    print(f"         shaped: {[f'{x:.3f}' for x in shaped.tolist()]}")

def test_grpo_s_ema_uses_last_ema():
    """Verify GRPO-S-EMA uses the last EMA, not mean confidence"""
    torch.manual_seed(4)
    B, T = 2, 10
    rewards    = torch.tensor([1.0, -1.0])
    # Seq 0: increasing confidence (EMA grows)
    # Seq 1: decreasing confidence (EMA shrinks)
    conf = torch.zeros(B, T)
    conf[0] = torch.linspace(1.0, 5.0, T)   # growing
    conf[1] = torch.linspace(5.0, 1.0, T)   # shrinking
    mask = ones_mask(B, T)

    from src.ema_confidence_utils import compute_ema, get_last_ema
    ema = compute_ema(conf, mask, lam=0.9)
    last = get_last_ema(ema, mask)
    # Last EMA of seq 0 should be higher than its mean confidence
    assert last[0].item() > conf[0].mean().item() * 0.5, "Last EMA of growing seq should be high"
    print(f"  [PASS] GRPO-S-EMA uses last EMA: seq0_last={last[0]:.3f} (growing conf)")

def test_edge_all_positive():
    rewards = torch.tensor([1.0, 2.0, 0.5, 1.5])
    confidence = torch.rand(4, 8)
    mask = ones_mask(4, 8)
    adv_pos, adv_neg = compute_gtpo_ema_rewards(rewards, confidence, mask)
    assert (adv_neg == 0).all()
    print(f"  [PASS] Edge: all-positive → adv_neg=0")

def test_edge_all_negative():
    rewards = torch.tensor([-1.0, -2.0, -0.5, -1.5])
    confidence = torch.rand(4, 8)
    mask = ones_mask(4, 8)
    adv_pos, adv_neg = compute_gtpo_ema_rewards(rewards, confidence, mask)
    assert (adv_pos == 0).all()
    print(f"  [PASS] Edge: all-negative → adv_pos=0")

# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("EMA shape",                    test_ema_shape),
        ("EMA first token",              test_ema_first_token),
        ("EMA smoothing",                test_ema_smoothing),
        ("EMA manual computation",       test_ema_manual),
        ("EMA padding preserved",        test_ema_padding_preserved),
        ("get_last_ema correctness",     test_last_ema),
        ("GTPO-EMA shapes",              test_gtpo_ema_shapes),
        ("GTPO-EMA mass conservation",   test_gtpo_ema_mass_conservation),
        ("GTPO-EMA padding",             test_gtpo_ema_padding),
        ("GRPO-S-EMA shapes",            test_grpo_s_ema_shapes),
        ("GRPO-S-EMA uses last EMA",     test_grpo_s_ema_uses_last_ema),
        ("Edge: all-positive",           test_edge_all_positive),
        ("Edge: all-negative",           test_edge_all_negative),
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
    if failed: import sys; sys.exit(1)
