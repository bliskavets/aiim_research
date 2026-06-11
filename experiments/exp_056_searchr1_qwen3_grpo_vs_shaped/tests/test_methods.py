"""
Unit tests for the three shaping methods used in exp_049 (no training).
Run: pytest tests/ -q   (needs torch, CPU is fine)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.entropy_utils import compute_grpo_s_rewards
from src.confidence_utils import confidence_from_logits, compute_gtpo_conf_rewards
from src.ema_flipped_utils import (
    confidence_from_logits as conf_flipped,
    compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages,
)


def _toy_batch(B=4, T=6, V=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g)
    mask = torch.ones(B, T)
    mask[0, 4:] = 0  # seq 0 shorter
    rewards = torch.tensor([2.0, -1.0, 3.0, -1.5])  # 2 O+, 2 O-
    return logits, mask, rewards


# ── confidence metric ───────────────────────────────────────────────────────

def test_confidence_peaked_gt_flat():
    """Peaked (near one-hot) logits give LARGER C than flat logits — the
    property the flipped variant relies on (top-k log-probs very negative)."""
    V = 50
    peaked = torch.full((1, 1, V), -10.0); peaked[0, 0, 0] = 10.0
    flat = torch.zeros(1, 1, V)
    c_peaked = confidence_from_logits(peaked, top_k=20)
    c_flat = confidence_from_logits(flat, top_k=20)
    assert c_peaked.item() > c_flat.item()


def test_confidence_nonnegative():
    logits, _, _ = _toy_batch()
    c = confidence_from_logits(logits, top_k=20)
    assert (c >= 0).all()
    assert torch.allclose(c, conf_flipped(logits, top_k=20))


# ── GRPO-S entropy (candidate A) ─────────────────────────────────────────────

def test_grpo_s_signs_and_shape():
    _, mask, rewards = _toy_batch()
    entropies = torch.rand(rewards.shape[0], mask.shape[1]) * 0.3
    shaped, h_avg = compute_grpo_s_rewards(
        rewards=rewards, entropies=entropies, completion_mask=mask,
        beta1=1.0, beta2=0.1, reward_threshold=0.0,
    )
    assert shaped.shape == rewards.shape
    assert h_avg.shape == rewards.shape
    assert (shaped[rewards > 0] > 0).all()   # O+ stays positive
    assert (shaped[rewards <= 0] < 0).all()  # O- stays negative


# ── GTPO-Conf (candidate B) ──────────────────────────────────────────────────

def test_gtpo_conf_token_advantages():
    logits, mask, rewards = _toy_batch()
    conf = confidence_from_logits(logits, top_k=20)
    adv_pos, adv_neg = compute_gtpo_conf_rewards(
        rewards=rewards, confidence=conf, completion_mask=mask,
        alpha1=1.0, alpha2=0.1, top_k=20, reward_threshold=0.0,
    )
    assert adv_pos.shape == conf.shape
    assert adv_neg.shape == conf.shape
    # O- sequences contribute nothing to adv_pos and vice versa
    is_pos = rewards > 0
    assert torch.count_nonzero(adv_pos[~is_pos]) == 0
    assert torch.count_nonzero(adv_neg[is_pos]) == 0
    assert torch.isfinite(adv_pos + adv_neg).all()


# ── GTPO-EMA-flipped (candidate C) ───────────────────────────────────────────

def test_ema_recursion_and_mask():
    conf = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    ema = compute_ema_vectorized(conf, mask, lam=0.5)
    assert torch.isclose(ema[0, 0], torch.tensor(1.0))
    assert torch.isclose(ema[0, 1], torch.tensor(1.5))   # 0.5*1 + 0.5*2
    assert torch.isclose(ema[0, 2], ema[0, 1])           # frozen past mask


def test_gtpo_ema_flipped_advantages():
    logits, mask, rewards = _toy_batch()
    conf = conf_flipped(logits, top_k=20)
    adv = compute_gtpo_ema_flipped_advantages(
        rewards=rewards, confidence=conf, completion_mask=mask,
        alpha1=0.9, alpha2=0.1, lam=0.9, reward_threshold=0.0,
    )
    assert adv.shape == conf.shape
    assert torch.isfinite(adv).all()
    # padding tokens of the short sequence carry no advantage
    assert torch.count_nonzero(adv[0, 4:]) == 0


# ── chunked second-forward (memory-safe confidence) ─────────────────────────

class _FakeLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Per-token deterministic logits via a fixed embedding-like projection.
    Each row's output depends only on its own token ids, so micro-batching
    over the batch dim must be exactly equivalent to a single forward."""
    def __init__(self, vocab_in, V, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(vocab_in, V, generator=g)

    def __call__(self, input_ids, attention_mask=None, logits_to_keep=None):
        return _FakeLMOutput(self.W[input_ids])  # (b, L, V)


def _ref_confidence(model, input_ids, logits_to_keep, conf_fn, top_k=20):
    full = model(input_ids=input_ids).logits[:, :-1, :]
    full = full[:, -logits_to_keep:, :]
    return conf_fn(full, top_k=top_k)


def test_confidence_chunked_matches_single_forward():
    from src.confidence_utils import (
        confidence_from_logits as cf,
        confidence_from_model_chunked as chunked,
    )
    B, L, Vin, V = 6, 9, 30, 40
    logits_to_keep = 5
    g = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, Vin, (B, L), generator=g)
    attn = torch.ones(B, L)
    model = _FakeModel(Vin, V)
    ref = _ref_confidence(model, input_ids, logits_to_keep, cf)
    for mb in (1, 2, 4, B, B + 3):  # also test micro_bs >= B (single chunk)
        out = chunked(model, input_ids, attn, logits_to_keep, top_k=20, micro_bs=mb)
        assert out.shape == (B, logits_to_keep)
        assert torch.allclose(out, ref, atol=1e-6), f"mismatch at micro_bs={mb}"


def test_confidence_chunked_matches_single_forward_ema_module():
    from src.ema_flipped_utils import (
        confidence_from_logits as cf,
        confidence_from_model_chunked as chunked,
    )
    B, L, Vin, V = 5, 8, 25, 35
    logits_to_keep = 4
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, Vin, (B, L), generator=g)
    attn = torch.ones(B, L)
    model = _FakeModel(Vin, V, seed=1)
    ref = _ref_confidence(model, input_ids, logits_to_keep, cf)
    for mb in (1, 2, 3, B):
        out = chunked(model, input_ids, attn, logits_to_keep, top_k=20, micro_bs=mb)
        assert out.shape == (B, logits_to_keep)
        assert torch.allclose(out, ref, atol=1e-6), f"mismatch at micro_bs={mb}"
