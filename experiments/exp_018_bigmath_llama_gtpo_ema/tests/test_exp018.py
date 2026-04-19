"""
Unit tests for exp_018: GTPO-EMA (EMA confidence shaping).
Tests: confidence_from_logits, compute_ema_vectorized, compute_gtpo_ema_advantages.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.ema_confidence_utils import (
    confidence_from_logits,
    compute_ema_vectorized,
    compute_gtpo_ema_advantages,
    EPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_params():
    """Small batch parameters for testing."""
    return {"B": 4, "T": 8, "V": 32, "top_k": 4}


@pytest.fixture
def logits(batch_params):
    B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
    torch.manual_seed(42)
    return torch.randn(B, T, V)


@pytest.fixture
def mask(batch_params):
    B, T = batch_params["B"], batch_params["T"]
    # Simulate variable-length sequences: first 2 seqs use all T tokens,
    # last 2 seqs use only T//2 tokens
    m = torch.ones(B, T)
    m[2:, T // 2:] = 0.0
    return m


@pytest.fixture
def rewards(batch_params):
    B = batch_params["B"]
    # Half positive, half negative
    r = torch.tensor([2.0, -1.0, 3.0, -0.5])
    assert r.shape[0] == B
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Test: confidence_from_logits
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceFromLogits:
    def test_output_shape(self, logits, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        assert conf.shape == (B, T), f"Expected ({B}, {T}), got {conf.shape}"

    def test_values_non_negative(self, logits, batch_params):
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        assert (conf >= 0).all(), "Confidence values must be >= 0"

    def test_top_k_clamp(self, batch_params):
        """top_k > V should not crash; should clamp to V."""
        B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
        logits = torch.randn(B, T, V)
        conf = confidence_from_logits(logits, top_k=V + 100)
        assert conf.shape == (B, T)
        assert (conf >= 0).all()

    def test_deterministic(self, logits, batch_params):
        conf1 = confidence_from_logits(logits, top_k=batch_params["top_k"])
        conf2 = confidence_from_logits(logits, top_k=batch_params["top_k"])
        assert torch.allclose(conf1, conf2)

    def test_uniform_logits_gives_max_confidence(self, batch_params):
        """Uniform logits → all top-k log probs equal → confidence = log(V)."""
        B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
        logits = torch.zeros(B, T, V)  # uniform distribution
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        expected = torch.log(torch.tensor(float(V)))
        assert torch.allclose(conf, expected.expand(B, T), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_ema_vectorized
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEMAVectorized:
    def test_output_shape(self, logits, mask, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        ema = compute_ema_vectorized(conf, mask, lam=0.9)
        assert ema.shape == (B, T), f"Expected ({B}, {T}), got {ema.shape}"

    def test_ema_at_t0_equals_first_confidence(self, logits, mask, batch_params):
        """EMA at t=0 should equal confidence[:,0] * mask[:,0]."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        ema = compute_ema_vectorized(conf, mask, lam=0.9)
        expected = conf[:, 0] * mask[:, 0]
        assert torch.allclose(ema[:, 0], expected, atol=1e-6)

    def test_ema_values_between_min_and_max_confidence(self, logits, mask, batch_params):
        """For valid positions, EMA should be between min and max of confidence."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        ema = compute_ema_vectorized(conf, mask, lam=0.9)
        # Only check valid positions
        valid = mask.bool()
        ema_valid = ema[valid]
        conf_valid = conf[valid]
        assert (ema_valid >= conf_valid.min() - 1e-5).all(), "EMA below conf min"
        assert (ema_valid <= conf_valid.max() + 1e-5).all(), "EMA above conf max"

    def test_ema_held_constant_at_padding(self, logits, mask, batch_params):
        """At padding positions, EMA should equal the last valid EMA value."""
        B, T = batch_params["B"], batch_params["T"]
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        ema = compute_ema_vectorized(conf, mask, lam=0.9)
        # Check seqs with padding (seqs 2 and 3 have padding at T//2 onwards)
        half = T // 2
        for i in [2, 3]:
            last_valid_ema = ema[i, half - 1]
            for t in range(half, T):
                assert torch.allclose(
                    ema[i, t], last_valid_ema, atol=1e-6
                ), f"EMA not held at padding: seq {i}, t={t}"

    def test_lambda_zero_ema_equals_confidence(self, logits, mask, batch_params):
        """With lam=0.0, EMA = current confidence at each step."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        ema = compute_ema_vectorized(conf, mask, lam=0.0)
        # At valid positions, ema[:,t] should equal conf[:,t]
        valid = mask.bool()
        assert torch.allclose(ema[valid], conf[valid], atol=1e-6)

    def test_lambda_one_ema_equals_first_value(self, logits, mask, batch_params):
        """With lam=1.0, EMA always equals the first confidence value."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        ema = compute_ema_vectorized(conf, mask, lam=1.0)
        first = ema[:, 0].unsqueeze(1).expand_as(ema)
        valid = mask.bool()
        assert torch.allclose(ema[valid], first[valid], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_gtpo_ema_advantages
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGTPOEMAAdvantages:
    def test_output_shape(self, logits, mask, rewards, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.1,
            lam=0.9,
            reward_threshold=0.0,
        )
        assert adv.shape == (B, T), f"Expected ({B}, {T}), got {adv.shape}"

    def test_zeros_at_masked_positions(self, logits, mask, rewards, batch_params):
        """Token advantages must be zero at padding positions."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.1,
            lam=0.9,
            reward_threshold=0.0,
        )
        padding = ~mask.bool()
        assert (adv[padding] == 0.0).all(), "Non-zero advantages at padding positions"

    def test_o_plus_sequences_have_positive_base(self, logits, mask, batch_params):
        """O+ sequences (reward > threshold) should have nonzero advantages at valid positions."""
        B, T = batch_params["B"], batch_params["T"]
        # All positive rewards → all O+
        pos_rewards = torch.tensor([2.0, 1.0, 3.0, 0.5])
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=pos_rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.0,  # no EMA bonus to isolate base term
            lam=0.9,
            reward_threshold=0.0,
        )
        valid = mask.bool()
        # With alpha2=0, base advantage is group-normalized → mean~0, but not all zero
        assert adv[valid].abs().sum() > 0, "All O+ advantages are zero"

    def test_o_minus_sequences_have_negative_base(self, logits, mask, batch_params):
        """O- sequences (reward <= threshold) should have nonzero advantages at valid positions."""
        # All negative rewards → all O-
        neg_rewards = torch.tensor([-2.0, -1.0, -3.0, -0.5])
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=neg_rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.0,
            lam=0.9,
            reward_threshold=0.0,
        )
        valid = mask.bool()
        assert adv[valid].abs().sum() > 0, "All O- advantages are zero"

    def test_mixed_o_plus_o_minus(self, logits, mask, rewards, batch_params):
        """Mixed O+/O-: advantages at valid positions should not all be zero."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.1,
            lam=0.9,
            reward_threshold=0.0,
        )
        valid = mask.bool()
        assert adv[valid].abs().sum() > 0, "All advantages are zero in mixed case"

    def test_alpha2_zero_reduces_to_base_grpo(self, logits, mask, rewards, batch_params):
        """With alpha2=0, advantages should be purely group-normalized base."""
        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.0,
            lam=0.9,
            reward_threshold=0.0,
        )
        # Base advantages should be constant per sequence at valid positions
        # (all valid tokens in a seq should share the same base_adv * alpha1)
        for i in range(adv.shape[0]):
            valid_positions = mask[i].bool()
            if valid_positions.any():
                vals = adv[i, valid_positions]
                # All values in seq should be equal (same base_adv broadcast)
                assert torch.allclose(vals, vals[0].expand_as(vals), atol=1e-5), \
                    f"Base advantages not uniform in seq {i}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Basic pipeline consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineConsistency:
    def test_full_pipeline_no_crash(self, batch_params):
        """Full pipeline from logits to advantages should run without error."""
        B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        mask = torch.ones(B, T)
        rewards = torch.randn(B)

        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        assert conf.shape == (B, T)

        ema = compute_ema_vectorized(conf, mask, lam=0.9)
        assert ema.shape == (B, T)

        adv = compute_gtpo_ema_advantages(
            rewards=rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.1,
            lam=0.9,
            reward_threshold=0.0,
        )
        assert adv.shape == (B, T)
        assert not torch.isnan(adv).any(), "NaN in advantages"
        assert not torch.isinf(adv).any(), "Inf in advantages"

    def test_two_sequences(self):
        """Pipeline works with B=2 (minimum for std normalization)."""
        B, T, V = 2, 5, 16
        logits = torch.randn(B, T, V)
        mask = torch.ones(B, T)
        rewards = torch.tensor([1.5, -0.5])

        conf = confidence_from_logits(logits, top_k=4)
        ema = compute_ema_vectorized(conf, mask)
        adv = compute_gtpo_ema_advantages(rewards, conf, mask)

        assert adv.shape == (B, T)
        assert not torch.isnan(adv).any()

    def test_all_same_rewards(self, batch_params):
        """When all rewards are equal, base advantage should be zero."""
        B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
        logits = torch.randn(B, T, V)
        mask = torch.ones(B, T)
        rewards = torch.ones(B) * 2.0  # all same → std=0

        conf = confidence_from_logits(logits, top_k=batch_params["top_k"])
        adv = compute_gtpo_ema_advantages(
            rewards=rewards,
            confidence=conf,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.0,  # no EMA bonus
            lam=0.9,
            reward_threshold=0.0,
        )
        # base_adv = (r - mean) / (std + EPS) ≈ 0 when all rewards equal
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-4), \
            "Advantages should be ~0 when all rewards equal and alpha2=0"
