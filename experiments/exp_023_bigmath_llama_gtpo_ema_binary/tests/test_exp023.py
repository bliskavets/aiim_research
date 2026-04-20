"""
Unit tests for exp_023: GTPO-EMA with binary O+/O- from answer_exact reward.
Tests: reward_cache, binary mask, compute_gtpo_ema_advantages with signed rewards.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.reward_cache import _CACHE
from src.ema_confidence_utils import (
    confidence_from_logits,
    compute_gtpo_ema_advantages,
    compute_ema_vectorized,
)


class TestRewardCache:
    def setup_method(self):
        _CACHE.clear()

    def test_set_get_clear(self):
        assert _CACHE.get() is None
        _CACHE.set([1.5, -1.0], threshold=0.0)
        assert _CACHE.get().tolist() == [True, False]
        _CACHE.clear()
        assert _CACHE.get() is None

    def test_threshold_zero_includes_zero(self):
        _CACHE.set([0.0, -0.001], threshold=0.0)
        assert _CACHE.get().tolist() == [True, False]


class TestAnswerExactMapping:
    def test_all_reward_values(self):
        """Maps all possible answer_exact scores correctly with threshold=0."""
        scores = [3.0, 1.5, 1.0, 0.5, 0.0, -1.5]
        _CACHE.set(scores, threshold=0.0)
        expected = [True, True, True, True, True, False]
        assert _CACHE.get().tolist() == expected


class TestBinaryRewardsWithEMA:
    def test_signed_rewards_no_nan(self):
        """Signed ±1 rewards should not produce NaN in EMA advantages."""
        B, T, V = 16, 32, 64
        torch.manual_seed(0)
        logits = torch.randn(B, T, V)
        confidence = confidence_from_logits(logits, top_k=20)
        mask = torch.ones(B, T)
        binary = torch.tensor([True] * 10 + [False] * 6)
        signed = torch.where(binary, 1.0, -1.0)

        adv = compute_gtpo_ema_advantages(
            rewards=signed,
            confidence=confidence,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.1, lam=0.9,
            reward_threshold=0.0,
        )
        assert adv.shape == (B, T)
        assert not torch.isnan(adv).any()
        assert not torch.isinf(adv).any()

    def test_all_pos_group_base_adv_zero(self):
        """When all rewards are +1, base_adv = (r - mean)/std = 0 since mean=1, r=1."""
        B, T, V = 8, 10, 32
        torch.manual_seed(1)
        logits = torch.randn(B, T, V)
        confidence = confidence_from_logits(logits, top_k=10)
        mask = torch.ones(B, T)
        signed = torch.ones(B)

        # alpha2=0 to isolate base term
        adv = compute_gtpo_ema_advantages(
            rewards=signed,
            confidence=confidence,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.0, lam=0.9,
            reward_threshold=0.0,
        )
        valid = mask.bool()
        assert torch.allclose(adv[valid], torch.zeros_like(adv[valid]), atol=1e-4)

    def test_mixed_group_nonzero_advantages(self):
        """Mixed O+/O-: advantages should be nonzero at valid positions."""
        B, T, V = 8, 10, 32
        torch.manual_seed(2)
        logits = torch.randn(B, T, V)
        confidence = confidence_from_logits(logits, top_k=10)
        mask = torch.ones(B, T)
        signed = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

        adv = compute_gtpo_ema_advantages(
            rewards=signed,
            confidence=confidence,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.1, lam=0.9,
            reward_threshold=0.0,
        )
        assert adv[mask.bool()].abs().sum() > 0

    def test_zero_tokens_at_padding(self):
        """Padding positions must have zero advantage."""
        B, T, V = 4, 8, 16
        torch.manual_seed(3)
        logits = torch.randn(B, T, V)
        confidence = confidence_from_logits(logits, top_k=4)
        mask = torch.ones(B, T)
        mask[2:, T // 2:] = 0.0  # last 2 seqs have shorter length
        signed = torch.tensor([1.0, -1.0, 1.0, -1.0])

        adv = compute_gtpo_ema_advantages(
            rewards=signed,
            confidence=confidence,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.1, lam=0.9,
            reward_threshold=0.0,
        )
        padding = ~mask.bool()
        assert (adv[padding] == 0).all()


class TestCacheShapeMatch:
    def test_cache_shape_matches_batch(self):
        """Cache mask must be len=B*G for trainer to accept it."""
        G_times_B = 16 * 4
        scores = [1.0] * G_times_B
        _CACHE.set(scores, threshold=0.0)
        assert _CACHE.get().shape[0] == G_times_B
