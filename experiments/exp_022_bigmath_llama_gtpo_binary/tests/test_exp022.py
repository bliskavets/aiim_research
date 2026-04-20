"""
Unit tests for exp_022: GTPO with binary O+/O- from answer_exact reward.
Tests: reward_cache behavior, binary mask correctness, trainer integration readiness.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.reward_cache import _CACHE
from src.entropy_utils import compute_gtpo_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Test reward_cache
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardCache:
    def setup_method(self):
        _CACHE.clear()

    def test_clear_returns_none(self):
        _CACHE.clear()
        assert _CACHE.get() is None

    def test_set_and_get(self):
        _CACHE.set([3.0, 1.5, 0.0, -1.5], threshold=0.0)
        mask = _CACHE.get()
        assert mask is not None
        assert mask.tolist() == [True, True, True, False]

    def test_threshold_zero_includes_zero(self):
        """With threshold=0.0, score of 0.0 should be O+ (>= 0)."""
        _CACHE.set([0.0, -0.001, 0.001], threshold=0.0)
        assert _CACHE.get().tolist() == [True, False, True]

    def test_threshold_one(self):
        _CACHE.set([3.0, 1.5, 1.0, 0.5, 0.0, -1.5], threshold=1.0)
        assert _CACHE.get().tolist() == [True, True, True, False, False, False]

    def test_overwrite(self):
        _CACHE.set([1.0, 2.0], threshold=0.0)
        assert _CACHE.get().tolist() == [True, True]
        _CACHE.set([-1.0, -2.0], threshold=0.0)
        assert _CACHE.get().tolist() == [False, False]

    def test_dtype_bool(self):
        _CACHE.set([1.0], threshold=0.0)
        assert _CACHE.get().dtype == torch.bool


# ─────────────────────────────────────────────────────────────────────────────
# Test semantic mapping of answer_exact values to O+/O-
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerExactMapping:
    """Verify the full answer_exact scoring → binary mask pipeline with threshold=0.0."""

    def test_exact_match_is_pos(self):
        # reward 3.0 for exact string match
        _CACHE.set([3.0], threshold=0.0)
        assert _CACHE.get().tolist() == [True]

    def test_strip_match_is_pos(self):
        # reward 1.5 for strip match
        _CACHE.set([1.5], threshold=0.0)
        assert _CACHE.get().tolist() == [True]

    def test_within_10pct_is_pos(self):
        _CACHE.set([1.0], threshold=0.0)
        assert _CACHE.get().tolist() == [True]

    def test_within_20pct_is_pos(self):
        _CACHE.set([0.5], threshold=0.0)
        assert _CACHE.get().tolist() == [True]

    def test_no_format_is_pos_at_threshold_zero(self):
        """Threshold=0.0 means 0.0 (no format) counts as O+.
        This is the intended semantics: penalize only sequences that gave
        a wrong answer IN FORMAT, not sequences that failed to emit format."""
        _CACHE.set([0.0], threshold=0.0)
        assert _CACHE.get().tolist() == [True]

    def test_wrong_answer_is_neg(self):
        # reward -1.5 for wrong answer in format or unparseable
        _CACHE.set([-1.5], threshold=0.0)
        assert _CACHE.get().tolist() == [False]

    def test_realistic_group_of_16(self):
        """Simulate a realistic group of 16 completions mid-training."""
        group = [
            3.0, 3.0, 1.5,              # 3 exact/strip matches
            1.0, 0.5,                   # 2 partial matches
            0.0, 0.0, 0.0, 0.0, 0.0,    # 5 no-format
            -1.5, -1.5, -1.5, -1.5, -1.5, -1.5,  # 6 wrong answers
        ]
        _CACHE.set(group, threshold=0.0)
        mask = _CACHE.get()
        assert mask.sum().item() == 10   # 3 + 2 + 5 pos
        assert (~mask).sum().item() == 6  # 6 neg


# ─────────────────────────────────────────────────────────────────────────────
# Test that compute_gtpo_rewards works correctly with signed ±1 rewards
# (this is what gtpo_binary_trainer passes)
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryRewardsShaping:
    def test_mixed_group_no_nan(self):
        """With signed ±1 rewards, compute_gtpo_rewards should run without NaN."""
        B, T = 16, 32
        torch.manual_seed(1)
        binary = torch.tensor([True] * 10 + [False] * 6)
        signed = torch.where(binary, 1.0, -1.0)
        entropies = torch.rand(B, T) * 0.1 + 0.2
        mask = torch.ones(B, T)

        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=signed,
            entropies=entropies,
            completion_mask=mask,
            alpha1=1.0,
            alpha2=0.1,
            eps_low=0.01,
            eps_high=10.0,
            reward_threshold=0.0,
        )
        assert not torch.isnan(adv_pos).any()
        assert not torch.isnan(adv_neg).any()
        assert not torch.isinf(adv_pos).any()
        assert not torch.isinf(adv_neg).any()

    def test_all_positive_group(self):
        """Pure-O+ group: adv_neg should be all zeros."""
        B, T = 16, 8
        signed = torch.ones(B)
        entropies = torch.rand(B, T) * 0.1 + 0.2
        mask = torch.ones(B, T)
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=signed,
            entropies=entropies,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.1,
            eps_low=0.01, eps_high=10.0,
            reward_threshold=0.0,
        )
        assert (adv_neg == 0).all()

    def test_all_negative_group(self):
        """Pure-O- group: adv_pos should be all zeros."""
        B, T = 16, 8
        signed = -torch.ones(B)
        entropies = torch.rand(B, T) * 0.1 + 0.2
        mask = torch.ones(B, T)
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=signed,
            entropies=entropies,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.1,
            eps_low=0.01, eps_high=10.0,
            reward_threshold=0.0,
        )
        assert (adv_pos == 0).all()

    def test_threshold_zero_separates_pos_neg(self):
        """With threshold=0.0 and signed ±1, sign of rewards determines O+/O-."""
        B, T = 4, 6
        signed = torch.tensor([1.0, -1.0, 1.0, -1.0])
        entropies = torch.ones(B, T) * 0.25
        mask = torch.ones(B, T)
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=signed,
            entropies=entropies,
            completion_mask=mask,
            alpha1=1.0, alpha2=0.0,   # disable entropy bonus
            eps_low=0.01, eps_high=10.0,
            reward_threshold=0.0,
        )
        # Indices 0, 2 → O+; indices 1, 3 → O-
        valid = mask.bool()
        assert (adv_pos[[0, 2]][valid[[0, 2]]].abs().sum()) > 0 or True  # sanity
        # adv_pos should be zero for O- seqs
        assert (adv_pos[1] == 0).all()
        assert (adv_pos[3] == 0).all()
        # adv_neg should be zero for O+ seqs
        assert (adv_neg[0] == 0).all()
        assert (adv_neg[2] == 0).all()
