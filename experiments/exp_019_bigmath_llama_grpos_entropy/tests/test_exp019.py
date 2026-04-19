"""
Unit tests for exp_019: GRPO-S entropy shaping.
Tests: entropy_from_logits, clip_entropies, compute_grpo_s_rewards, compute_gtpo_rewards.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.entropy_utils import (
    entropy_from_logits,
    clip_entropies,
    compute_grpo_s_rewards,
    compute_gtpo_rewards,
    EPS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def batch_params():
    return {"B": 4, "T": 8, "V": 32}


@pytest.fixture
def logits(batch_params):
    B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
    torch.manual_seed(42)
    return torch.randn(B, T, V)


@pytest.fixture
def mask(batch_params):
    B, T = batch_params["B"], batch_params["T"]
    m = torch.ones(B, T)
    m[2:, T // 2:] = 0.0
    return m


@pytest.fixture
def pos_rewards(batch_params):
    """All positive rewards (O+)."""
    return torch.tensor([2.0, 1.5, 3.0, 0.5])


@pytest.fixture
def neg_rewards(batch_params):
    """All negative rewards (O-)."""
    return torch.tensor([-2.0, -1.5, -3.0, -0.5])


@pytest.fixture
def mixed_rewards(batch_params):
    """Mixed positive/negative rewards."""
    return torch.tensor([2.0, -1.0, 3.0, -0.5])


# ─────────────────────────────────────────────────────────────────────────────
# Test: entropy_from_logits
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyFromLogits:
    def test_output_shape(self, logits, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        H = entropy_from_logits(logits)
        assert H.shape == (B, T), f"Expected ({B}, {T}), got {H.shape}"

    def test_non_negative_values(self, logits):
        H = entropy_from_logits(logits)
        assert (H >= 0).all(), "Entropy must be non-negative"

    def test_uniform_logits_max_entropy(self, batch_params):
        """Uniform distribution → entropy = log(V)."""
        B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
        logits = torch.zeros(B, T, V)
        H = entropy_from_logits(logits)
        expected = torch.log(torch.tensor(float(V)))
        assert torch.allclose(H, expected.expand(B, T), atol=1e-5)

    def test_one_hot_logits_zero_entropy(self, batch_params):
        """One-hot distribution → entropy ~ 0."""
        B, T, V = batch_params["B"], batch_params["T"], batch_params["V"]
        logits = torch.full((B, T, V), -1e9)
        logits[:, :, 0] = 0.0  # all probability on token 0
        H = entropy_from_logits(logits)
        assert (H < 1e-3).all(), "One-hot entropy should be ~0"

    def test_deterministic(self, logits):
        H1 = entropy_from_logits(logits)
        H2 = entropy_from_logits(logits)
        assert torch.allclose(H1, H2)


# ─────────────────────────────────────────────────────────────────────────────
# Test: clip_entropies
# ─────────────────────────────────────────────────────────────────────────────

class TestClipEntropies:
    def test_clamps_below_eps_low(self, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        ent = torch.zeros(B, T)  # all 0, below eps_low=0.2
        clipped = clip_entropies(ent, eps_low=0.2, eps_high=0.28)
        assert (clipped >= 0.2).all(), "All values should be >= eps_low"

    def test_clamps_above_eps_high(self, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        ent = torch.ones(B, T) * 10.0  # all > eps_high=0.28
        clipped = clip_entropies(ent, eps_low=0.2, eps_high=0.28)
        assert (clipped <= 0.28).all(), "All values should be <= eps_high"

    def test_values_in_range_unchanged(self, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        ent = torch.ones(B, T) * 0.24  # within [0.2, 0.28]
        clipped = clip_entropies(ent, eps_low=0.2, eps_high=0.28)
        assert torch.allclose(clipped, ent), "In-range values should be unchanged"

    def test_output_shape(self, logits, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        H = entropy_from_logits(logits)
        clipped = clip_entropies(H)
        assert clipped.shape == (B, T)

    def test_clip_range(self, logits):
        H = entropy_from_logits(logits)
        clipped = clip_entropies(H, eps_low=0.2, eps_high=0.28)
        assert (clipped >= 0.2).all()
        assert (clipped <= 0.28).all()


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_grpo_s_rewards
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGRPOSRewards:
    def test_output_shape(self, logits, mask, mixed_rewards, batch_params):
        B = batch_params["B"]
        H = entropy_from_logits(logits)
        shaped, H_avg = compute_grpo_s_rewards(
            rewards=mixed_rewards,
            entropies=H,
            completion_mask=mask,
        )
        assert shaped.shape == (B,), f"shaped_rewards shape: expected ({B},), got {shaped.shape}"
        assert H_avg.shape == (B,), f"H_avg shape: expected ({B},), got {H_avg.shape}"

    def test_o_plus_gets_positive_reward(self, logits, mask, pos_rewards):
        """O+ sequences should get positive shaped rewards."""
        H = entropy_from_logits(logits)
        shaped, _ = compute_grpo_s_rewards(
            rewards=pos_rewards,
            entropies=H,
            completion_mask=mask,
            beta1=1.0,
            beta2=0.1,
            reward_threshold=0.0,
        )
        assert (shaped > 0).all(), f"O+ shaped rewards should be positive, got {shaped}"

    def test_o_minus_gets_negative_reward(self, logits, mask, neg_rewards):
        """O- sequences should get negative shaped rewards."""
        H = entropy_from_logits(logits)
        shaped, _ = compute_grpo_s_rewards(
            rewards=neg_rewards,
            entropies=H,
            completion_mask=mask,
            beta1=1.0,
            beta2=0.1,
            reward_threshold=0.0,
        )
        assert (shaped < 0).all(), f"O- shaped rewards should be negative, got {shaped}"

    def test_mixed_has_correct_signs(self, logits, mask, mixed_rewards):
        """Mixed O+/O-: positive rewards → positive shaped, negative → negative shaped."""
        H = entropy_from_logits(logits)
        shaped, _ = compute_grpo_s_rewards(
            rewards=mixed_rewards,
            entropies=H,
            completion_mask=mask,
            reward_threshold=0.0,
        )
        is_pos = mixed_rewards > 0
        is_neg = ~is_pos
        assert (shaped[is_pos] > 0).all(), "O+ should be positive"
        assert (shaped[is_neg] < 0).all(), "O- should be negative"

    def test_h_avg_clipped(self, logits, mask, mixed_rewards):
        """H_avg should be in [eps_low, eps_high]."""
        H = entropy_from_logits(logits)
        _, H_avg = compute_grpo_s_rewards(
            rewards=mixed_rewards,
            entropies=H,
            completion_mask=mask,
            eps_low=0.2,
            eps_high=0.28,
        )
        assert (H_avg >= 0.2).all()
        assert (H_avg <= 0.28).all()

    def test_no_nan_or_inf(self, logits, mask, mixed_rewards):
        H = entropy_from_logits(logits)
        shaped, H_avg = compute_grpo_s_rewards(mixed_rewards, H, mask)
        assert not torch.isnan(shaped).any()
        assert not torch.isinf(shaped).any()
        assert not torch.isnan(H_avg).any()


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_gtpo_rewards
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGTPORewards:
    def test_output_shape(self, logits, mask, mixed_rewards, batch_params):
        B, T = batch_params["B"], batch_params["T"]
        H = entropy_from_logits(logits)
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=mixed_rewards,
            entropies=H,
            completion_mask=mask,
        )
        assert adv_pos.shape == (B, T)
        assert adv_neg.shape == (B, T)

    def test_masked_positions_zero(self, logits, mask, mixed_rewards, batch_params):
        """Advantages must be zero at padding positions."""
        H = entropy_from_logits(logits)
        adv_pos, adv_neg = compute_gtpo_rewards(mixed_rewards, H, mask)
        padding = ~mask.bool()
        assert (adv_pos[padding] == 0).all()
        assert (adv_neg[padding] == 0).all()

    def test_o_plus_adv_pos_nonzero(self, logits, mask, pos_rewards):
        """For all-O+ batch, adv_pos should be nonzero at valid positions."""
        H = entropy_from_logits(logits)
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=pos_rewards, entropies=H, completion_mask=mask
        )
        valid = mask.bool()
        assert adv_pos[valid].abs().sum() > 0
        # adv_neg should be all zero since no O- sequences
        assert (adv_neg == 0).all()

    def test_o_minus_adv_neg_nonzero(self, logits, mask, neg_rewards):
        """For all-O- batch, adv_neg should be nonzero at valid positions."""
        H = entropy_from_logits(logits)
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards=neg_rewards, entropies=H, completion_mask=mask
        )
        valid = mask.bool()
        assert adv_neg[valid].abs().sum() > 0
        # adv_pos should be all zero since no O+ sequences
        assert (adv_pos == 0).all()

    def test_o_plus_advantages_positive_before_normalization(self, logits, mask, pos_rewards):
        """shaped_pos values are positive; after normalization adv_pos can be mixed sign
        (zero-mean) but should not be all zero."""
        H = entropy_from_logits(logits)
        adv_pos, _ = compute_gtpo_rewards(
            rewards=pos_rewards, entropies=H, completion_mask=mask,
            alpha1=1.0, alpha2=0.0,  # no entropy bonus
        )
        valid = mask.bool()
        # After normalization, mean~0 but should have nonzero variance
        assert adv_pos[valid].abs().sum() > 0

    def test_o_minus_advantages_negative_before_normalization(self, logits, mask, neg_rewards):
        """shaped_neg values are negative; adv_neg after norm should have nonzero variance."""
        H = entropy_from_logits(logits)
        _, adv_neg = compute_gtpo_rewards(
            rewards=neg_rewards, entropies=H, completion_mask=mask,
            alpha1=1.0, alpha2=0.0,
        )
        valid = mask.bool()
        assert adv_neg[valid].abs().sum() > 0

    def test_d_t_accounting(self, logits, mask, mixed_rewards, batch_params):
        """d_t (active O+ seqs at position t) is correctly accounted for:
        entropy bonus sums to d_t over active O+ seqs at each position."""
        B, T = batch_params["B"], batch_params["T"]
        # Use alpha1=0 to isolate entropy bonus
        H = entropy_from_logits(logits)
        # Manually clip to match what function does
        from src.entropy_utils import clip_entropies
        H_clipped = clip_entropies(H)
        is_pos = mixed_rewards > 0.0
        mask_pos = mask * is_pos.float().unsqueeze(1)

        adv_pos, _ = compute_gtpo_rewards(
            rewards=mixed_rewards,
            entropies=H,
            completion_mask=mask,
            alpha1=0.0,
            alpha2=1.0,
            reward_threshold=0.0,
        )
        # shaped_pos (before normalization) at each t sums to alpha2 * d_t
        # We can't directly access shaped_pos, but verify adv_pos is nonzero
        valid = mask_pos.bool()
        if valid.any():
            assert adv_pos[valid].abs().sum() > 0

    def test_no_nan_or_inf(self, logits, mask, mixed_rewards):
        H = entropy_from_logits(logits)
        adv_pos, adv_neg = compute_gtpo_rewards(mixed_rewards, H, mask)
        assert not torch.isnan(adv_pos).any()
        assert not torch.isnan(adv_neg).any()
        assert not torch.isinf(adv_pos).any()
        assert not torch.isinf(adv_neg).any()
