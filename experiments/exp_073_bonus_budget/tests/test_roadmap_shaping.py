"""CPU unit tests for the exp_071–075 roadmap shaping functions. No training/GPU."""
import unsloth  # noqa: F401  (import order: unsloth before trl-adjacent imports)
import math
import torch
from src.roadmap_shaping import (
    group_has_signal, head_branching_from_sorted,
    branch_advantages, flipped_budget_advantages, surprisal_advantages,
)


# ── 071: zero-variance gate ─────────────────────────────────────────────
def test_gate_zero_variance():
    assert not group_has_signal(torch.zeros(4))            # std(R)=0 -> TRL adv all 0
    assert group_has_signal(torch.tensor([1.0, -0.5, -0.5, 0.0]))


# ── 072: branching signal ───────────────────────────────────────────────
def _sorted_lp(probs):
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.log(p).sort(descending=True).values.unsqueeze(0)


def test_branching_bounds_and_direction():
    peaked = head_branching_from_sorted(_sorted_lp([0.997, 1e-3, 1e-3, 5e-4, 5e-4]))
    flat = head_branching_from_sorted(_sorted_lp([0.2, 0.2, 0.2, 0.2, 0.2]))
    assert 0.0 <= peaked.item() < 0.1                      # near-deterministic head -> h≈0
    assert flat.item() > 0.999                             # uniform head -> h=1
    mid = head_branching_from_sorted(_sorted_lp([0.5, 0.3, 0.1, 0.06, 0.04]))
    assert 0.0 < mid.item() < 1.0


def test_branch_advantages_direction():
    # 2 correct + 2 wrong rollouts, T=3, all active
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    mask = torch.ones(4, 3)
    h = torch.tensor([[0.9, 0.9, 0.9],                     # correct, branching
                      [0.1, 0.1, 0.1],                     # correct, peaked
                      [0.9, 0.9, 0.9],                     # wrong, branching
                      [0.1, 0.1, 0.1]])                    # wrong, peaked
    shaped = branch_advantages(adv, h, mask, alpha1=0.9, alpha2=0.1)
    # O+: branching rollout out-earns peaked one (bonus ∝ h)
    assert shaped[0].mean() > shaped[1].mean()
    # O−: peaked wrong rollout punished harder (penalty ∝ 1−h) -> more negative
    assert shaped[3].mean() < shaped[2].mean()
    # polarity signs preserved
    assert shaped[0].mean() > 0 and shaped[3].mean() < 0


# ── 073: length-invariant budget ────────────────────────────────────────
def test_budget_row_sums_equal():
    # O+ rollouts of very different lengths; without budget the long one harvests more
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    mask = torch.zeros(4, 8)
    mask[0, :8] = 1                                        # long correct
    mask[1, :2] = 1                                        # short correct
    mask[2, :6] = 1; mask[3, :3] = 1                       # wrong pair
    sig = torch.rand(4, 8) * 3 + 1.0                       # positive C-like signal
    _, bonus, pen = flipped_budget_advantages(adv, sig, mask, return_parts=True)
    # per-rollout harvested bonus mass is equal within the polarity
    assert torch.allclose(bonus[0].sum(), bonus[1].sum(), atol=1e-4)
    assert torch.allclose(pen[2].sum(), pen[3].sum(), atol=1e-4)
    # and equals the polarity's mean active length
    assert torch.allclose(bonus[0].sum(), torch.tensor((8 + 2) / 2.0), atol=1e-4)


def test_budget_vs_nobudget_length_incentive():
    # sanity: WITHOUT budget the long rollout's bonus sum exceeds the short one's
    adv = torch.tensor([1.0, 1.0, -1.0])
    mask = torch.zeros(3, 8); mask[0, :8] = 1; mask[1, :2] = 1; mask[2, :4] = 1
    h = torch.full((3, 8), 0.5)
    _, bonus, _ = branch_advantages(adv, h, mask, budget=False, return_parts=True)
    assert bonus[0].sum() > bonus[1].sum() * 2             # ∝ length
    _, bonus_b, _ = branch_advantages(adv, h, mask, budget=True, return_parts=True)
    assert torch.allclose(bonus_b[0].sum(), bonus_b[1].sum(), atol=1e-4)


# ── 074: surprisal credit ───────────────────────────────────────────────
def test_surprisal_directions():
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    mask = torch.ones(4, 2)
    s = torch.tensor([[3.0, 3.0],                          # correct, surprising
                      [0.1, 0.1],                          # correct, confident
                      [3.0, 3.0],                          # wrong, surprising (exploratory)
                      [0.1, 0.1]])                         # wrong, confident
    shaped = surprisal_advantages(adv, s, mask, alpha2=0.1)
    # O+: surprising tokens get extra credit over confident ones
    assert shaped[0].mean() > shaped[1].mean()
    # O−: confident wrong tokens punished harder than exploratory wrong ones
    assert shaped[3].mean() < shaped[2].mean()
    # additive on base: O+ stays positive, O− stays negative
    assert shaped[0].mean() > 0 and shaped[3].mean() < 0


def test_surprisal_cold_start_reduces_to_grpo():
    # constant surprisal -> z=0 -> exactly the GRPO base
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3)
    s = torch.full((2, 3), 2.0)
    shaped = surprisal_advantages(adv, s, mask, alpha2=0.1)
    assert torch.allclose(shaped, adv.unsqueeze(1) * mask, atol=1e-5)


if __name__ == "__main__":
    test_gate_zero_variance(); test_branching_bounds_and_direction()
    test_branch_advantages_direction(); test_budget_row_sums_equal()
    test_budget_vs_nobudget_length_incentive(); test_surprisal_directions()
    test_surprisal_cold_start_reduces_to_grpo()
    print("all roadmap shaping tests passed")
