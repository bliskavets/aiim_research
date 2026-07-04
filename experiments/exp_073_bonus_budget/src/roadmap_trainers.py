"""
roadmap_trainers.py — trainers for the exp_071–075 roadmap setups.
All inherit GroupShapedBase (group-visible FIXED pattern). Base config everywhere:
λ=0.7, top_k=5, α₁=0.9, α₂=0.1 — matching the current best
(gtpo_ema_flipped FIXED + pos_discount λ0.7 k5) so each setup changes ONE thing.
"""
import torch
from .novel_trainers import GroupShapedBase
from .ema_flipped_utils import compute_ema_vectorized
from .novel_shaping import flipped_advantages, position_discount, token_logprobs_chunked
from .roadmap_shaping import (
    group_has_signal, head_branching_from_model_chunked,
    branch_advantages, flipped_budget_advantages, surprisal_advantages,
)


class _RoadmapBase(GroupShapedBase):
    """Shared helpers: gate bookkeeping + position discount grid."""

    def _log(self, key, val):
        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault(f"{self.tag}/{key}", []).append(val)

    def _gate(self, out):
        """Zero-variance gate: return GRPO advantages (zeros) when the group has no
        terminal signal; None when shaping should proceed."""
        if group_has_signal(out["advantages"]):
            self._log("gated", 0.0)
            return None
        self._log("gated", 1.0)
        cm = out["completion_mask"]
        return out["advantages"].unsqueeze(1) * cm

    def _posdisc(self, cm):
        T = cm.size(1)
        return position_discount(T, self.pos_tau, cm.device).unsqueeze(0).expand(cm.size(0), T)


class ZVGatePosdiscTrainer(_RoadmapBase):
    """exp_071: current best (posdisc λ0.7 k5) + zero-variance gate. When std(R)=0
    within the group, fall back to plain-GRPO zeros instead of sending every rollout
    to O− (which injects correctness-free penalty noise — the omnimath failure)."""
    tag = "zvgate"

    def _token_advantage(self, out):
        gated = self._gate(out)
        if gated is not None:
            return gated
        cids, cm, _, _, _ = self._grid(out)
        ema = compute_ema_vectorized(self._confidence(out), cm, lam=self.lam)
        return flipped_advantages(out["advantages"], ema, cm, self.alpha1, self.alpha2,
                                  reward_threshold=self.reward_threshold,
                                  bonus_mult=self._posdisc(cm))


class BranchEntropyTrainer(_RoadmapBase):
    """exp_072: bounded branching signal h = H(renormalized top-k head)/log k instead
    of C. O+ bonus ∝ EMA(h) (reward branch points), O− penalty ∝ 1−EMA(h) (blame
    peaked wrong tokens). No reciprocal → bounded by construction. posdisc kept."""
    tag = "branch_entropy"

    def _token_advantage(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        h = head_branching_from_model_chunked(
            self.model, input_ids, attn, ltk, top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)
        h_ema = compute_ema_vectorized(h, cm, lam=self.lam).clamp(0.0, 1.0)
        valid = cm.bool()
        if valid.any():
            self._log("mean_h", h[valid].mean().item())
        return branch_advantages(out["advantages"], h_ema, cm, self.alpha1, self.alpha2,
                                 reward_threshold=self.reward_threshold,
                                 bonus_mult=self._posdisc(cm))


class BudgetFlippedTrainer(_RoadmapBase):
    """exp_073: current best signal (1/EMA(C) bonus, EMA(C) penalty) but the α₂ mass a
    rollout can harvest is made LENGTH-INVARIANT (per-rollout Σ_t bonus = polarity mean
    active length) — replaces the position discount as the anti-length-farming device."""
    tag = "flipped_budget"

    def _token_advantage(self, out):
        cids, cm, _, _, _ = self._grid(out)
        ema = compute_ema_vectorized(self._confidence(out), cm, lam=self.lam)
        return flipped_budget_advantages(out["advantages"], ema, cm, self.alpha1,
                                         self.alpha2, reward_threshold=self.reward_threshold)


class SurprisalCreditTrainer(_RoadmapBase):
    """exp_074: minimal-machinery variant — additive per-polarity z-scored surprisal
    of the REALIZED token (no top-k forward; one logprob gather). Reward surprising
    tokens in correct rollouts, punish confident tokens in wrong ones. posdisc kept."""
    tag = "surprisal_credit"

    def _token_advantage(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        lp = token_logprobs_chunked(
            self.model, input_ids, attn, ltk, cids,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)
        s = (-lp) * cm                                     # surprisal of sampled token
        valid = cm.bool()
        if valid.any():
            self._log("mean_s", s[valid].mean().item())
        return surprisal_advantages(out["advantages"], s, cm, self.alpha2,
                                    reward_threshold=self.reward_threshold,
                                    bonus_mult=self._posdisc(cm))


class FinalComboTrainer(_RoadmapBase):
    """exp_075: the paper candidate — zero-variance gate + branching signal +
    length-invariant budget (budget replaces posdisc), λ=0.7, k=5."""
    tag = "final_combo"

    def _token_advantage(self, out):
        gated = self._gate(out)
        if gated is not None:
            return gated
        cids, cm, input_ids, attn, ltk = self._grid(out)
        h = head_branching_from_model_chunked(
            self.model, input_ids, attn, ltk, top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)
        h_ema = compute_ema_vectorized(h, cm, lam=self.lam).clamp(0.0, 1.0)
        valid = cm.bool()
        if valid.any():
            self._log("mean_h", h[valid].mean().item())
        return branch_advantages(out["advantages"], h_ema, cm, self.alpha1, self.alpha2,
                                 reward_threshold=self.reward_threshold, budget=True)


ROADMAP_TRAINERS = {
    "zvgate": ZVGatePosdiscTrainer,
    "branch_entropy": BranchEntropyTrainer,
    "flipped_budget": BudgetFlippedTrainer,
    "surprisal_credit": SurprisalCreditTrainer,
    "final_combo": FinalComboTrainer,
}
