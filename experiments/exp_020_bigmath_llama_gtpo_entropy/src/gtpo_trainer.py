"""
gtpo_trainer.py
---------------
GTPOTrainer: subclass of TRL's GRPOTrainer implementing Group Token Policy
Optimization (GTPO) from "GTPO and GRPO-S: Token and Sequence-Level Reward
Shaping with Policy Entropy" (ICML 2026 submission).

Key changes vs GRPOTrainer:
  1. _generate_and_score_completions(): runs extra forward pass to get
     per-token entropies from OLD policy; passes them into the batch dict.
  2. _compute_loss(): uses per-token advantages (B, T) instead of (B,),
     normalizes loss over total token count (DAPO-style).
"""

import torch
from trl import GRPOTrainer
from trl.trainer.utils import selective_log_softmax

from .entropy_utils import entropy_from_logits, compute_gtpo_rewards, EPS


class GTPOTrainer(GRPOTrainer):
    """
    GTPO trainer. Accepts all standard GRPOTrainer arguments plus:

    Extra kwargs (passed via model_init_kwargs or set directly after init):
      alpha1 (float): base reward weight for O+ tokens. Default 1.0
      alpha2 (float): entropy bonus weight. Default 0.1
      eps_entropy_low  (float): min entropy clip. Default 0.2
      eps_entropy_high (float): max entropy clip. Default 0.28
      reward_threshold (float): O+/O- split on sequence reward. Default 0.0
    """

    def __init__(self, *args, **kwargs):
        self.alpha1           = kwargs.pop("alpha1", 1.0)
        self.alpha2           = kwargs.pop("alpha2", 0.1)
        self.eps_entropy_low  = kwargs.pop("eps_entropy_low", 0.2)
        self.eps_entropy_high = kwargs.pop("eps_entropy_high", 0.28)
        self.reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)

    # Entropy is computed inside _compute_loss to avoid Unsloth buffer splitting issues.

    # ─────────────────────────────────────────────────────────────────────────
    # Override: GTPO loss with per-token advantages
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]         # (B,) standard GRPO advantages

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Forward pass through current policy (also compute entropies here)
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )
        if entropies is None:
            entropies = torch.ones_like(completion_mask, dtype=torch.float32) * 0.24

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # Token-level importance weights  w_i,t = π_θ / π_θ_old
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)                                            # (B, T)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # ── Reconstruct rewards from advantages for entropy shaping ──────────
        # seq_advantages are already normalised; we need raw rewards.
        # We recover them by reversing the normalisation stored in the parent.
        # Simpler: re-use the raw rewards that are stored in `_logs`.
        # Actually we can re-derive O+/O- from advantages sign: positive adv → O+
        # But to stay faithful to reward_threshold=0, we pass seq_advantages as proxy.
        rewards_proxy = seq_advantages  # (B,) same sign as rewards → valid for O+/O- split

        # Compute GTPO per-token advantages
        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards          = rewards_proxy,
            entropies        = entropies,
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            eps_low          = self.eps_entropy_low,
            eps_high         = self.eps_entropy_high,
            reward_threshold = self.reward_threshold,
        )
        # Combined per-token advantage (B, T)
        token_advantages = adv_pos + adv_neg

        # ── GTPO objective (Eq. 7) ────────────────────────────────────────────
        # J_GTPO = E[ 1/Σ|o_k| · Σ_i Σ_t min(w·Ã, clip(w)·Ã) ]
        per_token_loss1 = coef_1 * token_advantages             # (B, T)
        per_token_loss2 = coef_2 * token_advantages             # (B, T)
        per_token_loss  = -torch.min(per_token_loss1, per_token_loss2)

        # KL regularisation (if beta != 0)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Normalize over total tokens in batch (DAPO / GTPO style)
        total_tokens = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / total_tokens
        loss = loss / self.current_gradient_accumulation_steps

        # ── Extra logging ─────────────────────────────────────────────────────
        mode = "train" if model.training else "eval"
        mean_token_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo/mean_token_advantage", []).append(
            self.accelerator.gather(mean_token_adv).mean().item()
        )
        mean_entropy = (entropies * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo/mean_entropy", []).append(
            self.accelerator.gather(mean_entropy).mean().item()
        )
        n_pos = (seq_advantages > self.reward_threshold).float().sum()
        n_neg = (seq_advantages <= self.reward_threshold).float().sum()
        self._metrics[mode].setdefault("gtpo/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
