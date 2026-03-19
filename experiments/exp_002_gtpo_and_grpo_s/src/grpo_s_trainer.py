"""
grpo_s_trainer.py
-----------------
GRPOSTrainer: subclass of TRL's GRPOTrainer implementing Sequence-Level GRPO
(GRPO-S) from "GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with
Policy Entropy" (ICML 2026 submission).

Key changes vs GRPOTrainer:
  1. _generate_and_score_completions(): adds per-token entropies (old policy)
     and replaces standard advantages with GRPO-S shaped advantages.
  2. _compute_loss(): uses sequence-level IS weights (mean of token IS weights)
     instead of token-level IS weights.
"""

import torch
from trl import GRPOTrainer

from .entropy_utils import compute_grpo_s_rewards, EPS


class GRPOSTrainer(GRPOTrainer):
    """
    GRPO-S trainer. Accepts all standard GRPOTrainer arguments plus:

    Extra kwargs:
      beta1 (float): base reward weight. Default 1.0
      beta2 (float): entropy bonus weight. Default 0.1
      eps_entropy_low  (float): min entropy clip. Default 0.2
      eps_entropy_high (float): max entropy clip. Default 0.28
      reward_threshold (float): O+/O- split threshold. Default 0.0
    """

    def __init__(self, *args, **kwargs):
        self.beta1            = kwargs.pop("beta1", 1.0)
        self.beta2            = kwargs.pop("beta2", 0.1)
        self.eps_entropy_low  = kwargs.pop("eps_entropy_low", 0.2)
        self.eps_entropy_high = kwargs.pop("eps_entropy_high", 0.28)
        self.reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Override: replace advantages with GRPO-S shaped advantages
    # ─────────────────────────────────────────────────────────────────────────

    # No override of _generate_and_score_completions needed for GRPO-S.
    # Entropy is computed inside _compute_loss using the current model
    # (which at loss-compute time is the old policy for the first gradient step).
    # This avoids issues with Unsloth's buffered input splitting.

    # ─────────────────────────────────────────────────────────────────────────
    # Override: GRPO-S loss with sequence-level IS weights
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        grpo_advantages = inputs["advantages"]          # (B,) standard GRPO advantages

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Forward pass through current policy
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        # ── Compute entropies from current forward pass ───────────────────────
        # We recompute entropy here using current model logits as proxy for old policy.
        # This is a slight approximation but avoids Unsloth buffer splitting issues.
        with torch.no_grad():
            _, entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            )
        if entropies is None:
            # Fallback: uniform entropy approximation
            entropies = torch.ones_like(completion_mask, dtype=torch.float32) * 0.24

        # ── Compute GRPO-S shaped advantages ──────────────────────────────────
        shaped_rewards, seq_avg_entropy = compute_grpo_s_rewards(
            rewards          = grpo_advantages,   # same sign as original rewards
            entropies        = entropies,
            completion_mask  = completion_mask,
            beta1            = self.beta1,
            beta2            = self.beta2,
            eps_low          = self.eps_entropy_low,
            eps_high         = self.eps_entropy_high,
            reward_threshold = self.reward_threshold,
        )

        # Normalize within groups → advantages (B,)
        B = shaped_rewards.shape[0]
        G = self.num_generations
        shaped_grouped = shaped_rewards.view(-1, G)
        mean_s = shaped_grouped.mean(dim=1, keepdim=True)
        std_s  = shaped_grouped.std(dim=1, keepdim=True).clamp(min=EPS)
        advantages = ((shaped_grouped - mean_s) / std_s).view(B)

        # ── Sequence-level IS weight (Eq. 11) ────────────────────────────────
        # ŵ_i(θ) = (1/|o_i|) Σ_t w_i,t(θ)   where w_i,t = π_θ / π_θ_old
        log_ratio       = per_token_logps - old_per_token_logps          # (B, T)
        token_is_ratio  = torch.exp(log_ratio)                           # (B, T)
        seq_lengths     = completion_mask.sum(dim=1).clamp(min=1)        # (B,)
        seq_is_weight   = (token_is_ratio * completion_mask).sum(dim=1) / seq_lengths  # (B,)

        # ── GRPO-S objective (Eq. 10) ────────────────────────────────────────
        # J_GRPO-S = E[ 1/G · Σ_i min(ŵ_i·Â_i, clip(ŵ_i)·Â_i) ]
        coef_1 = seq_is_weight                                            # (B,)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_seq_loss1 = coef_1 * advantages                              # (B,)
        per_seq_loss2 = coef_2 * advantages                              # (B,)
        per_seq_loss  = -torch.min(per_seq_loss1, per_seq_loss2)         # (B,)

        # KL regularisation
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            # Aggregate KL to sequence level
            per_seq_kl = (per_token_kl * completion_mask).sum(dim=1) / seq_lengths
            per_seq_loss = per_seq_loss + self.beta * per_seq_kl

        # Normalize over G sequences (standard GRPO normalization)
        loss = per_seq_loss.mean()
        loss = loss / self.current_gradient_accumulation_steps

        # ── Extra logging ─────────────────────────────────────────────────────
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("grpo_s/mean_seq_is_weight", []).append(
            self.accelerator.gather(seq_is_weight).mean().item()
        )
        self._metrics[mode].setdefault("grpo_s/mean_seq_entropy", []).append(
            self.accelerator.gather(seq_avg_entropy).mean().item()
        )
        n_pos = (grpo_advantages > 0).float().sum()
        n_neg = (grpo_advantages <= 0).float().sum()
        self._metrics[mode].setdefault("grpo_s/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item()
        )

        return loss
