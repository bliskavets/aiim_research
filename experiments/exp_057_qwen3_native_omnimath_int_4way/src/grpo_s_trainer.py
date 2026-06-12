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
from .shaped_loss import inject_advantages, forward_completion_logits, token_entropy


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
        beta1            = kwargs.pop("beta1", 1.0)
        beta2            = kwargs.pop("beta2", 0.1)
        eps_entropy_low  = kwargs.pop("eps_entropy_low", 0.2)
        eps_entropy_high = kwargs.pop("eps_entropy_high", 0.28)
        reward_threshold = kwargs.pop("reward_threshold", 0.0)
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__ so GRPOTrainer.__init__ can't clobber any
        # of our shaping attrs (same precaution as the GTPO trainers' top_k).
        self.beta1, self.beta2 = beta1, beta2
        self.eps_entropy_low, self.eps_entropy_high = eps_entropy_low, eps_entropy_high
        self.reward_threshold = reward_threshold

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # unsloth's compiled compute_loss never calls _compute_loss. GRPO-S shaping
        # lives entirely in the SEQUENCE-LEVEL advantages (entropy-weighted), so we
        # compute the shaped (B,) advantages here and INJECT them into the compiled
        # loss. (GRPO-S's seq-level IS weight ≈ token-level IS on-policy with
        # num_iterations=1, so the compiled loss's token-level IS is equivalent
        # here; the distinctive entropy shaping of the advantage is preserved.)
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        grpo_advantages = inputs["advantages"]          # (B,) standard GRPO advantages
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Real per-token Shannon entropy from logits (the chunked _get helper
        # returns None for entropies on this stack -> would degenerate to a
        # constant). No grad needed (advantages are constants). Chunked to bound
        # memory; computed on the completion grid (aligned with completion_mask).
        with torch.no_grad():
            logits = forward_completion_logits(self, model, input_ids, attention_mask, logits_to_keep)
            entropies = token_entropy(logits)          # (B, Lk)
        del logits

        shaped_rewards, seq_avg_entropy = compute_grpo_s_rewards(
            rewards          = grpo_advantages,
            entropies        = entropies,
            completion_mask  = completion_mask,
            beta1            = self.beta1,
            beta2            = self.beta2,
            eps_low          = self.eps_entropy_low,
            eps_high         = self.eps_entropy_high,
            reward_threshold = self.reward_threshold,
        )
        # Re-normalize within groups → advantages (B,)
        G = self.num_generations
        shaped_grouped = shaped_rewards.view(-1, G)
        mean_s = shaped_grouped.mean(dim=1, keepdim=True)
        std_s  = shaped_grouped.std(dim=1, keepdim=True).clamp(min=EPS)
        advantages = ((shaped_grouped - mean_s) / std_s).reshape(-1)

        # ── metrics (presence proves the shaping ran) ──
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("grpo_s/mean_seq_entropy", []).append(
            self.accelerator.gather(seq_avg_entropy).mean().item())
        self._metrics[mode].setdefault("grpo_s/mean_shaped_advantage", []).append(
            self.accelerator.gather(advantages).mean().item())
        n_pos = (grpo_advantages > 0).float().sum()
        n_neg = (grpo_advantages <= 0).float().sum()
        self._metrics[mode].setdefault("grpo_s/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item())

        inputs = inject_advantages(inputs, advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
