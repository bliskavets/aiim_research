"""
grpo_s_trainer.py
-----------------
GRPOSTrainer: Sequence-Level GRPO (GRPO-S) entropy-weighted advantage shaping,
on the Search-R1 multi-turn base.

Shaping lives entirely in the SEQUENCE-LEVEL advantages (entropy-weighted), so we
compute the shaped (B,) advantages in a `compute_loss` override and INJECT them
into unsloth's compiled loss. Overriding `_compute_loss` is silently bypassed on
the unsloth stack (the compiled compute_loss never calls it) — that made the old
GRPO-S run plain GRPO. See src/shaped_loss.py and exp_057's SHAPING_BYPASS_BUGFIX.md.

Note: the old hand-written loss used a sequence-level IS weight (mean of token IS
weights). With num_iterations=1 (on-policy) the compiled loss's token-level IS is
equivalent here; the distinctive entropy shaping of the advantage is preserved.
"""

import torch
from .searchr1_trainer import SearchR1GRPOTrainer
from .entropy_utils import compute_grpo_s_rewards, EPS
from .shaped_loss import entropy_from_model_chunked, inject_advantages


class GRPOSTrainer(SearchR1GRPOTrainer):
    """
    GRPO-S trainer (Search-R1 base). Extra kwargs:
      beta1 (float): base reward weight. Default 1.0
      beta2 (float): entropy bonus weight. Default 0.1
      eps_entropy_low  (float): min entropy clip. Default 0.2
      eps_entropy_high (float): max entropy clip. Default 0.28
      reward_threshold (float): O+/O- split threshold. Default 0.0
      conf_micro_bs (int): batch chunk for the no-grad entropy forward.
    """

    def __init__(self, *args, **kwargs):
        beta1            = kwargs.pop("beta1", 1.0)
        beta2            = kwargs.pop("beta2", 0.1)
        eps_entropy_low  = kwargs.pop("eps_entropy_low", 0.2)
        eps_entropy_high = kwargs.pop("eps_entropy_high", 0.28)
        reward_threshold = kwargs.pop("reward_threshold", 0.0)
        conf_micro_bs    = kwargs.pop("conf_micro_bs", 2)
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__ so GRPOTrainer.__init__ can't clobber them.
        self.beta1, self.beta2 = beta1, beta2
        self.eps_entropy_low, self.eps_entropy_high = eps_entropy_low, eps_entropy_high
        self.reward_threshold = reward_threshold
        self.conf_micro_bs = conf_micro_bs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        grpo_advantages = inputs["advantages"]          # (B,) standard GRPO advantages
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Real per-token Shannon entropy on the completion grid, chunked over the
        # batch dim to bound memory (A100). No grad (advantages are constants).
        entropies = entropy_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs,
        )  # (B, Lk)

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
