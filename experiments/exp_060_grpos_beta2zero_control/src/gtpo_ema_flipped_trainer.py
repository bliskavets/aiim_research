"""
gtpo_ema_flipped_trainer.py
---------------------------
Trainer for variant C of pure-proof GTPO-EMA — same skeleton as exp_025,
with O+/O- signal roles swapped. See ema_flipped_utils.py for rationale.
"""

import torch
from trl import GRPOTrainer

from .ema_flipped_utils import (
    confidence_from_logits,
    compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages,
    EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import forward_completion_logits, inject_advantages


class GTPOEMAFlippedTrainer(GRPOTrainer):
    """
    GTPO-EMA with flipped O+/O- signal roles (variant C).

    Extra kwargs (same defaults as exp_025):
        alpha1 (float): base reward weight.           Default 0.9
        alpha2 (float): EMA-confidence bonus weight.  Default 0.1
                        (α₁+α₂=1 for Prop 2.3 conservation)
        lam    (float): EMA decay λ ∈ (0,1).          Default 0.9
        top_k  (int):   top-k for confidence.         Default 20
        reward_threshold (float): O+/O- split.        Default 0.0
        format_tag_patterns (list[list[int]] | None): if provided, token-id
            subsequences for the format tags. On those positions the
            per-token shaped advantage is replaced by the seq-level GRPO
            advantage (no EMA shaping on format-control tokens).
    """

    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__: GRPOTrainer.__init__ defines its own
        # self.top_k (vLLM sampling top-k, default None) which would otherwise
        # clobber our confidence top_k -> None and crash _compute_loss.
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # unsloth replaces trl.GRPOTrainer with a compiled _UnslothGRPOTrainer
        # whose compute_loss is self-contained (never calls _compute_loss), which
        # bypassed the old _compute_loss shaping entirely. Instead of fighting it,
        # we compute the per-token shaped advantage here and INJECT it (as a 2-D
        # advantages tensor) into the compiled loss, which then owns the
        # memory-efficient chunked gradient. See src/shaped_loss.py.
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]            # (B,) GRPO seq advantages
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            logits = forward_completion_logits(self, model, input_ids, attention_mask, logits_to_keep)
            confidence = confidence_from_logits(logits, top_k=self.top_k)   # (B, Lk)
        del logits

        token_advantages = compute_gtpo_ema_flipped_advantages(
            rewards          = seq_advantages,
            confidence       = confidence,
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            lam              = self.lam,
            reward_threshold = self.reward_threshold,
        )
        # Revert format-tag tokens to the seq-level advantage (no EMA shaping there).
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # ── metrics (presence of these in the log proves the shaping ran) ──
        mode = "train" if model.training else "eval"
        total_tokens = completion_mask.sum().clamp(min=1.0)
        ema = compute_ema_vectorized(confidence, completion_mask, lam=self.lam)
        mean_ema = (ema * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_ema_flipped/mean_ema", []).append(
            self.accelerator.gather(mean_ema).mean().item())
        mean_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_ema_flipped/mean_token_advantage", []).append(
            self.accelerator.gather(mean_adv).mean().item())
        n_pos = (seq_advantages > self.reward_threshold).float().sum()
        n_neg = (seq_advantages <= self.reward_threshold).float().sum()
        self._metrics[mode].setdefault("gtpo_ema_flipped/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
