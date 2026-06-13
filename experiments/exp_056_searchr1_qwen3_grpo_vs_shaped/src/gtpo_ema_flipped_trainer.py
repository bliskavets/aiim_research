"""
gtpo_ema_flipped_trainer.py
---------------------------
Variant C of pure-proof GTPO-EMA — O+/O- signal roles swapped (see
ema_flipped_utils.py), on the Search-R1 multi-turn base.

Shaping is applied by overriding `compute_loss` and INJECTING a per-token shaped
advantage into unsloth's compiled loss (which owns the memory-efficient gradient).
Overriding `_compute_loss` is silently bypassed on the unsloth stack — see
src/shaped_loss.py and exp_057's SHAPING_BYPASS_BUGFIX.md.
"""

import torch
from .searchr1_trainer import SearchR1GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked,
    compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages,
    EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages


class GTPOEMAFlippedTrainer(SearchR1GRPOTrainer):
    """
    GTPO-EMA with flipped O+/O- signal roles (variant C), Search-R1 base.

    Extra kwargs (same defaults as exp_025):
        alpha1 (float): base reward weight.           Default 0.9
        alpha2 (float): EMA-confidence bonus weight.  Default 0.1  (α₁+α₂=1)
        lam    (float): EMA decay λ ∈ (0,1).          Default 0.9
        top_k  (int):   top-k for confidence.         Default 20
        reward_threshold (float): O+/O- split.        Default 0.0
        format_tag_patterns (list[list[int]] | None): see GTPOConfTrainer.
        conf_micro_bs (int): batch chunk for the no-grad confidence forward.
    """

    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__ so GRPOTrainer.__init__ can't clobber top_k.
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]            # (B,) GRPO seq advantages
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        confidence = confidence_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep,
            top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs,
        )  # (B, Lk)

        token_advantages = compute_gtpo_ema_flipped_advantages(
            rewards          = seq_advantages,
            confidence       = confidence,
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            lam              = self.lam,
            reward_threshold = self.reward_threshold,
        )
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # ── metrics (presence proves the shaping ran) ──
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
