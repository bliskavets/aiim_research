"""
gtpo_conf_trainer.py
--------------------
GTPOConfTrainer: GTPO variant using confidence-based reward shaping
instead of entropy-based shaping (exp_005), on the Search-R1 multi-turn base.

Confidence metric: C_i,t = -mean_{top-k}(log π(v | context))

Shaping is applied by overriding `compute_loss` and INJECTING a per-token shaped
advantage into unsloth's compiled loss (which owns the memory-efficient gradient).
Overriding `_compute_loss` does NOT work on the unsloth stack — the compiled
compute_loss never calls it, so the shaping was silently bypassed (ran plain
GRPO). See src/shaped_loss.py and exp_057's SHAPING_BYPASS_BUGFIX.md.
"""

import torch
from trl import GRPOTrainer
from .confidence_utils import (
    confidence_from_model_chunked,
    compute_gtpo_conf_rewards,
    EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages


class GTPOConfTrainer(GRPOTrainer):
    """
    GTPO with confidence-based token-level reward shaping (Search-R1 base).

    Extra kwargs:
        alpha1 (float): base reward weight. Default 1.0
        alpha2 (float): confidence bonus weight. Default 0.1
        top_k  (int):   top-k tokens for confidence. Default 20
        reward_threshold (float): O+/O- split. Default 0.0
        format_tag_patterns (list[list[int]] | None): token-id subsequences for
            the format tags; on those positions the per-token shaped advantage is
            replaced by the seq-level GRPO advantage (no shaping on tag tokens).
        conf_micro_bs (int): batch chunk for the no-grad confidence forward.
    """

    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 1.0)
        alpha2               = kwargs.pop("alpha2", 0.1)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__: GRPOTrainer.__init__ defines its own
        # self.top_k (vLLM sampling top-k, default None) which would otherwise
        # clobber our confidence top_k -> None and crash the shaping.
        self.alpha1, self.alpha2, self.top_k = alpha1, alpha2, top_k
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

        # No-grad confidence on the completion grid, chunked over batch (A100 mem).
        confidence = confidence_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep,
            top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs,
        )  # (B, Lk)

        adv_pos, adv_neg = compute_gtpo_conf_rewards(
            rewards          = seq_advantages,
            confidence       = confidence,
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            top_k            = self.top_k,
            reward_threshold = self.reward_threshold,
        )
        token_advantages = adv_pos + adv_neg   # (B, Lk)
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # ── metrics (presence of these in the log proves the shaping ran) ──
        mode = "train" if model.training else "eval"
        total_tokens = completion_mask.sum().clamp(min=1.0)
        mean_conf = (confidence * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_conf/mean_confidence", []).append(
            self.accelerator.gather(mean_conf).mean().item())
        mean_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_conf/mean_token_advantage", []).append(
            self.accelerator.gather(mean_adv).mean().item())
        n_pos = (seq_advantages > self.reward_threshold).float().sum()
        n_neg = (seq_advantages <= self.reward_threshold).float().sum()
        self._metrics[mode].setdefault("gtpo_conf/frac_pos", []).append(
            (n_pos / (n_pos + n_neg + EPS)).item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
