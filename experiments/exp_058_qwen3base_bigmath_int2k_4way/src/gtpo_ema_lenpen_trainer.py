"""
gtpo_ema_lenpen_trainer.py
--------------------------
exp_058 NEW method #1: gtpo_ema_flipped + a Lagrange-like LENGTH PENALTY.

gtpo_ema_flipped rewards low-confidence (exploratory) tokens in O+ paths, so on
the base model it farms length (640 -> 3400 tok) and collapses. We subtract a
per-sequence penalty from the shaped advantage:

    pen_i = alpha_len * max(0, |o_i| - L)          (0 < L < max_completion_tokens)
    Ã_{i,t} <- Ã_{i,t} - pen_i        (broadcast over the sequence's valid tokens)

so completions longer than L lose advantage in proportion to the overshoot. The
4 original candidates are untouched; this is an additive new method.
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked,
    compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages,
    EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages


class GTPOEMAFlippedLenPenTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        alpha_len            = kwargs.pop("alpha_len", 0.0015)
        length_L             = kwargs.pop("length_L", 1024)
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs
        self.alpha_len, self.length_L = alpha_len, length_L

    def _length_penalty(self, completion_mask):
        """pen_i = alpha_len * max(0, |o_i| - L)  -> (B,1)."""
        lengths = completion_mask.sum(dim=1)                          # (B,)
        pen = self.alpha_len * (lengths - self.length_L).clamp(min=0.0)
        return pen.unsqueeze(1)                                       # (B,1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        confidence = confidence_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep, top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)

        token_advantages = compute_gtpo_ema_flipped_advantages(
            rewards=seq_advantages, confidence=confidence, completion_mask=completion_mask,
            alpha1=self.alpha1, alpha2=self.alpha2, lam=self.lam,
            reward_threshold=self.reward_threshold)
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # ── Lagrange-like length penalty (the only change vs gtpo_ema_flipped) ──
        pen = self._length_penalty(completion_mask)                  # (B,1)
        token_advantages = token_advantages - pen * completion_mask  # subtract on valid tokens

        # ── metrics ──
        mode = "train" if model.training else "eval"
        total_tokens = completion_mask.sum().clamp(min=1.0)
        ema = compute_ema_vectorized(confidence, completion_mask, lam=self.lam)
        self._metrics[mode].setdefault("gtpo_ema_lenpen/mean_ema", []).append(
            self.accelerator.gather((ema * completion_mask).sum() / total_tokens).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen/mean_len", []).append(
            self.accelerator.gather(completion_mask.sum(dim=1).float().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen/mean_len_penalty", []).append(
            self.accelerator.gather(pen.mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen/frac_penalized", []).append(
            self.accelerator.gather((pen > 0).float().mean()).mean().item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
