"""
gtpo_ema_flipped_trainer.py  (exp_028)
--------------------------------------
Flipped pure-proof GTPO-EMA, with O+/O- split driven by `_CACHE.mask`
(populated by `reward_answer_exact` at scoring time) instead of by the
z-scored advantage.
"""

import torch
from trl import GRPOTrainer

from .ema_flipped_utils import (
    confidence_from_logits,
    compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages,
    EPS,
)
from .reward_cache import _CACHE


class GTPOEMAFlippedTrainer(GRPOTrainer):
    """
    Extra kwargs:
        alpha1 (float): base weight.             Default 0.9
        alpha2 (float): EMA-confidence weight.   Default 0.1
                        (α₁+α₂=1 keeps Prop 2.3 conservation)
        lam    (float): EMA decay λ ∈ (0,1).     Default 0.9
        top_k  (int):   top-k for confidence.    Default 20
    """

    def __init__(self, *args, **kwargs):
        self.alpha1 = kwargs.pop("alpha1", 0.9)
        self.alpha2 = kwargs.pop("alpha2", 0.1)
        self.lam    = kwargs.pop("lam", 0.9)
        self.top_k  = kwargs.pop("top_k", 20)
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        with torch.no_grad():
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            raw_out = model(**model_inputs)
            logits = raw_out.logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            confidence = confidence_from_logits(logits, top_k=self.top_k)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # O+/O- from reward_cache (populated by reward_answer_exact at scoring)
        mask = _CACHE.get()
        if mask is None:
            # Safe fallback: seq_advantages > 0
            mask = (inputs["advantages"] > 0).detach().cpu()
        is_pos = mask[: confidence.size(0)].to(confidence.device)

        token_advantages = compute_gtpo_ema_flipped_advantages(
            is_pos          = is_pos,
            confidence      = confidence,
            completion_mask = completion_mask,
            alpha1          = self.alpha1,
            alpha2          = self.alpha2,
            lam             = self.lam,
        )

        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_loss  = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl

        total_tokens = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / total_tokens
        loss = loss / self.current_gradient_accumulation_steps

        mode = "train" if model.training else "eval"
        ema = compute_ema_vectorized(confidence, completion_mask, lam=self.lam)
        mean_ema = (ema * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_ema_flipped/mean_ema", []).append(
            self.accelerator.gather(mean_ema).mean().item()
        )
        self._metrics[mode].setdefault("gtpo_ema_flipped/frac_pos", []).append(
            is_pos.float().mean().item()
        )

        return loss
