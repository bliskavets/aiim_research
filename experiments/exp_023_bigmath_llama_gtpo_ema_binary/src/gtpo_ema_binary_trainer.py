"""
gtpo_ema_binary_trainer.py
--------------------------
GTPO-EMA trainer with binary O+/O- split from answer_exact reward.

Key difference from exp_018 gtpo_ema_trainer.py:
  - O+/O- derived from reward_cache (populated by stashing reward_answer_exact)
  - Signed ±1 mask passed as `rewards` into compute_gtpo_ema_advantages with
    reward_threshold=0.0. The base_adv inside that function becomes a z-scored
    binary signal, which still gives a meaningful group-relative direction.
"""

import torch
from trl import GRPOTrainer

from .ema_confidence_utils import (
    confidence_from_logits,
    compute_gtpo_ema_advantages,
    EPS,
)
from .reward_cache import _CACHE


class GTPoEMABinaryTrainer(GRPOTrainer):
    """
    GTPO-EMA trainer with binary O+/O- split from answer_exact.

    Extra kwargs:
      alpha1 (float): base advantage weight. Default 1.0
      alpha2 (float): EMA confidence bonus weight. Default 0.1
      top_k  (int):   top-k for confidence. Default 20
      lam    (float): EMA decay. Default 0.9
    """

    def __init__(self, *args, **kwargs):
        self.alpha1 = kwargs.pop("alpha1", 1.0)
        self.alpha2 = kwargs.pop("alpha2", 0.1)
        self.top_k  = kwargs.pop("top_k", 20)
        self.lam    = kwargs.pop("lam", 0.9)
        super().__init__(*args, **kwargs)

    def _compute_loss(self, model, inputs):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        device         = completion_mask.device

        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
        )

        with torch.no_grad():
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            raw_out = model(**model_inputs)
            logits = raw_out.logits[:, :-1, :][:, -logits_to_keep:, :]
            confidence = confidence_from_logits(logits, top_k=self.top_k)

        # ── Binary O+/O- from reward_cache ────────────────────────────────────
        binary_mask = _CACHE.get()
        if binary_mask is None:
            raise RuntimeError(
                "binary correctness cache is empty — ensure reward_answer_exact "
                "populates _CACHE before _compute_loss runs"
            )
        if binary_mask.shape[0] != completion_mask.shape[0]:
            raise ValueError(
                f"cache mask shape {binary_mask.shape} does not match batch "
                f"{completion_mask.shape[0]}"
            )
        binary_mask = binary_mask.to(device)
        signed_rewards = torch.where(
            binary_mask,
            torch.tensor( 1.0, device=device),
            torch.tensor(-1.0, device=device),
        )

        token_advantages = compute_gtpo_ema_advantages(
            rewards         = signed_rewards,
            confidence      = confidence,
            completion_mask = completion_mask,
            alpha1          = self.alpha1,
            alpha2          = self.alpha2,
            lam             = self.lam,
            reward_threshold = 0.0,  # split on signed_rewards sign
        )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss = -torch.min(coef_1 * token_advantages, coef_2 * token_advantages)

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
        mean_conf = (confidence * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_ema_binary/mean_confidence", []).append(
            self.accelerator.gather(mean_conf).mean().item()
        )
        mean_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_ema_binary/mean_token_advantage", []).append(
            self.accelerator.gather(mean_adv).mean().item()
        )
        frac_pos = binary_mask.float().mean()
        self._metrics[mode].setdefault("gtpo_ema_binary/frac_pos", []).append(
            self.accelerator.gather(frac_pos).mean().item()
        )

        return loss
