"""
gtpo_binary_trainer.py
----------------------
GTPO trainer with binary O+/O- split based on answer_exact reward
(instead of z-scored advantages).

Key difference from exp_020 gtpo_trainer.py:
  - O+/O- is derived from `reward_cache._CACHE.mask` (populated by the
    stashing `reward_answer_exact` in train.py)
  - We convert the bool mask to a signed ±1 tensor and pass it as `rewards`
    to `compute_gtpo_rewards` with `reward_threshold=0.0`. The utility
    function only uses rewards for the O+/O- split (shaped values are
    driven by the entropy bonus), so the existing code works unchanged.
"""

import torch
from trl import GRPOTrainer

from .entropy_utils import compute_gtpo_rewards, EPS
from .reward_cache import _CACHE


class GTPOBinaryTrainer(GRPOTrainer):
    """
    GTPO trainer with binary O+/O- split from answer_exact reward.

    Extra kwargs:
      alpha1 (float): base reward weight. Default 1.0
      alpha2 (float): entropy bonus weight. Default 0.1
      eps_entropy_low  (float): min entropy clip. Default 0.2
      eps_entropy_high (float): max entropy clip. Default 0.28
    """

    def __init__(self, *args, **kwargs):
        self.alpha1           = kwargs.pop("alpha1", 1.0)
        self.alpha2           = kwargs.pop("alpha2", 0.1)
        self.eps_entropy_low  = kwargs.pop("eps_entropy_low", 0.2)
        self.eps_entropy_high = kwargs.pop("eps_entropy_high", 0.28)
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

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )
        if entropies is None:
            entropies = torch.ones_like(completion_mask, dtype=torch.float32) * 0.24

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

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

        adv_pos, adv_neg = compute_gtpo_rewards(
            rewards          = signed_rewards,
            entropies        = entropies,
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            eps_low          = self.eps_entropy_low,
            eps_high         = self.eps_entropy_high,
            reward_threshold = 0.0,  # split on signed_rewards sign
        )
        token_advantages = adv_pos + adv_neg

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
        mean_token_adv = (token_advantages * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_binary/mean_token_advantage", []).append(
            self.accelerator.gather(mean_token_adv).mean().item()
        )
        mean_entropy = (entropies * completion_mask).sum() / total_tokens
        self._metrics[mode].setdefault("gtpo_binary/mean_entropy", []).append(
            self.accelerator.gather(mean_entropy).mean().item()
        )
        frac_pos = binary_mask.float().mean()
        self._metrics[mode].setdefault("gtpo_binary/frac_pos", []).append(
            self.accelerator.gather(frac_pos).mean().item()
        )

        return loss
