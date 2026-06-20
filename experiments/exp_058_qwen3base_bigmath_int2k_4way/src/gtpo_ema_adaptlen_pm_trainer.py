"""
gtpo_ema_adaptlen_pm_trainer.py
-------------------------------
exp_058 method #5: gtpo_ema_flipped + adaptive length penalty computed SEPARATELY
within the O+ / O- subgroups of each group (pm = plus/minus). The group is split
by seq-advantage sign (O+ = adv>0 'correct', O- = adv<0 'incorrect'); each
polarity gets its own knee L_+/L_- = max((Lmin+Lmax)/2, Lmean) over that
polarity's lengths and the piecewise penalty in [-0.5,0], centered within the
polarity. See adaptive_lenpen_utils.adaptive_length_penalty_polarity.

Applied ALWAYS. Mechanics identical to the other lenpen trainers: pen_rel is
computed in _generate_and_score_completions (full group + advantages available)
and SUBTRACTED from the shaped per-token advantage in compute_loss.
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages
from .adaptive_lenpen_utils import adaptive_length_penalty_polarity


class GTPOEMAFlippedAdaptLenPMTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        cm = out["completion_mask"]
        lengths = cm.sum(dim=1).float()                              # (B_gen,)
        adv = out["advantages"].detach().float().reshape(-1)         # seq advantage sign -> O+/O-
        pen_rel, L_own = adaptive_length_penalty_polarity(lengths, adv, self.num_generations)
        out["len_pen"] = pen_rel
        out["adapt_L"] = L_own
        return out

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

        # per-polarity adaptive group-relative length penalty (propagated from generation)
        B = completion_mask.shape[0]
        pen_rel = inputs.get("len_pen")
        if pen_rel is None or pen_rel.shape[0] != B:
            pen_rel = torch.zeros(B, device=completion_mask.device, dtype=token_advantages.dtype)
            pen_present = 0.0
        else:
            pen_rel = pen_rel.to(token_advantages.dtype); pen_present = 1.0
        token_advantages = token_advantages - pen_rel.unsqueeze(1) * completion_mask

        # ── metrics ──
        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("gtpo_ema_adaptlen_pm/mean_len", []).append(
            self.accelerator.gather(completion_mask.sum(dim=1).float().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_adaptlen_pm/pen_rel_absmean", []).append(
            self.accelerator.gather(pen_rel.abs().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_adaptlen_pm/pen_present", []).append(pen_present)
        adapt_L = inputs.get("adapt_L")
        if adapt_L is not None and adapt_L.shape[0] == B:
            aL = adapt_L.to(token_advantages.dtype)
            finite = torch.isfinite(aL)
            if finite.any():
                self._metrics[mode].setdefault("gtpo_ema_adaptlen_pm/mean_L", []).append(
                    self.accelerator.gather(aL[finite].mean()).mean().item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
