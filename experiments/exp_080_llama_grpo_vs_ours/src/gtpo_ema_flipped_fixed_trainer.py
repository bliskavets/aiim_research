"""
gtpo_ema_flipped_fixed_trainer.py
---------------------------------
FIX for the B=1-microbatch degeneracy diagnosed in DIAG_LENGTH_EXPLOSION.md.

The bare gtpo_ema_flipped recomputes the shaped per-token advantage inside
compute_loss, which unsloth feeds ONE completion at a time (B=1). The shaping is a
GROUP operation (per-position Σ over O+/O- and per-polarity z-norm need the whole
num_generations group), so with B=1 it collapses to a degenerate per-sequence
constant that inverts the reward and mildly rewards length.

This trainer computes the shaped per-token advantage ONCE in
_generate_and_score_completions, where the FULL group is present (and the policy
is still θ_old — matching the paper's π_θold entropy), then propagates it as a
2-D (G, Lk) tensor through the buffer; compute_loss just INJECTS the per-completion
slice (no recomputation, no B=1 group op). This is the method-level fix flagged as
the follow-up.
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages


class GTPOEMAFlippedFixedTrainer(GRPOTrainer):
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
        self._fixed_warned = False

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        try:
            cids = out["completion_ids"]
            cm   = out["completion_mask"]
            seq_adv = out["advantages"]
            input_ids      = torch.cat([out["prompt_ids"], cids], dim=1)
            attention_mask = torch.cat([out["prompt_mask"], cm], dim=1)
            logits_to_keep = cids.size(1)
            confidence = confidence_from_model_chunked(
                self.model, input_ids, attention_mask, logits_to_keep,
                top_k=self.top_k,
                pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
                micro_bs=self.conf_micro_bs)
            token_adv = compute_gtpo_ema_flipped_advantages(
                rewards=seq_adv, confidence=confidence, completion_mask=cm,
                alpha1=self.alpha1, alpha2=self.alpha2, lam=self.lam,
                reward_threshold=self.reward_threshold)            # (G, Lk) — FULL group
            if self.format_tag_patterns:
                tag_mask = build_tag_mask(cids, self.format_tag_patterns)
                token_adv = apply_tag_mask_to_token_advantages(token_adv, seq_adv, tag_mask)
            out["shaped_adv"] = token_adv.to(seq_adv.dtype)
        except Exception as e:
            if not self._fixed_warned:
                print(f"[gtpo_ema_flipped_fixed] WARN group shaping failed, "
                      f"falling back to seq advantage: {e!r}", flush=True)
                self._fixed_warned = True
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")
        completion_mask = inputs["completion_mask"]
        logits_to_keep = inputs["completion_ids"].size(1)
        B, W = completion_mask.shape

        shaped = inputs.get("shaped_adv")
        if shaped is not None and shaped.shape[0] == B and shaped.shape[1] == W:
            token_advantages = shaped.to(completion_mask.dtype if completion_mask.is_floating_point()
                                         else torch.float32) * completion_mask
            used_fixed = 1.0
        else:
            # fallback: broadcast the seq advantage (no shaping) — keeps training valid
            token_advantages = inputs["advantages"].unsqueeze(1).float() * completion_mask
            used_fixed = 0.0

        mode = "train" if model.training else "eval"
        tot = completion_mask.sum().clamp(min=1.0)
        self._metrics[mode].setdefault("gtpo_ema_flipped_fixed/mean_token_advantage", []).append(
            self.accelerator.gather((token_advantages * completion_mask).sum() / tot).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_flipped_fixed/used_group_shaped", []).append(used_fixed)

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
