"""
nucleus_c_trainer.py — exp_068. Base = gtpo_ema_flipped (FIXED) + pos_discount,
but C uses a DYNAMIC nucleus (top-p) k per token instead of a fixed top-k.
n_it = #{leading tokens with cumulative prob ≤ top_p}, clamped ≥ min_k;
C_it = −mean of those n_it log-probs. EMA(λ) → flipped shaping → pos_discount g(t)
→ per-polarity z-norm, all on the FULL group (group-visible FIXED pattern).
"""
import torch
from .novel_trainers import GroupShapedBase
from .ema_flipped_utils import compute_ema_vectorized
from .novel_shaping import flipped_advantages, position_discount
from .nucleus_c import nucleus_C_from_model_chunked


class NucleusCTrainer(GroupShapedBase):
    tag = "nucleus_c"

    def __init__(self, *args, **kwargs):
        self._top_p = kwargs.pop("nucleus_top_p", 0.9)
        self._min_k = kwargs.pop("min_k", 1)
        self._cap = kwargs.pop("nucleus_cap", 256)
        self._floor = kwargs.pop("floor", 0.3)
        super().__init__(*args, **kwargs)

    def _token_advantage(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        C, n = nucleus_C_from_model_chunked(
            self.model, input_ids, attn, ltk, self._top_p, min_k=self._min_k, cap=self._cap,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)
        ema = compute_ema_vectorized(C, cm, lam=self.lam)
        T = cm.size(1)
        g = position_discount(T, self.pos_tau, cm.device).unsqueeze(0).expand(cm.size(0), T)
        # metric: mean nucleus size over valid tokens
        valid = cm.bool()
        mode = "train" if self.model.training else "eval"
        if valid.any():
            self._metrics[mode].setdefault("nucleus_c/mean_n", []).append(n[valid].mean().item())
        return flipped_advantages(out["advantages"], ema, cm, self.alpha1, self.alpha2,
                                  reward_threshold=self.reward_threshold, bonus_mult=g)
