"""
rank_c_trainer.py — exp_069. Base = gtpo_ema_flipped (FIXED) + pos_discount,
but C uses a per-token adaptive k = clamp(rank_of_sampled_token, min_k, cap).
EMA(λ) -> flipped shaping -> pos_discount g(t), all on the FULL group.
"""
import torch
from .novel_trainers import GroupShapedBase
from .ema_flipped_utils import compute_ema_vectorized
from .novel_shaping import flipped_advantages, position_discount
from .rank_c import rank_C_from_model_chunked


class RankCTrainer(GroupShapedBase):
    tag = "rank_c"

    def __init__(self, *args, **kwargs):
        self._cap = kwargs.pop("rank_cap", 5)
        self._min_k = kwargs.pop("min_k", 1)
        super().__init__(*args, **kwargs)

    def _token_advantage(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        C, k = rank_C_from_model_chunked(
            self.model, input_ids, attn, cids, ltk, cap=self._cap, min_k=self._min_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)
        ema = compute_ema_vectorized(C, cm, lam=self.lam)
        T = cm.size(1)
        g = position_discount(T, self.pos_tau, cm.device).unsqueeze(0).expand(cm.size(0), T)
        valid = cm.bool()
        mode = "train" if self.model.training else "eval"
        if valid.any():
            self._metrics[mode].setdefault("rank_c/mean_k", []).append(k[valid].mean().item())
        return flipped_advantages(out["advantages"], ema, cm, self.alpha1, self.alpha2,
                                  reward_threshold=self.reward_threshold, bonus_mult=g)
