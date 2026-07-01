"""
adaptive_discount_trainer.py — exp_065: adaptive pos_discount on the FIXED λ=0.7
flipped shaping. One trainer, discount_kind in {p1, c1, pc1, pc2}. g_{i,t} multiplies
the α₂ exploration bonus; the shaping signal stays EMA(C) with λ (=0.7). All computed
on the FULL group in _generate_and_score (group-visible FIXED pattern, no B=1 bug).
"""
import torch
from .novel_trainers import GroupShapedBase
from .ema_flipped_utils import compute_ema_vectorized, EPS
from .novel_shaping import flipped_advantages, token_logprobs_chunked
from .adaptive_discount import g_p1, g_pc1, g_c1, g_pc2


class AdaptiveDiscountTrainer(GroupShapedBase):
    tag = "adisc"

    def __init__(self, *args, **kwargs):
        self._kind = kwargs.pop("discount_kind", "pc1")
        self._floor = kwargs.pop("floor", 0.3)
        self._gmin = kwargs.pop("g_min", 0.2)
        self._gmax = kwargs.pop("g_max", 1.5)
        super().__init__(*args, **kwargs)
        self.tag = f"adisc_{self._kind}"

    def _token_advantage(self, out):
        cids, cm, input_ids, attn, ltk = self._grid(out)
        conf = self._confidence(out)                              # C (G,T)
        ema = compute_ema_vectorized(conf, cm, lam=self.lam)      # λ=0.7 shaping signal
        valid = cm.bool()
        T = cm.size(1)
        if self._kind == "p1":
            g = g_p1(T, self.pos_tau, self._floor, cm.device).unsqueeze(0).expand(cm.size(0), T)
        elif self._kind == "pc1":
            m = ema[valid].mean(); sd = ema[valid].std()
            g = g_pc1(ema, m, sd, self.pos_tau, self._floor)
        elif self._kind == "pc2":
            C_ref = conf[valid].mean()
            g = g_pc2(conf, C_ref, self.pos_tau, self._gmin, self._gmax)
        elif self._kind == "c1":
            plk = ("logits_to_keep" in self.model_kwarg_keys)
            lp = token_logprobs_chunked(self.model, input_ids, attn, ltk, cids,
                                        pass_logits_to_keep=plk, micro_bs=self.conf_micro_bs)
            s = -lp
            s_ref = s[valid].mean()
            g = g_c1(s, s_ref, self._gmin)
        else:
            raise ValueError(f"unknown discount_kind: {self._kind}")
        g = g * cm                                               # zero on pad
        # metric: mean g over active tokens
        mode = "train" if self.model.training else "eval"
        self._metrics[mode].setdefault(f"{self.tag}/mean_g", []).append(
            (g[valid].mean() if valid.any() else torch.tensor(0.0)).item())
        return flipped_advantages(out["advantages"], ema, cm, self.alpha1, self.alpha2,
                                  reward_threshold=self.reward_threshold, bonus_mult=g)
