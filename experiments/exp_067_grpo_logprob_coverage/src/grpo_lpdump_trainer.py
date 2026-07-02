"""
grpo_lpdump_trainer.py — plain GRPO (loss unchanged) that DUMPS, per rollout token,
the sorted top-K logprobs of the policy distribution (for studying how top-k / nucleus
covers probability mass) plus the rollout token sequences. Saves per sampled step to
diag/<tag>/step_XXXX.npz. Used to design adaptive top-k / top-p (nucleus) selection.
"""
import os
import numpy as np
import torch
from trl import GRPOTrainer


@torch.no_grad()
def topk_logprobs_chunked(model, input_ids, attention_mask, logits_to_keep, K,
                          target_ids=None, pass_logits_to_keep=False, micro_bs=1):
    """Return (topk_lp (G,T,K) sorted-desc log-probs, sampled_lp (G,T) log-prob of
    target_ids or None). log_softmax over the FULL vocab, chunked over batch."""
    B = input_ids.size(0)
    tks, samps = [], []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        tks.append(lp.topk(K, dim=-1).values.to(torch.float16).cpu())
        if target_ids is not None:
            samps.append(lp.gather(-1, target_ids[s:e].unsqueeze(-1)).squeeze(-1).to(torch.float16).cpu())
        del logits, lp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tk = torch.cat(tks, 0)
    sp = torch.cat(samps, 0) if target_ids is not None else None
    return tk, sp


class GRPOLpdumpTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self._K = kwargs.pop("dump_k", 128)
        self._conf_micro_bs = kwargs.pop("conf_micro_bs", 1)
        self._dump_dir = kwargs.pop("dump_dir", "diag/lpdump")
        self._save_every = kwargs.pop("save_every", 1)
        self._max_save = kwargs.pop("max_save_steps", 100)
        super().__init__(*args, **kwargs)
        os.makedirs(self._dump_dir, exist_ok=True)
        self._gen = 0
        self._saved = 0
        self._warned = False

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)   # plain GRPO scoring
        self._gen += 1
        if self._saved < self._max_save and (self._gen % self._save_every == 0):
            try:
                cids = out["completion_ids"]; cm = out["completion_mask"]
                input_ids = torch.cat([out["prompt_ids"], cids], dim=1)
                attn = torch.cat([out["prompt_mask"], cm], dim=1)
                ltk = cids.size(1)
                tk, sp = topk_logprobs_chunked(
                    self.model, input_ids, attn, ltk, self._K, target_ids=cids,
                    pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
                    micro_bs=self._conf_micro_bs)
                np.savez_compressed(
                    os.path.join(self._dump_dir, f"step_{self._gen:04d}.npz"),
                    completion_ids=cids.cpu().numpy().astype(np.int32),
                    completion_mask=cm.cpu().numpy().astype(np.int8),
                    topk_lp=tk.numpy(),                    # (G, T, K) fp16, sorted desc
                    sampled_lp=sp.numpy(),                 # (G, T) fp16
                    step=np.array([int(self.state.global_step)]))
                self._saved += 1
            except Exception as e:
                if not self._warned:
                    print(f"[lpdump] save skipped: {e!r}", flush=True); self._warned = True
        return out
