"""
gtpo_conf_trainer.py
--------------------
GTPOConfTrainer: GRPO + per-token confidence-weighted advantages (exp_047).

Ported from exp_005 to TRL 0.23.1 (override compute_loss, not _compute_loss).
Per-token loops are fully vectorised — no Python for-t-in-range(T) overhead.
Ghost guard added for batches where all rewards are identical.

Confidence:  C_i,t = -mean_{v∈top-k}(log π_v)  — large = uncertain
O+ bonus  ∝ log(1+C)     → reward uncertain/exploratory tokens on correct paths
O- penalty ∝ log(1+1/C)  → punish confident tokens on wrong paths
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from trl import GRPOTrainer

from .confidence_utils import confidence_from_log_probs, compute_gtpo_conf_rewards, EPS


class GTPOConfTrainer(GRPOTrainer):
    """
    GRPO + per-token confidence reward shaping (GTPO-Conf).

    Extra __init__ kwargs:
        rollout_log_dir   (str):  directory for .npz files.
        correctness_store (dict): {completion_text: (is_correct, gt, extracted)}
        conf_top_k        (int):  top-K for confidence. Default 20.
        save_every_steps  (int):  npz every N steps. Default 1.
        alpha1 (float): base weight.              Default 1.0
        alpha2 (float): confidence bonus weight.  Default 0.1
        reward_threshold (float): O+/O- split.    Default 0.0
    """

    def __init__(self, *args,
                 rollout_log_dir="rollout_logs",
                 correctness_store=None,
                 conf_top_k=20,
                 save_every_steps=1,
                 alpha1=1.0,
                 alpha2=0.1,
                 reward_threshold=0.0,
                 **kwargs):
        self.rollout_log_dir   = rollout_log_dir
        self.correctness_store = correctness_store if correctness_store is not None else {}
        self._conf_top_k       = int(conf_top_k)
        self.save_every_steps  = max(1, int(save_every_steps))
        self.alpha1            = float(alpha1)
        self.alpha2            = float(alpha2)
        self.reward_threshold  = float(reward_threshold)
        self._log_step         = 0
        os.makedirs(self.rollout_log_dir, exist_ok=True)
        super().__init__(*args, **kwargs)
        print(f"[GTPOConfTrainer] α1={self.alpha1} α2={self.alpha2} "
              f"top_k={self._conf_top_k} threshold={self.reward_threshold}")
        print(f"[GTPOConfTrainer] O+ bonus ∝ log(1+C); O- penalty ∝ log(1+1/C); vectorised")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if hasattr(self, "model_kwarg_keys") and "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        raw_out = model(**model_inputs)

        if "logits_to_keep" in model_inputs:
            logits = raw_out.logits[:, :-1, :].float().contiguous()
        else:
            logits = raw_out.logits[:, -(logits_to_keep + 1):-1, :].float().contiguous()

        log_probs = F.log_softmax(logits, dim=-1)

        per_token_logps = torch.gather(
            log_probs, 2, completion_ids.unsqueeze(2)
        ).squeeze(2)

        with torch.no_grad():
            confidence, topk_lp, topk_ids = confidence_from_log_probs(
                log_probs.detach(), top_k=self._conf_top_k
            )

        del logits, log_probs

        if seq_advantages.std() < EPS:
            if self._log_step % self.save_every_steps == 0:
                try:
                    self._log_rollout(inputs, confidence, topk_lp, topk_ids)
                except Exception:
                    pass
            self._log_step += 1
            return per_token_logps.sum() * 0.0

        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # ── GTPO-Conf per-token advantages ────────────────────────────────────
        adv_pos, adv_neg = compute_gtpo_conf_rewards(
            rewards          = seq_advantages,
            confidence       = confidence.detach(),
            completion_mask  = completion_mask,
            alpha1           = self.alpha1,
            alpha2           = self.alpha2,
            reward_threshold = self.reward_threshold,
        )
        token_advantages = adv_pos + adv_neg   # (B, T)

        # ── PPO clipping ──────────────────────────────────────────────────────
        log_ratio = per_token_logps - old_per_token_logps
        coef_1    = torch.exp(log_ratio)
        eps_low   = getattr(self, "epsilon_low",  getattr(self, "epsilon", 0.2))
        eps_high  = getattr(self, "epsilon_high", getattr(self, "epsilon", 0.2))
        coef_2    = torch.clamp(coef_1, 1.0 - eps_low, 1.0 + eps_high)

        per_token_loss = -torch.min(coef_1 * token_advantages, coef_2 * token_advantages)

        beta = getattr(self, "beta", 0.0)
        if beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_lp = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_lp - per_token_logps)
                - (ref_lp - per_token_logps) - 1
            )
            per_token_loss = per_token_loss + beta * per_token_kl

        total_tokens = completion_mask.sum().clamp(min=1.0)
        grad_acc     = getattr(self, "current_gradient_accumulation_steps", 1)
        loss = (per_token_loss * completion_mask).sum() / total_tokens / grad_acc

        # ── metrics ───────────────────────────────────────────────────────────
        mode = "train" if model.training else "eval"
        with torch.no_grad():
            mean_conf      = (confidence * completion_mask).sum() / total_tokens
            mean_token_adv = (token_advantages * completion_mask).sum() / total_tokens
            n_pos = (seq_advantages > self.reward_threshold).float().sum()
            n_neg = (seq_advantages <= self.reward_threshold).float().sum()

        self._metrics[mode].setdefault("gtpo_conf/mean_confidence", []).append(
            self.accelerator.gather(mean_conf).mean().item())
        self._metrics[mode].setdefault("gtpo_conf/mean_token_adv",  []).append(
            self.accelerator.gather(mean_token_adv).mean().item())
        self._metrics[mode].setdefault("gtpo_conf/frac_pos",        []).append(
            (n_pos / (n_pos + n_neg + EPS)).item())

        if self._log_step % self.save_every_steps == 0:
            try:
                self._log_rollout(inputs, confidence, topk_lp, topk_ids)
            except Exception as e:
                import traceback
                err_path = os.path.join(self.rollout_log_dir,
                                        f"error_step_{self._log_step}.txt")
                with open(err_path, "w") as _f:
                    _f.write(traceback.format_exc())
                print(f"[GTPOConfTrainer] WARNING step {self._log_step}: {e}")
        self._log_step += 1
        return loss

    def _log_rollout(self, inputs, confidence, topk_lp, topk_ids):
        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        prompt_ids      = inputs["prompt_ids"]
        prompt_mask     = inputs["prompt_mask"]
        seq_advantages  = inputs["advantages"]
        B = completion_ids.size(0)

        mask_arr    = completion_mask.cpu().long().numpy().astype(np.int8)
        comp_ids_np = completion_ids.cpu().long().numpy().astype(np.int32)

        is_correct        = np.zeros(B, dtype=bool)
        gt_answer         = ""
        extracted_answers = [""] * B

        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if self.correctness_store and tok is not None:
            for i in range(B):
                length = int(mask_arr[i].sum())
                text   = tok.decode(comp_ids_np[i, :length].tolist(),
                                    skip_special_tokens=False)
                val = self.correctness_store.pop(text, None)
                if val is None:
                    text_stripped = text.rstrip()
                    for eos in ["<|im_end|>", "</s>", "<|endoftext|>"]:
                        if text_stripped.endswith(eos):
                            text_stripped = text_stripped[:-len(eos)].rstrip()
                            break
                    val = self.correctness_store.pop(text_stripped, None)
                if val is not None:
                    ic, gt, ext = val
                    is_correct[i]        = ic
                    extracted_answers[i] = ext
                    if gt:
                        gt_answer = gt

        out_path = os.path.join(self.rollout_log_dir, f"step_{self._log_step:05d}.npz")
        np.savez_compressed(
            out_path,
            step              = np.array(self._log_step, dtype=np.int64),
            completion_ids    = comp_ids_np,
            completion_mask   = mask_arr,
            confidence        = confidence.cpu().numpy().astype(np.float16),
            topk_log_probs    = topk_lp.cpu().numpy().astype(np.float16),
            topk_token_ids    = topk_ids.cpu().long().numpy().astype(np.int32),
            advantages        = seq_advantages.float().cpu().numpy().astype(np.float32),
            is_correct        = is_correct,
            gt_answer         = np.array(gt_answer.encode("utf-8")),
            extracted_answers = np.array([a.encode("utf-8") for a in extracted_answers]),
            prompt_ids        = prompt_ids[0].cpu().long().numpy().astype(np.int32),
            prompt_mask       = prompt_mask[0].cpu().long().numpy().astype(np.int8),
        )
        n_correct = int(is_correct.sum())
        total_tok = completion_mask.sum().clamp(min=1.0)
        mean_c    = (confidence * completion_mask).sum() / total_tok
        print(f"[GTPOConfTrainer] step {self._log_step:05d} → {os.path.basename(out_path)}  "
              f"correct={n_correct}/{B}  gt={gt_answer!r}  "
              f"mean_conf={mean_c.item():.3f}  "
              f"mean_adv={float(seq_advantages.float().mean()):.3f}  "
              f"store_remaining={len(self.correctness_store)}")
