"""
rollout_logger_trainer.py
-------------------------
Extends GRPOTrainer (via unsloth's patched version) to save per-rollout data
at every training step. Since unsloth replaces compute_loss entirely and never
calls _compute_loss, we override compute_loss directly: call super() for the
actual loss, then do a lightweight inference pass to compute per-token
top-K confidences and save everything to disk.

One .npz file per step: <rollout_log_dir>/step_NNNNN.npz
Keys in each file:
  step              : ()         int64
  completion_ids    : (B, T)     int32    – padded to max_completion_length
  completion_mask   : (B, T)     int8     – 1 for valid tokens, 0 for padding
  confidence        : (B, T)     float16  – C_{i,t} = -mean_{top-K} log π(v|ctx)
  topk_log_probs    : (B, T, K)  float16  – top-K log probs per token position
  topk_token_ids    : (B, T, K)  int32    – corresponding token IDs
  advantages        : (B,)       float32  – normalised group advantage
  is_correct        : (B,)       bool     – True if reward_answer_exact == 3.0
  gt_answer         : ()         bytes    – ground-truth answer string (UTF-8)
  extracted_answers : (B,)       bytes    – per-rollout extracted answer strings

Correctness alignment
---------------------
GRPOTrainer calls reward functions once for the full generation batch, but
calls compute_loss once per prompt-group. A flat list buffer therefore drifts
out of sync.

correctness_store is a dict  { full_completion_text → (is_correct, gt, extracted) }
populated by the reward wrapper in train.py.  _log_rollout decodes completion_ids
and looks up by full text.  Using the full text (not a prefix) avoids key
collisions when multiple rollouts for the same problem share a long identical
<think> prefix.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from trl import GRPOTrainer


class RolloutLoggerTrainer(GRPOTrainer):
    """
    GRPOTrainer that saves per-rollout data to disk at every training step.

    Extra __init__ kwargs:
        rollout_log_dir   (str):  directory for .npz files. Created if absent.
        correctness_store (dict): shared mutable dict
                                  { completion_text: (is_correct, gt_answer, extracted) }
                                  populated by reward_answer_exact in train.py.
        conf_top_k        (int):  top-K for confidence metric. Default 20.
        save_every_steps  (int):  log every N steps (default 1 = every step).
    """

    def __init__(self, *args, rollout_log_dir="rollout_logs",
                 correctness_store=None, conf_top_k=20, save_every_steps=1, **kwargs):
        self.rollout_log_dir  = rollout_log_dir
        self.correctness_store = correctness_store if correctness_store is not None else {}
        self._conf_top_k      = int(conf_top_k)
        self.save_every_steps = max(1, int(save_every_steps))
        os.makedirs(self.rollout_log_dir, exist_ok=True)
        self._log_step = 0
        super().__init__(*args, **kwargs)
        print(f"[RolloutLogger] conf_top_k={self._conf_top_k}  log_dir={self.rollout_log_dir}")

    # ── override compute_loss ─────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if self._log_step % self.save_every_steps == 0:
            try:
                self._log_rollout(model, inputs)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                err_path = os.path.join(self.rollout_log_dir, f"error_step_{self._log_step}.txt")
                with open(err_path, "w") as _f:
                    _f.write(tb)
                print(f"[RolloutLogger] WARNING step {self._log_step}: {e}")
        self._log_step += 1
        return loss

    # ── logging helper ────────────────────────────────────────────────────────

    def _log_rollout(self, model, inputs):
        completion_ids  = inputs["completion_ids"]   # (B, T)
        completion_mask = inputs["completion_mask"]  # (B, T)
        prompt_ids      = inputs["prompt_ids"]       # (B, P)
        prompt_mask     = inputs["prompt_mask"]      # (B, P)
        seq_advantages  = inputs["advantages"]       # (B,)
        B               = completion_ids.size(0)
        logits_to_keep  = completion_ids.size(1)

        # Forward pass → per-token confidence
        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            raw_out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits    = raw_out.logits[:, -(logits_to_keep + 1):-1, :].float().contiguous()
            log_probs = F.log_softmax(logits, dim=-1)
            topk_lp, topk_ids = torch.topk(log_probs, self._conf_top_k, dim=-1)
            confidence = -topk_lp.mean(dim=-1)   # (B, T)

        # ── resolve correctness via full-text-keyed store ─────────────────────
        tok = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
        mask_arr    = completion_mask.cpu().long().numpy().astype(np.int8)
        comp_ids_np = completion_ids.cpu().long().numpy().astype(np.int32)

        is_correct        = np.zeros(B, dtype=bool)
        gt_answer         = ""
        extracted_answers = [""] * B

        if self.correctness_store and tok is not None:
            for i in range(B):
                length = int(mask_arr[i].sum())
                text   = tok.decode(comp_ids_np[i, :length].tolist(),
                                    skip_special_tokens=False)
                val = self.correctness_store.pop(text, None)
                if val is not None:
                    ic, gt, ext = val
                    is_correct[i]        = ic
                    extracted_answers[i] = ext
                    if gt:
                        gt_answer = gt   # same for all rollouts in this group
                else:
                    # Mismatch: TRL may have stripped a trailing special token.
                    # Retry after stripping the last token from the decoded text.
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
        elif self.correctness_store and tok is None:
            print(f"[RolloutLogger] step {self._log_step}: no tokenizer – is_correct all False")

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
        print(f"[RolloutLogger] step {self._log_step:05d} → {os.path.basename(out_path)}  "
              f"correct={n_correct}/{B}  gt={gt_answer!r}  "
              f"mean_adv={float(seq_advantages.float().mean()):.3f}  "
              f"store_remaining={len(self.correctness_store)}")
