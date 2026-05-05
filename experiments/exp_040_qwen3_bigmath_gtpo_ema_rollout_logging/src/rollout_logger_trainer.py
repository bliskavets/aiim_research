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
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from trl import GRPOTrainer


class RolloutLoggerTrainer(GRPOTrainer):
    """
    GRPOTrainer that saves per-rollout data to disk at every training step.

    Overrides compute_loss (not _compute_loss) because unsloth's patched
    GRPOTrainer skips _compute_loss entirely.

    Extra __init__ kwargs:
        rollout_log_dir    (str):  directory for .npz files. Created if absent.
        correctness_buffer (list): shared mutable list populated by the
                                   reward_answer_exact wrapper in train.py.
                                   Drained B entries per compute_loss call.
        conf_top_k         (int):  top-K for confidence metric. Default 20.
                                   Named conf_top_k to avoid collision with
                                   GRPOTrainer's vLLM sampling top_k.
        save_every_steps   (int):  log every N steps (default 1 = every step).
    """

    def __init__(self, *args, rollout_log_dir="rollout_logs",
                 correctness_buffer=None, conf_top_k=20, save_every_steps=1, **kwargs):
        self.rollout_log_dir    = rollout_log_dir
        self.correctness_buffer = correctness_buffer if correctness_buffer is not None else []
        self._conf_top_k        = int(conf_top_k)   # plain Python int, no conflict w/ GRPOTrainer
        self.save_every_steps   = max(1, int(save_every_steps))
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

        # Forward pass through model to get logits → per-token confidence
        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.no_grad():
            raw_out = model(input_ids=input_ids, attention_mask=attention_mask)
            # logit[i] predicts token[i+1]; we want logits for completion positions
            # → last (T+1) outputs, drop the last one
            logits    = raw_out.logits[:, -(logits_to_keep + 1):-1, :].float().contiguous()
            log_probs = F.log_softmax(logits, dim=-1)
            # Use _conf_top_k directly (plain Python int) — no min() needed;
            # Qwen3 vocab=151936 >> 20, so topk never exceeds vocab
            topk_lp, topk_ids = torch.topk(log_probs, self._conf_top_k, dim=-1)
            confidence = -topk_lp.mean(dim=-1)   # (B, T)

        # Drain correctness labels from the shared buffer
        is_correct = np.zeros(B, dtype=bool)
        if len(self.correctness_buffer) >= B:
            labels = self.correctness_buffer[:B]
            del self.correctness_buffer[:B]
            is_correct = np.array(labels, dtype=bool)
        elif self.correctness_buffer:
            n = len(self.correctness_buffer)
            is_correct[:n] = np.array(self.correctness_buffer[:n], dtype=bool)
            del self.correctness_buffer[:n]

        out_path = os.path.join(self.rollout_log_dir, f"step_{self._log_step:05d}.npz")
        np.savez_compressed(
            out_path,
            step           = np.array(self._log_step, dtype=np.int64),
            completion_ids = completion_ids.cpu().long().numpy().astype(np.int32),
            completion_mask= completion_mask.cpu().long().numpy().astype(np.int8),
            confidence     = confidence.cpu().numpy().astype(np.float16),
            topk_log_probs = topk_lp.cpu().numpy().astype(np.float16),
            topk_token_ids = topk_ids.cpu().long().numpy().astype(np.int32),
            advantages     = seq_advantages.float().cpu().numpy().astype(np.float32),
            is_correct     = is_correct,
            prompt_ids     = prompt_ids[0].cpu().long().numpy().astype(np.int32),
            prompt_mask    = prompt_mask[0].cpu().long().numpy().astype(np.int8),
        )
        n_correct = int(is_correct.sum())
        print(f"[RolloutLogger] step {self._log_step:05d} → {os.path.basename(out_path)}  "
              f"correct={n_correct}/{B}  mean_adv={float(seq_advantages.float().mean()):.3f}")
