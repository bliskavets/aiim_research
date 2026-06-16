"""
shaped_loss.py
--------------
Helpers to make per-token reward shaping work on trl 0.23.1 + unsloth.

Why injection (not a hand-written loss): on this stack the real, memory-efficient
gradient for the policy logps is produced INSIDE unsloth's chunked custom autograd
(`UnslothEfficientGRPO`), invoked by the compiled `compute_loss`. Recomputing
logps ourselves either OOMs (full logits with grad, ~28 GB extra) or comes back
detached (`_get_per_token_logps_and_entropies` returns no-grad logps). So we let
the compiled loss own the gradient and only INJECT a per-token advantage.

Key fact: `grpo_compute_loss` does `if advantages.dim() == 1: advantages =
advantages.unsqueeze(1)` and otherwise uses `advantages` element-wise against the
per-token logps. So a 2-D `advantages` of shape (B, W) is consumed as a per-token
advantage. W is the loss's internal (left-packed) grid width = stored
old/ref_per_token_logps width (verified: 6293 vs completion 6144). The real
completion tokens occupy the LAST `logits_to_keep` columns of that grid, so we
left-pad our completion-grid (B, Lk) advantage with (W - Lk) zeros.

`forward_completion_logits` gives the completion-grid logits (no grad at call
site) so the trainer can compute confidence / entropy for the shaping.
"""
import torch
import torch.nn.functional as F


def forward_completion_logits(trainer, model, input_ids, attention_mask, logits_to_keep):
    """Full logits over the last `logits_to_keep` (completion) positions: (B, Lk, V)."""
    mi = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "logits_to_keep" in trainer.model_kwarg_keys:
        mi["logits_to_keep"] = logits_to_keep + 1
    logits = model(**mi).logits          # (B, L, V)
    logits = logits[:, :-1, :]           # next-token alignment
    logits = logits[:, -logits_to_keep:, :]
    return logits


def token_entropy(logits, chunk=512):
    """Shannon entropy per token from logits (B, Lk, V), chunked over the
    sequence dim to bound memory. Returns (B, Lk) float32. No grad."""
    B, Lk, _ = logits.shape
    ent = torch.empty(B, Lk, device=logits.device, dtype=torch.float32)
    for i in range(0, Lk, chunk):
        lp = torch.log_softmax(logits[:, i:i + chunk, :].float(), dim=-1)
        ent[:, i:i + chunk] = -(lp.exp() * lp).sum(-1)
    return ent


def loss_grid_width(inputs, logits_to_keep):
    """Width W of the compiled loss's per-token grid = stored old/ref logps width
    (which the loss reconstructs via max_left_pad). Falls back to logits_to_keep."""
    for k in ("old_per_token_logps", "ref_per_token_logps"):
        v = inputs.get(k)
        if v is not None:
            return v.shape[1]
    return logits_to_keep


def widen_token_advantages(token_adv, width):
    """Left-pad a completion-grid (B, Lk) per-token advantage to the loss grid
    (B, width) with zeros (real completion tokens are the LAST Lk columns of the
    loss grid; the left padding lands on masked positions)."""
    Lk = token_adv.shape[1]
    if width == Lk:
        return token_adv
    if width < Lk:
        return token_adv[:, -width:]
    return F.pad(token_adv, (width - Lk, 0))


def inject_advantages(inputs, advantages, logits_to_keep):
    """Return a shallow-copied inputs dict with `advantages` replaced.

    - 1-D (B,) advantages (seq-level, e.g. GRPO-S) pass through unchanged; the
      compiled loss broadcasts them as usual.
    - 2-D (B, Lk) per-token advantages are left-padded to the loss grid width.
    """
    new_inputs = dict(inputs)
    if advantages.dim() == 1:
        new_inputs["advantages"] = advantages
    else:
        new_inputs["advantages"] = widen_token_advantages(advantages, loss_grid_width(inputs, logits_to_keep))
    return new_inputs
