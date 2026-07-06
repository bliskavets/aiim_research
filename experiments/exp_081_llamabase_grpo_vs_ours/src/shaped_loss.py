"""
shaped_loss.py
--------------
Helpers to make per-token reward shaping actually run on trl 0.23.1 + unsloth,
ported from exp_057's SHAPING_BYPASS_BUGFIX.

Why injection (not a hand-written loss): unsloth replaces trl.GRPOTrainer with a
compiled `_UnslothGRPOTrainer` whose `compute_loss` is self-contained and NEVER
calls `_compute_loss`. So a subclass `_compute_loss` (where the old shaping lived)
is dead code — the shaped methods silently ran plain GRPO (verified: exp_056's
shaped runs logged ZERO `<method>/*` metrics). The real memory-efficient gradient
is produced inside unsloth's chunked custom autograd, invoked by the compiled
`compute_loss`. So we let the compiled loss own the gradient and only INJECT a
per-token shaped advantage.

Key fact: `grpo_compute_loss` does `if advantages.dim() == 1: advantages =
advantages.unsqueeze(1)` and otherwise uses `advantages` element-wise against the
per-token logps. So a 2-D `(B, W)` advantages tensor is consumed per-token. W is
the loss's internal (left-packed) grid width = stored old/ref_per_token_logps
width, which can exceed the completion width Lk. Real completion tokens occupy the
LAST Lk columns of that grid, so a completion-grid (B, Lk) advantage is LEFT-padded
with (W - Lk) zeros.

A100-80GB note (vs exp_057's H200-143GB): the no-grad confidence/entropy forward
is chunked over the batch dim (see confidence_from_model_chunked /
entropy_from_model_chunked) so the full (B, Lk, V) fp32 logits tensor over Qwen3's
~152k vocab is never materialized at once.
"""
import torch
import torch.nn.functional as F


def token_entropy_from_logits(logits, chunk=512):
    """Shannon entropy per token from logits (b, Lk, V), chunked over the seq dim
    to bound memory. Returns (b, Lk) float32. No grad."""
    b, Lk, _ = logits.shape
    ent = torch.empty(b, Lk, device=logits.device, dtype=torch.float32)
    for i in range(0, Lk, chunk):
        lp = torch.log_softmax(logits[:, i:i + chunk, :].float(), dim=-1)
        ent[:, i:i + chunk] = -(lp.exp() * lp).sum(-1)
    return ent


@torch.no_grad()
def entropy_from_model_chunked(model, input_ids, attention_mask, logits_to_keep,
                               pass_logits_to_keep=False, micro_bs=2, seq_chunk=512):
    """Per-token Shannon entropy (B, Lk) on the completion grid, computed by
    running the model forward in micro-batches over the batch dim (so the full
    (B, Lk, V) logits are never held at once) and reducing each micro-batch's
    logits to entropy. No grad."""
    B = input_ids.size(0)
    chunks = []
    for s in range(0, B, micro_bs):
        e = min(s + micro_bs, B)
        mi = {"input_ids": input_ids[s:e], "attention_mask": attention_mask[s:e]}
        if pass_logits_to_keep:
            mi["logits_to_keep"] = logits_to_keep + 1
        logits = model(**mi).logits[:, :-1, :]      # next-token alignment
        logits = logits[:, -logits_to_keep:, :]     # (b, Lk, V)
        chunks.append(token_entropy_from_logits(logits, chunk=seq_chunk))
        del logits
    out = torch.cat(chunks, dim=0)                  # (B, Lk)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # defrag the freed forward logits before the backward
    return out


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
    (B, width) with zeros (real completion tokens are the LAST Lk columns; the left
    padding lands on masked positions)."""
    Lk = token_adv.shape[1]
    if width == Lk:
        return token_adv
    if width < Lk:
        return token_adv[:, -width:]
    return F.pad(token_adv, (width - Lk, 0))


def inject_advantages(inputs, advantages, logits_to_keep):
    """Return a shallow-copied inputs dict with `advantages` replaced.

    - 1-D (B,) advantages (seq-level, e.g. GRPO-S) pass through unchanged.
    - 2-D (B, Lk) per-token advantages are left-padded to the loss grid width.
    """
    new_inputs = dict(inputs)
    if advantages.dim() == 1:
        new_inputs["advantages"] = advantages
    else:
        new_inputs["advantages"] = widen_token_advantages(
            advantages, loss_grid_width(inputs, logits_to_keep))
    return new_inputs
