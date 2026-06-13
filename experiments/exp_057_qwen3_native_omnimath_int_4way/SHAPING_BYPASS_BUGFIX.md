# Bug: per-token shaping was silently bypassed (and the fix)

Found 2026-06-12 in exp_057 (Qwen3-4B, Omni-MATH int subset) after the `grpo`
and `gtpo_ema_flipped` reward curves came out **suspiciously identical**. The
shaped methods were not shaping anything — they ran plain GRPO. This note records
the root cause, the evidence, and the fix, because it very likely also affects
**exp_055** and the wider exp_049→056 shaping arc *on this unsloth stack*.

Stack: unsloth 2026.3.7 · unsloth_zoo 2026.4.9 · trl 0.23.1 · torch 2.9.1+cu128 ·
vllm 0.16.0 · transformers 4.57.6 · NVIDIA H200 143 GB.

## Symptom

`gtpo_ema_flipped` (912 steps) and `grpo` (492 steps) produced step-matched
reward/boxed/clip curves within noise of each other (corr-level identical) — what
you'd expect from two identical GRPO runs off the same seed.

## Root cause — four layers

The shaping trainers (`GRPOSTrainer`, `GTPOConfTrainer`, `GTPOEMAFlippedTrainer`)
injected their per-token shaping by overriding **`_compute_loss`**. That is the
right hook on a plain `trl.GRPOTrainer`. On this stack it failed four ways:

1. **unsloth bypasses `_compute_loss`.** unsloth replaces `trl.GRPOTrainer` with a
   compiled `_UnslothGRPOTrainer` (written to `unsloth_compiled_cache/UnslothGRPOTrainer.py`)
   whose `compute_loss` is **self-contained and never calls `_compute_loss`**. HF
   Trainer calls `compute_loss`, so the subclass `_compute_loss` (all the shaping)
   was dead code. Proof:
   - `GTPOEMAFlippedTrainer.compute_loss.__qualname__ == _UnslothGRPOTrainer.compute_loss`
     (the override resolves to the compiled method, not ours);
   - the real `gtpo_ema_flipped` run logged **zero** `gtpo_ema_flipped/*` shaped
     metrics. → it was plain GRPO.

2. **`self.top_k` clobbered to `None`.** `GRPOTrainer.__init__` defines its own
   `self.top_k` (vLLM sampling top-k, default None). The trainers set `self.top_k`
   in `__init__` *before* `super().__init__()`, so it was overwritten → the
   confidence top-k became None and crashed the moment the shaping ran.

3. **Hand-written `_compute_loss` is incompatible with trl 0.23.1.** Even after
   routing `compute_loss → _compute_loss`, it crashed on a shape mismatch:
   `_get_per_token_logps_and_entropies` returns logps on a **left-packed grid**
   `W = Lk + max_left_pad` (e.g. 6293 vs completion `Lk = 6144`), matching the
   stored `old/ref_per_token_logps` grid (corr(new, old) == 1.0), **not** the
   `completion_mask` / `completion_ids` grid. And it returns those logps
   **detached** — the real, memory-efficient gradient is produced only inside
   unsloth's chunked custom autograd (`UnslothEfficientGRPO`). Recomputing
   grad-logps by hand from full logits OOMs (~28 GB extra on H200).

4. **Method-design caveat (independent of the stack).** Even once shaping runs,
   `ema_flipped_utils._znorm_over_active` z-normalizes each polarity group, which
   (a) **washes out `alpha1`/`alpha2`** (changing 0.9/0.1 → 0.1/0.9 moves the
   output by 3e-5) and (b) **centers each polarity at mean 0**, discarding the
   sequence-reward magnitude — so the shaped advantage correlates with the GRPO
   seq-advantage at only ~0.05. The shaping is "doing something", but that
   something is largely confidence structure orthogonal to reward. (Tag mask is
   fine: 4 single special-token ids, ~2.5% of a real completion.)

## The fix — inject the advantage, don't reimplement the loss

Rather than fight unsloth's compiled loss, let it own the (memory-efficient,
correct) gradient and only **inject the per-token shaped advantage**. Key fact:
`grpo_compute_loss` does `if advantages.dim() == 1: advantages = advantages.unsqueeze(1)`
and otherwise uses `advantages` element-wise against the per-token logps — so a
**2-D `(B, W)` advantages tensor is consumed per-token natively**.

Each shaping trainer now overrides `compute_loss` (top of MRO, so it is actually
called) and:
1. does ONE no-grad full-logits forward on the completion grid → confidence
   (GTPO) or Shannon entropy (GRPO-S; the chunked helper returns None for
   entropy here, which had silently degenerated to a constant 0.24);
2. computes the per-token shaped advantage `(B, Lk)` on the completion grid (+ the
   tag mask);
3. **left-pads it to the loss grid width** `W = old/ref_per_token_logps.shape[1]`
   (real completion tokens are the last `Lk` columns of that grid) and writes it
   into `inputs["advantages"]`;
4. calls `super().compute_loss(...)` — unsloth's compiled loss then computes the
   chunked grad-logps and applies clipping/KL with our per-token advantage.

GRPO-S is seq-level, so it injects a 1-D `(B,)` shaped advantage (passed through
unchanged). Shaping params are now set **after** `super().__init__()` so
`self.top_k` survives.

Files: `src/shaped_loss.py` (helpers: `forward_completion_logits`,
`token_entropy`, `loss_grid_width`, `widen_token_advantages`, `inject_advantages`)
and the `compute_loss` override in `src/grpo_s_trainer.py`,
`src/gtpo_conf_trainer.py`, `src/gtpo_ema_flipped_trainer.py`.

## Verification (live, on the running fixed runs)

- shaped metrics (`<method>/mean_confidence` etc.) present on **67/67** logged
  steps → the override runs and the advantage is injected every step;
- `grad_norm` > 1e-3 on **67/67** steps (mean ≈ 0.062), not the dead ~1e-6 of the
  bypass;
- distinctive proof the shaping changes the gradient: on steps where all rollouts
  of a prompt got the same reward (`frac_reward_zero_std = 1.0`), plain GRPO had
  `grad_norm ≈ 5e-6` (dead), but fixed `gtpo_conf` has `grad_norm ≈ 0.046` — the
  per-token confidence signal drives a gradient where seq-level GRPO has none;
- `grpo_s_entropy` now logs real per-token entropy (≈0.225, varies) instead of
  the constant-0.24 fallback;
- no OOM (peak ≈ 132 / 143 GB), 19 unit tests pass (incl.
  `tests/test_shaping_diagnostics.py`).

## How to not get burned again

**Always check that a shaped run logs its `<method>/...` metrics.** If they are
absent, the shaping is being bypassed and the run is plain GRPO regardless of
which trainer class you instantiated. Diagnostics: `diagnose_shaping.py`
(synthetic), `tests/test_shaping_diagnostics.py`.

## Implications

The first exp_057 "shaped" results (and almost certainly exp_055's "all methods
within ±0.13" null, same stack) were plain GRPO. The exp_049→056 shaping arc on
this unsloth version needs re-auditing the same way (check for shaped metrics in
the logs). A corrected exp_057 sweep with shaping actually applied is running.
