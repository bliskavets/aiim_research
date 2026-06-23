# exp_058 — why bare gtpo_ema_flipped explodes in length (diagnostic)

Re-ran the bare `gtpo_ema_flipped` (identical math, seed 3407 → reproduces the
collapse: length 640→**3584** by step ~300, boxed +1.5→**−0.4**) with per-step
logging of generations, rewards and the shaped per-token advantage. Code:
`src/gtpo_ema_flipped_diag_trainer.py`, `analyze_diag.py`. Figure:
`figures/exp058_diag_mechanism.png`. Raw dumps: `diag/*.jsonl` (gitignored).

## TL;DR — root cause

**unsloth feeds `compute_loss` ONE completion at a time (B=1 microbatch), but the
flipped-EMA shaping is defined over a full group of `num_generations` completions.**
With a single completion the shaping degenerates, and the degenerate signal both
**inverts the reward** and carries a **mild positive length bias** → the model
rambles to the 3584-token cap and abandons concise correct answers.

## Evidence (all from the 420-step diagnostic, B=1 confirmed: 1680 records = 4×420)

| window | mean len | O+ adv (correct) | O− adv (wrong) | O+ − O− | corr(len, adv) |
|---|---|---|---|---|---|
| early (0–100)  |  865 | −0.47 | −0.80 | **+0.34** (correct ranked above wrong) | +0.08 |
| mid (150–250)  | 2516 | −0.47 | −0.19 | **−0.29** (INVERTED) | +0.10 |
| late (300–419) | 3117 | −0.47 | −0.13 | **−0.34** (INVERTED) | +0.12 |

1. **Per-token advantage is CONSTANT within a completion** (all 10 position-bins
   identical) — with B=1 the per-position group sums collapse, so there is no
   within-sequence token structure; the signal is one scalar per completion.
2. **z-norm over a near-constant single completion is degenerate**: it divides by
   ~0 std, so most completions get ≈0 but a tail is **blown up to ±6**
   (|adv|>3 in ~3% of completions; max 5.96) — noise, not token quality.
3. **Reward inversion**: genuine O+ (correct, seq_adv>0) completions are pinned at
   adv ≈ −0.47 the ENTIRE run, while genuine O− (wrong) completions drift UP
   (−0.80→−0.13). They cross around step ~100: from mid-training on, **wrong
   completions receive a HIGHER advantage than correct ones** (O+−O− = −0.34).
4. **Length incentive**: corr(completion length, shaped advantage) is consistently
   **positive (+0.08…+0.12)** — longer completions get a higher advantage,
   compounded over hundreds of steps.
5. **What the collapse looks like**: late incorrect rollouts fill the 3584 cap with
   degenerate repetition, e.g. a tail of `".xtext\n.xtext\n.xtext\n…"` and no
   `\boxed{}` — the model loops low-content filler to max length.

## Why B=1 breaks this specific shaping

`compute_gtpo_ema_flipped_advantages` (see `src/ema_flipped_utils.py`) does, per
timestep t and polarity:
- O+: `bonus_t = (1/EMA_t / Σ_{k∈O⁺_t} 1/EMA_k) · d_t`, `shaped = α₁ + α₂·bonus`
- O−: `penalty_t = (EMA_t / Σ_{k∈O⁻_t} EMA_k) · h_t`, `shaped = −(α₁ + α₂·penalty)`

then z-normalises Ã⁺ and Ã⁻ **separately over their active tokens**. These are
group operations: the Σ over O+/O- at position t and the per-polarity z-norm only
make sense when several completions of the same prompt are present together. With
B=1:
- the only completion in its polarity ⇒ the normalised weight is 1 ⇒ `shaped`
  becomes the **constant** `±(α₁+α₂) = ±1` across all its tokens;
- z-norm of a constant ⇒ 0/EPS ⇒ **0, or a blow-up** wherever a few tokens differ
  slightly (EMA=0 at the start, tag-masked structural tokens). The result is a
  per-sequence constant that no longer encodes "better/worse than my group peers"
  — the entire point of the GTPO group-relative signal is lost.

This is the **same B=1-microbatch failure** already seen for `grpo_s_conf`
(`view(-1,G)` crash) and noted in the project memory: under unsloth, when
`bs×ga` is a multiple of `num_generations`, the trainer keeps `bs=1`, so any
shaping that needs the whole group inside `compute_loss` is silently degenerate.

## Why the length-penalty fixes worked anyway

The working fixes (`gtpo_ema_lenpen`, `_gated`, the L-sweep, per-polarity adaptive)
compute their length penalty in `_generate_and_score_completions`, where the FULL
group IS available, and propagate a per-completion scalar to `compute_loss`. They
add a genuine group-relative "short beats long" ranking on top of the (degenerate)
shaped advantage — which is exactly why they restrain the length drift while the
bare method cannot. They do not, however, repair the underlying reward inversion,
which is why no shaped config beats GRPO.

## Implication

To make gtpo_ema_flipped itself correct (not just length-controlled), the shaped
advantage must be computed where the full group is visible
(`_generate_and_score_completions`) and propagated per-token, instead of being
recomputed inside the B=1 `compute_loss`. That is a method change, not a tuning
knob — filed as a follow-up.
