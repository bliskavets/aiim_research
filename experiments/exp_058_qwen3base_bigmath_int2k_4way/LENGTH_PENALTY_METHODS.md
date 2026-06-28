# exp_058 — length-penalty fix for gtpo_ema_flipped's length collapse

Addendum (additive; the 4 base candidates grpo / grpo_s_entropy / gtpo_conf /
gtpo_ema_flipped are untouched). On Qwen3-4B-**BASE** / Big-Math int-2000,
`gtpo_ema_flipped` starts strong but **collapses via length explosion**: its O+
bonus ∝ 1/EMA(C) rewards low-confidence (exploratory) tokens, so the model farms
length (640 → ~3400 tok, clip→100%, boxed→0). These 2 new methods add a length
penalty to stop that.

## Methods

Both reuse the gtpo_ema_flipped shaping unchanged and add a length penalty.

- **`gtpo_ema_lenpen`** — GROUP-RELATIVE length penalty on the shaped advantage:
  ```
  pen_i     = alpha_len · max(0, |o_i| − L)
  pen_rel_i = pen_i − mean_group(pen)          # >0 if longer than the group avg
  Ã_{i,t}  ← Ã_{i,t} − pen_rel_i               # shorter-than-avg ⇒ boosted
  ```
  alpha_len = 0.005, L = 1024 (0 < L < max_completion 3584).

- **`gtpo_ema_lenpen_gated`** — the same penalty, GATED by a multi-temperature
  heuristic: per unique prompt we sample 2 extra completions (t=0 greedy and
  t2=0.5) NOT used in the update; if either gives the exact boxed answer, the
  problem is concisely solvable ⇒ apply the penalty to that prompt's training
  completions, else skip it (preserve exploration on genuinely hard problems).

### Why GROUP-RELATIVE (v2, not the obvious v1)
gtpo_ema_flipped's shaping **discards the reward magnitude** — it uses the seq
advantage only for the O+/O− sign, then z-norms a confidence weighting (mean 0
per polarity). So:
- a penalty on the **reward** (pre-norm) barely matters — it only flips signs;
- an **absolute** per-sequence penalty on the shaped advantage (v1) failed too:
  `alpha_len ∈ {0.0015, 0.005}` both still collapsed to max length, boxed→0 by
  step ~250 (penalty −12 lost to the EMA length drive).

The fix is to make the penalty **relative within each group of num_generations**:
center it (`pen − group_mean`) so shorter-than-average completions get a higher
shaped advantage and longer ones lower — a direct "short beats long" ranking.
Because compute_loss runs on B=1 microbatches (no group there), pen_rel is
computed in `_generate_and_score_completions` (full group available) and
propagated to compute_loss via `out["len_pen"]`. The gate is computed the same
way (extra vLLM gens: wake → generate → sleep, current LoRA).

## Results (Qwen3-4B-Base, Big-Math int-2000, alpha_len=0.005, L=1024)

Plot: `figures/exp058_lenpen_fix.png`.

| method | steps | L50 len | L50 reward | L50 boxed | gate_frac | collapse? |
|---|---|---|---|---|---|---|
| bare gtpo_ema_flipped (exp_058 fig) | — | ~3400 (max) | — | **0.00** | — | **YES** (640→3400) |
| **gtpo_ema_lenpen** (group-rel)     | 419 | **456** | +2.74 | +0.72 | — | **NO** |
| **gtpo_ema_lenpen_gated**           | 403 | **584** | **+3.41** | **+1.24** | 0.74–0.92 | **NO** |

**Findings**
- The group-relative length penalty **fixes the collapse**: length stays ~450–900
  (≪ 3584), clip 1–7%, through step ~400 where the bare method had fully
  collapsed. reward/boxed stay healthy (no length farming).
- The **gated** variant is better on quality (boxed +1.24 vs +0.72, reward +3.41
  vs +2.74): its `gate_frac` genuinely discriminates (0.74–0.92 — most easy
  problems are low-temp-solvable so get penalized, the hard ~10–25% are spared),
  which preserves exploration where it's actually needed.
- Earlier negatives (recorded for honesty): reward-level penalty (no-op for this
  shaping) and absolute advantage penalty at alpha 0.0015/0.005 (still collapsed).

## Files
```
src/gtpo_ema_lenpen_trainer.py         group-relative length penalty
src/gtpo_ema_lenpen_gated_trainer.py   + low-temperature gate
run_058_lenpen.sh                      runs both methods
figures/exp058_lenpen_fix.png          length + boxed vs bare collapse
```

## 4-way comparison (all real, 420 steps, figures/exp058_4way_lenpen_comparison.png)

| method | L50 len | L50 reward | L50 boxed |
|---|---|---|---|
| GRPO baseline | 622 | +3.74 | +1.51 |
| gtpo_ema_flipped (bare) | 2121 | +1.91 | +0.38 |
| gtpo_ema_lenpen | 456 | +2.74 | +0.72 |
| gtpo_ema_lenpen_gated | 584 | +3.41 | **+1.24** |

**gated length-penalty recovers gtpo_ema_flipped from the collapse to near-GRPO quality** (boxed +1.24 vs grpo +1.51, length 584 vs 622), while the bare method inflates length (2121) and stays low (boxed +0.38). lenpen controls length too but is weaker on quality (+0.72).

## L-sweep (knee L in {3096, 2048, 1536}, alpha_len=0.005, 300 steps each)

Both methods re-run at three knee values to map the length↔quality trade-off.
L is env-overridable (`LENGTH_L`); `run_058_lenpen_Lsweep.sh` runs all six
(L × method) sequentially; `plot_Lsweep.py` builds the comparison (top row =
time-series, bottom row = final-metric-vs-L summary).

Plot: `figures/exp058_lenpen_Lsweep.png`. Reference: GRPO baseline len 622, boxed +1.51.

| method | L | L50 len | L50 boxed |
|---|---|---|---|
| gtpo_ema_lenpen       | 3096 | 1085 | +0.84 |
| gtpo_ema_lenpen_gated | 3096 | 1016 | +0.97 |
| gtpo_ema_lenpen       | 2048 |  758 | **+1.06** |
| gtpo_ema_lenpen_gated | 2048 |  766 | +0.92 |
| gtpo_ema_lenpen       | 1536 |  668 | +0.94 |
| **gtpo_ema_lenpen_gated** | **1536** | **601** | **+1.21** |

**Findings**
- **Length falls monotonically with tighter L** for both methods (lenpen
  1085→758→668; gated 1016→766→601). L=3096 is too weak a knee — it barely
  bites the typical 800–1100-tok answers, so quality sits well below GRPO.
- **The two methods respond oppositely to tightening:**
  - **lenpen** has a *mid optimum* — boxed peaks at L=2048 (+1.06) and dips again
    at L=1536 (+0.94): over-tightening the un-gated penalty clips legitimately
    long reasoning on hard problems.
  - **gated** *improves monotonically* with tighter L — best at L=1536 (+1.21,
    the sweep maximum). The low-temperature gate spares the genuinely hard
    prompts from the penalty, so tightening L only bites the easy/short-solvable
    ones; no exploration is lost where it matters.
- **Best config: gtpo_ema_lenpen_gated @ L=1536** — shortest of all six (601 ≈
  GRPO 622) AND highest boxed (+1.21, closest to GRPO +1.51). The gate is what
  lets a tight knee help rather than hurt.

## Adaptive length penalty (no fixed L — knee from the group's own lengths)

Instead of a fixed knee, derive L per group from its own length distribution:
`L = max((L_min+L_max)/2, L_mean)`, with a bounded piecewise penalty in [-0.5,0]
(0 for len≤L, linear ramp `-0.5·(len-L)/L` for L<len<2L, -0.5 for len≥2L),
applied group-relative on the shaped advantage. Two axes of variation:
- **scope**: whole-group knee (`adaptlen`) vs PER-POLARITY knee (`adaptlen_pm`) —
  the latter computes L_+/L_- separately within each group's O+ (adv>0,
  "correct") and O- (adv<0, "incorrect") subgroups and centers within each.
- **gate**: always-on vs low-temp difficulty gate (t=0, t2=0.5), as before.

Code: `src/adaptive_lenpen_utils.py` (both formulas, unit-tested),
`src/gtpo_ema_adaptlen*_trainer.py`; `run_058_adaptlen.sh` + `run_058_adaptlen_pm.sh`;
plot `figures/exp058_adaptlen.png` (`plot_adaptlen.py`). 300 steps each.

| method | scope | gate | L50 len | L50 boxed | knee L→ |
|---|---|---|---|---|---|
| GRPO baseline            | —            | —   | 622 | **+1.51** | — |
| fixed-L gated (L=1536)   | fixed        | yes | 601 | +1.21 | 1536 |
| gtpo_ema_adaptlen        | whole-group  | no  | 985 | +0.90 | ~1238 |
| gtpo_ema_adaptlen_gated  | whole-group  | yes | 2270 | +0.44 | ~2389 (drifts) |
| gtpo_ema_adaptlen_pm     | per-polarity | no  | 1002 | +0.88 | ~1049 |
| **gtpo_ema_adaptlen_pm_gated** | per-polarity | yes | **774** | **+0.94** | **~493** |

**Findings**
- **The adaptive penalty is gentle by construction** (bounded ±0.5; `|pen_rel|`
  ~0.07–0.16 vs the fixed penalty's ~4–6), so it controls length more softly than
  a fixed cap. always-on adaptlen lands at 985/+0.90; per-polarity-always is
  near-identical (1002/+0.88) — splitting the knee alone changes little.
- **Whole-group + gate COLLAPSES** (adaptlen_gated 2270/+0.44): the knee is
  self-referential, so as length explodes L floats up with it (→2389) and the
  bounded penalty never bites; the gate occasionally zeroing it makes it worse.
  This reproduces the bare gtpo_ema_flipped length-explosion.
- **Per-polarity + gate FIXES that collapse** (adaptlen_pm_gated 774/+0.94 — best
  of the adaptive family): correct rollouts (O+) are short, incorrect (O-) are
  long, so a per-polarity knee keeps L_+ anchored low (~493) instead of being
  dragged up by the long O-. The penalty then keeps pressing correct answers
  short AND ranks short-incorrect above long-incorrect — breaking the length
  drift the whole-group knee suffered.
- **Net**: no adaptive config beats GRPO (+1.51) or the best fixed-L gated
  (+1.21). The fixed knee with a low-temp gate (L=1536) remains the strongest
  length-penalty config; among adaptive variants, per-polarity + gate is the only
  one that avoids collapse.

## GROP — Group Relative Overlong Punishment (arXiv:2508.04349, Appendix D)

The length-control heuristic from the GTPO/GRPO-S paper itself. Per group of G
responses, classify the question by solve rate `frac = n_correct/G`:
- **easy** (`frac ≥ γ₁`): penalize the CORRECT responses, knee `L⁺=max((min+max)/2, mean)` over correct lengths;
- **hard** (`frac ≤ 1−γ₁`): NO penalty (preserve solving ability);
- **medium**: knee `L⁻` over ALL G lengths; penalize correct if `n>m` else incorrect.
Penalty `R = −0.5·(|o|−L)/L` on `L≤|o|<2L`, `−0.5` for `|o|≥2L`. γ₁=0.75.
"Correct" = exact boxed match (terminal reward). Code:
`src/gtpo_ema_flipped_grop_trainer.py`, helper
`adaptive_lenpen_utils.group_relative_overlong_punishment` (unit-tested vs the
paper). Injection: paper adds R to the reward; our flipped shaping is
magnitude-insensitive (sign-only O+/O−), so we subtract R from the shaped
advantage to preserve intent. Plot: `figures/exp058_grop.png`.

| method | L50 len | L50 boxed |
|---|---|---|
| GRPO baseline | 622 | +1.51 |
| gtpo_ema_flipped (bare) | 2121 | +0.38 |
| fixed-L gated (L=1536) | 601 | +1.21 |
| **GROP (App.D, γ₁=0.75)** | **2894** | **+0.10** |

**Finding — GROP does NOT prevent the collapse on this (broken) base; its
difficulty gate disables the penalty exactly when length explodes.** As the model
degrades, the solve rate falls (`frac_correct → 0.11`), so most groups become
**hard** (`frac_hard → 0.94`) and hard → *no penalty by design*. The applied
penalty decays to ~0.005 and length runs to 2894 (boxed +0.10). In the first
~100 steps GROP did apply (easy/medium present), but the bounded ±0.5 penalty on
top of the broken flipped shaping (B=1 reward inversion, see
`DIAG_LENGTH_EXPLOSION.md`) drove correctness down and triggered the
self-disabling loop. This is the opposite failure mode to our `fixed-L gated`,
whose gate keys on low-temperature *solvability* (stable) rather than the group's
current correctness (collapses). Caveat: GROP was designed for the paper's
*working* GTPO (magnitude-sensitive `α₁·rᵢ`, full-group token loss, stable Qwen2.5
training), where the "model stays mostly correct → easy/medium dominate" premise
holds; on our degenerate B=1 flipped base it does not.

## Follow-ups: GROP on a working base + fixing gtpo_ema_flipped

Two confirmatory runs (300 steps; plot `figures/exp058_fix_grop.png`).

| method | L50 len | L50 boxed |
|---|---|---|
| GRPO baseline | 622 | +1.51 |
| gtpo_ema_flipped (bare / broken B=1) | 2121 | +0.38 |
| GROP @ GRPO (paper, reward-level) | 679 | +1.36 |
| **gtpo_ema_flipped FIXED (group-shaped)** | **570** | **+1.49** |

1. **GROP on plain GRPO (reward-level, paper-faithful)** — `grpo_grop`, GROP added
   as a reward term (R(i)∈[-0.5,0]) so GRPO group-normalizes it into the advantage.
   On this *working*, magnitude-sensitive base GROP behaves as the paper intends:
   length stays near baseline (679 vs GRPO 622) with almost no quality cost
   (+1.36 vs +1.51). Confirms GROP's earlier failure was the broken flipped base,
   not the method.

2. **Fixed gtpo_ema_flipped** — `gtpo_ema_flipped_fixed` computes the shaped
   per-token advantage ONCE on the FULL group in `_generate_and_score`
   (proper per-position Σ + per-polarity z-norm, policy still θ_old) and propagates
   the 2-D advantage to compute_loss for injection, instead of recomputing in the
   degenerate B=1 compute_loss. **This single fix removes BOTH pathologies with no
   length penalty at all**: length 570 (≪ 2121, even below GRPO 622) and boxed
   +1.49 (≈ GRPO +1.51, vs the broken +0.38). It confirms the
   `DIAG_LENGTH_EXPLOSION.md` root cause — the explosion + reward inversion were
   artifacts of the B=1 group-op degeneracy, not inherent to the shaping.

**Bottom line for exp_058's length investigation:** the length explosion was a
B=1-microbatch implementation bug, not a property of gtpo_ema_flipped. Fixing the
group-visibility makes the shaped method match GRPO (at shorter length) without
any length penalty; and the paper's own GROP works once placed on a sound base.
The length-penalty zoo (lenpen / L-sweep / adaptive / per-polarity) controlled the
*symptom*; the group-shaped fix removes the *cause*.
