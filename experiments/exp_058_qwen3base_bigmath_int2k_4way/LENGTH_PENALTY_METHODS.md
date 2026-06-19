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
