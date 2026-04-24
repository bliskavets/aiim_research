# Session Context — aiim_research GRPO modifications (updated 2026-04-24)

This file captures the state of the project after the sessions that
produced exp_017..033. Drop it into a fresh Claude Code session by
saying "read /mnt/data/aiim_research/SESSION_CONTEXT.md" and then
"load memory about this project".

---

## 0. Environment

- Repo: `/mnt/data/aiim_research`  (github.com/bliskavets/aiim_research, branch main)
- Model target: `meta-llama/Llama-3.2-3B-Instruct` with LoRA r=64
- Main disk is tiny; everything lives on `/mnt/data`
- GPU: single A100 80GB PCIe
- Docker image: `unsloth/unsloth:latest` (pulls to ~28GB)
  - `unsloth_docker` (older `2025.7.11-pt2.7.0-cu126`) is still up as a
    general-purpose container; do NOT use it for experiments that need
    the 2026.x stack.

### How to launch an experiment (pattern from exp_024 onward)
Key gotchas discovered the hard way:
1. `unsloth/unsloth:latest` has `/usr/local/bin/entrypoint.sh` which
   **swallows the CMD** and starts supervisord (jupyter/ollama/sshd).
   Always pass `--entrypoint /bin/bash` on `docker run`.
2. The image ships `numpy 2.4.1`, which is incompatible with `numba`
   (used by vllm ngram proposer). Pin `numpy<2.3`.
3. Image uid is 1001; host `mle` is uid 1000 → need `--user root`
   (and `chown -R mle:mle` at the end, which the launcher does).
4. `uv pip install unsloth==2026.3.7` pulls torch 2.10 and transformers
   5.3, which breaks the pre-installed vllm 0.11.2. Use
   `uv venv --system-site-packages` and then
   `uv pip install --no-deps unsloth==2026.3.7 unsloth_zoo`.
5. Correct overlay stack: unsloth 2026.3.7, unsloth_zoo 2026.4.9,
   trl 0.23.1, torch 2.9.0+cu128, vllm 0.11.2, transformers 4.57.1,
   numpy 2.2.6. Matches what exp_005 ran on.

See `experiments/exp_028_bigmath_int2k_flipped_gtpo_ema/run_028.sh`
for the canonical launcher.

---

## 1. Theory and methods used

All modifications extend GRPO via per-token reward shaping. Core
reference is `experiments/proof/GTPO-EMA-full.txt` (Def 1.1–1.5,
Prop 2.3 conservation, Prop 3.1 variance reduction).

Top-k "confidence" metric used in exp_005/006/010/025..033:

    C_{i,t} = - mean_{v ∈ top-k}( log π(v | context) )    k=20

Empirical fact established in exp_025/026 tests: **C grows with
peakedness, not with entropy**. A one-hot-ish distribution gives
C ≈ 9.5; a flat-uniform one gives ≈ 4.6. That inverts the narrative
intuition in Def 1.1 prose. The exp_026 "flipped" variant was
motivated by this: swap which group uses 1/EMA(C) vs EMA(C).

### exp_025 (pure-proof, not flipped)
Reward shaping per Def 1.4:

    O+: r̃⁺_{i,t} = α1·r_i + α2·(EMA_{i,t} / Σ_{k∈O⁺_t} EMA_{k,t}) · d_t
    O-: r̃⁻_{j,t} = -α1   + α2·(1/EMA_{j,t} / Σ_{k∈O⁻_t} 1/EMA_{k,t}) · h_t · (-1)

with α1+α2=1 for Prop 2.3 conservation, then **separate z-norm** on
O+ and O- (Def 1.5). Files: `experiments/exp_025_pure_proof_gtpo_ema/`.

### exp_026 and exp_028..033 (flipped)
Same skeleton as exp_025 but the weights are **swapped** between
groups:

    O+: bonus_{i,t}   = (1/EMA(C)_{i,t} / Σ 1/EMA) · d_t
    O-: penalty_{j,t} = (EMA(C)_{j,t}   / Σ EMA)   · h_t

Conservation still holds. Motivation: reward "flat/hesitant" tokens
in correct paths and penalize "peaked/confident" tokens in wrong
paths, which is the prose narrative of the proof.

### exp_028 onward (Big-Math int-2000)
On Big-Math the **O+/O- split** is driven by `reward_answer_exact`
via a module-level `reward_cache._CACHE.mask` — pattern from exp_022.
Threshold `answer_exact >= 1.0`:
  O+ = {+3.0 exact, +1.5 strip, +1.0 within-10%}
  O- = {+0.5 within-20%, 0.0 no-format, -1.5 wrong}

### Reward composition (reward ceiling is 9.5)
- `reward_format_exact`       max +3.0
- `reward_format_approximate` max +2.0
- `reward_answer_exact`       max +3.0 (also -1.5 for wrong-in-format)
- `reward_answer_numeric`     max +1.5
Sum = 9.5 at perfect behavior.

---

## 2. Experiments done (relevant ones since exp_017)

### Big-Math (1000 steps, 16 gens, bs=4)
  exp_017  GRPO baseline                 peak 9.5 @ 242
  exp_018  GTPO-EMA                      peak 9.5 @ 203
  exp_019  GRPO-S entropy                peak 9.5 @ 215
  exp_020  GTPO entropy                  peak 9.5 @ 187
  exp_021  GTPO-Conf                     peak 9.5 @ 230
  exp_022  GTPO binary O+/O-             peak 9.5 @ 202
  exp_023  GTPO-EMA binary               peak 9.5 @ 215

### GSM-8K (500 steps, 4 gens, bs=1)
  exp_024  repro of exp_005 (GTPO-Conf + GRPO-S-Conf). Ranking
           between the two variants FLIPPED vs the original run —
           which proves exp_005's win was run-to-run luck, not the
           method.
  exp_025  pure-proof GTPO-EMA. peak 9.5 @ 253, final reward 3.0,
           same level as the successful confidence variants.
  exp_026  flipped GTPO-EMA — the key experiment. peak 9.5 @ 358,
           KL 0.095 (lowest of the successful confidence methods).
           Graph: exp_026/figures/gsm8k_exp026_flipped_vs_grpo.png

### Big-Math integer-2000 (500 steps, 8 gens, bs=4, max_completion=2048)
Dataset: `SynthLabsAI/Big-Math-RL-Verified`, filtered to integer
answers, shuffled with seed 3407, first 2000 kept. All seven runs
hit the reward ceiling 9.5.

Last-50 step averages (the most meaningful comparison):

  config                           peak@step  r@L50  fmt@L50 ans@L50 KL@L50
  exp_027 α=1.0 /0.0  baseline      9.5@205   +5.95   2.79   +0.84   0.20
  exp_028 α=0.9 /0.1  flipped  EMA  9.5@194   +6.28   2.89   +0.97   0.68
  exp_029 α=0.7 /0.3  flipped  EMA  9.5@222   +5.91   2.73   +0.93   0.11
  exp_030 α=0.5 /0.5  flipped  EMA  9.5@222   +6.08   2.83   +0.85   1.27
  exp_031 α=0.3 /0.7  flipped  EMA  9.5@225   +5.95   2.75   +0.86   0.56
  exp_032 α=0.95/0.05 flipped  EMA  9.5@219   +6.12   2.84   +0.88   0.20
  exp_033 α=0.9 /0.1  flipped NO-EMA 9.5@202  +6.15   2.81   +1.01   0.14

Key findings:
 - α2 = 0.05..0.1 is the sweet spot for reward/format.
 - α2 >= 0.3 mostly just adds KL without adding reward.
 - exp_033 (EMA ablation) matches exp_028 on reward with LOWER KL,
   which contradicts Prop 3.1 (which predicted EMA should reduce
   variance). Likely explanation: EMA creates more per-token signal
   diversity across the sequence, raw-C is flatter and easier to
   clip.
 - exp_032 (α2=0.05) gives the best reward@500 (+3.69) at
   baseline-level KL.
 - Best overall balance so far: **α2 ≈ 0.05, EMA on** (exp_032) or
   **α2 = 0.1, no EMA** (exp_033). Both are better than baseline
   on reward with similar or lower KL.

### Presentation
- `presentations/grpo_update_april_2026/grpo_update_april_2026.pptx`
  29 slides. Slides 1..11 copied from `GRPO_Modifications_Report.pptx`;
  new slides 12..29 cover exp_026 first (headline: proof-based
  flipped works), then exp_021..025, exp_027..028.
- Build is reproducible via `build_pptx.py` in the same folder.
- Needs one refresh after exp_029..033 (currently shows exp_028 as
  the freshest Big-Math result).

---

## 3. What I'd propose next if asked

In rough order of bang-for-buck:

1. **Refresh the presentation**. Re-run build_pptx.py after adding
   slides for the α sweep (exp_028..033) and the headline finding
   that exp_032 matches exp_028 on reward at a 3x lower KL.
2. **Seed variance study**. Rerun exp_027 and exp_032 twice more
   each with `random_state = 3408, 3409`. We've been making claims
   based on single seeds; Big-Math int-2000 at 500 steps is where
   the method gap is measurable but variance could still eat half
   of it.
3. **Mix flipped with EMA-only or non-flipped pure-proof** on
   Big-Math. We have the GSM8K version (exp_025) but no Big-Math
   run of the non-flipped proof formula. That would tell us
   whether the "flip" is actually important or a coincidence.
4. **Entropy-based variant** (Variant A from the earlier plan).
   Use real Shannon entropy H = -Σ π log π instead of top-k C.
   Semantically matches the Def 1.1 prose, cost is one extra
   softmax reduction per token.
5. **Longer runs / harder dataset**. Big-Math int-2000 still hits
   the 9.5 ceiling, so we only see the gap on the approach phase.
   Try Big-Math all-answers or MATH-500 at 1000 steps.

---

## 4. File locations / scripts to know

- `experiments/compare_conf_methods_gsm8k.py`    per-variant plots + 6-method overlay on GSM-8K
- `experiments/compare_027_to_031_alpha_sweep.py` 5-way α sweep overlay
- `experiments/compare_027_028_032_033.py`        4-way weak-bonus + EMA ablation overlay
- `experiments/compare_017_to_023.py`             Big-Math overlay (old)
- `experiments/figures_comparison/`               all shared plots live here
- `experiments/proof/GTPO-EMA-full.txt`           reference pure-proof formulas
- `experiments/exp_026_flipped_conf_gtpo_ema/`    the "proof works" experiment
- `experiments/exp_028_bigmath_int2k_flipped_gtpo_ema/`  canonical Big-Math launcher
  - src/reward_cache.py is the O+/O- mask cache pattern; copy it for
    any Big-Math shaping experiment
  - run_028.sh is the canonical Docker launch pattern
- `CLAUDE.md`                                      high-level repo rules

---

## 5. Open gotchas / things to remember

- `exp_005` and `exp_006` were each single-seed runs whose rankings
  between GTPO-Conf, GRPO-S-Conf, and their EMA variants do NOT
  reproduce (exp_024 repro flipped the result). Treat claims about
  those original runs with skepticism.
- The "top-k confidence" metric is numerically NOT entropy. This was
  the motivating observation behind exp_026.
- On GSM-8K the reward ceiling (9.5) is hit so early that almost
  every successful method looks the same at convergence. Use
  Big-Math int-2000 (or harder) when comparing methods.
- Prop 3.1 (EMA variance reduction) predicts EMA should lower KL.
  exp_033 shows the opposite empirically. Worth a follow-up test.
- The in-session auth tokens (HF_TOKEN, GitHub PAT) are in my
  persistent memory (not in this file). Ask me to "load memory
  about this project" and I will have them.

---

## 6. Running / in-flight state

At the time of writing this file:
- All seven Big-Math int-2000 runs (exp_027..033) have completed.
- GPU is idle.
- Latest commits on `main`: exp_032/033 overlay and logs.

If you (future Claude) find running training processes, check
`docker ps | grep unsloth/unsloth` and
`tail -c 2000 /mnt/data/aiim_research/experiments/<name>/train.log`.
