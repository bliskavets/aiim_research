# HANDOFF — GRPO/GTPO per-token reward-shaping project (2026-07-09)

Comprehensive session-continuation context. Read this first in a new session, then
`analysis/exp055-070_deep_analysis.md` and `experiments/exp_079_drgrpo/README.md`.

---

## 0. The method (our contribution)

**Polarity-Aware Confidence-based Token-shaping** (written in FULL, no acronym — user
preference) — a per-token credit-assignment layer that composes with ANY GRPO-family RL
algorithm. It reshapes the per-token advantage; it does not change the objective, so it is
written `X + Polarity-Aware Confidence-based Token-shaping` (e.g. `GRPO + …`, `DAPO + …`).

Definition (the current best config, "gtpo_ema_flipped FIXED + pos_discount"):
- **Confidence signal** `C_{i,t} = −(1/k) Σ_{j≤k} log π_(j)` — mean of the top-k log-probs
  (peakedness), **k=5**, then EMA-smoothed with **λ=0.7**.
- **Polarity split** by terminal correctness: O+ (reward>thr) vs O−. FLIPPED roles —
  O+ bonus ∝ 1/EMA(C) (reward branching/exploratory tokens on CORRECT rollouts),
  O− penalty ∝ EMA(C) (punish over-confident tokens on WRONG rollouts). α₁=0.9, α₂=0.1.
  Per-polarity z-norm over active tokens.
- **Position discount** `g(t)=τ/(τ+t)`, τ=1024, multiplies ONLY the α₂ bonus.
- **FIXED / group-visible**: the shaped 2-D (G,T) advantage is computed on the FULL
  generation group inside `_generate_and_score_completions` and injected into `compute_loss`
  — NOT recomputed in the B=1 microbatch (which degenerates group ops). This is the
  `GroupShapedBase` pattern in `src/novel_trainers.py` (present in every exp_07x/08x folder).

Sits on Qwen3-4B-Base as the primary study; also tested on Llama-3.2-3B (see §4, §5).

---

## 1. Environment / infra (see memory: aiim_machine_env)

- Box: single H200 143GB. **All runs sequential** (one GPU). No docker, no /mnt/data.
- Repo: `/root/aiim/aiim_research`. Python venv with unsloth/trl/vllm: **`/root/aiim/venv`**
  (`/root/aiim/venv/bin/python`). The system python has NO matplotlib/datasets — always use
  the venv python for plots and training.
- Stack: unsloth 2026.3.7, trl **0.23.1**, vllm 0.16.0, torch 2.9.1+cu128. Import `unsloth`
  before `trl` (CPU unit tests too).
- HF token: read at runtime from `/workspace/.cache/huggingface/token` (never hardcode).
- GitHub: push to `main` authorized for the session; end commit messages with the
  Co-Authored-By trailer. Token from Claude memory `github_token`.
- **Standard hyperparameters** (unless a harness says otherwise): Qwen3-4B-Base, ng=4, bs=1,
  ga=4, lr 5e-6 cosine, warmup 0.1, 300 steps (`SMOKE_MAX_STEPS` env overrides), seed 3407,
  max_seq 4096 (512 prompt + 3584 completion), β=0 (no KL), LoRA r=64/α=64/7 modules.
- TRL 0.23.1 quirk: `loss_type` DEFAULTS to `'dapo'` (token-level) — our "GRPO" already uses
  DAPO-style token normalization; `epsilon_high=None` (symmetric clip).
- Running jobs via `nohup bash <chain>.sh > <chain>.console.log 2>&1 &`; sequential chains
  wait on a prior chain's completion marker in its console log (see `chain_*.sh`).
- **GPU is currently FREE** (exp_083 finished). No jobs running at handoff.

---

## 2. How to run / plot / tabulate (reusable tooling)

- Each experiment: `experiments/exp_NNN_name/` with `train.py`, `run_setup.sh`,
  `plot_compare.py`, `src/`, `tests/`, `README.md`. Logs `train_{dataset}_{method}.log`
  (gitignored). Figures in `figures/` (committed).
- `train.py --dataset {gsm8k,math500,bigmath,omnimath} --method {…}`. Methods are wired in
  `build_trainer()` + `SHAPING_CONFIG` + argparse choices.
- **Paper tables**: `skills/baseline_peak_table.py` — MODEL-AGNOSTIC generator of the
  "peak reward / min-steps-to-baseline-peak" tables. Rerun setups for a new model, point
  `--dirs` at its logs. Args: `--baseline-suffix --ours-suffix --baseline-label
  --ours-label --datasets --metric --window` (rolling window 30, full-window only).
- Datasets (integer-answer exact-match): gsm8k (easy), math500 (medium), bigmath (Big-Math
  int-2k), omnimath (hard — everything struggles here).
- Metric watched: `reward_answer_boxed/mean` (Qwen harness) or `reward_answer_exact/mean`
  (exp_050/Llama harness). "L50" = mean of last 50 logged steps.

---

## 3. The Qwen3-4B-Base study — DONE (the paper's core, all positive)

### 3a. Establishing the method (exp_055–070, see analysis/exp055-070_deep_analysis.md)
- Best fixed config: **k=3–5, λ=0.7, pos_discount** (exp_063 combo, exp_066 k-sweep).
- **k-sweep sign-reversal insight** (exp_066/068/069/070): C_k flips MEANING between k=1
  (C≈0 on deterministic tokens → 1/C bonus farms filler → COLLAPSE) and k≥2 (C high on
  peaked, low on branching → rewards decision points). Boundedness lemma: C_k ≥ log k, so
  1/C_k bounded iff k≥2. Every scheme with k_min=1 collapsed (fixed k=1, nucleus_c all
  top_p, rank_c k≤5); every k_min≥3 stable. Adaptive-k (nucleus exp_068, rank exp_069) is a
  DEAD END; rank_floor (exp_070) ≈ fixed-k (no gain). These negatives are the ablation.
- exp_067: LLM token dists are BIMODAL (median top-1 mass 0.95–0.98 but p5≈0.05) — justifies
  small-k head statistic over full-vocab entropy (the GTPO paper 2508.04349 uses entropy).

### 3b. Principled roadmap (exp_071–075) — all complete
- **071 zero-variance gate**: skip shaping when std(R)=0 (else all rollouts → O− = pure
  noise; ~40% of omnimath groups). No regression on easy/medium; omnimath transient help
  then slid back. KEEP as a free component.
- **072 branch_entropy**: bounded signal h=H(renorm top-5 head)/log5 ∈[0,1]; O+∝h, O−∝1−h,
  NO reciprocal. Ties tuned-C on math500/bigmath, below on gsm8k. The clean/robust
  estimator (safety story) — MATCHES the tuned heuristic.
- **073 flipped_budget**: length-invariant bonus budget replaces pos_discount. BEATS best on
  gsm8k (+2.54), below on math500. Budget = valid principled anti-length-farming device.
- **074 surprisal_credit**: additive z(−log p(o_t)), cheapest variant. NEGATIVE (worse than
  GRPO on 3/4) — value is in the top-k HEAD statistic, not sampled-token surprisal.
- **075 final_combo** (gate+branch+budget): most BALANCED, no collapse, ties best on
  omnimath, but doesn't beat tuned pos_discount overall. Robustness/theory, not raw win.
- Conclusion: none beats the tuned pos_discount λ0.7 k5 OVERALL; the principled components
  give robustness + a theory. α₂ for the h-scale is still UN-tuned (open follow-up).

### 3c. Cross-algorithm generality (exp_076–079) — DONE, the headline positive result
Our shaping (pos_discount λ0.7 k5) layered on 4 base RL algorithms, all 4 datasets:
- **076 DAPO** (clip-higher 0.2/0.28 + token loss + overlong mask), **077 DAPO + shaping**,
  **078 GSPO** (seq-level IS, eps 3e-4/4e-4), **079 Dr.GRPO** (dr_grpo loss, no reward std).
- **Result**: additive on token-level algos (GRPO/DAPO/Dr.GRPO) — big L50 gains and reaches
  the base algorithm's peak in ~⅓–½ the steps. **Dr.GRPO + shaping is the top config**
  (peaks +2.67/+2.51/+2.36). **GSPO is the boundary case**: its tight sequence-level clip
  absorbs per-token structure → shaping speeds early convergence but doesn't raise final
  accuracy. Base algos alone all ≤ GRPO at 300 steps/ng=4.
- **THE paper tables live in `experiments/exp_079_drgrpo/README.md`** (§ FINAL RESULTS): 4
  per-baseline tables + a consolidated 8-row stacked table + L50-final table. Method written
  in FULL as "X + Polarity-Aware Confidence-based Token-shaping". Also
  `analysis/baseline_tables_qwen3-4b-base.md`.

---

## 4. Second base model — Llama (exp_080–083) — the current frontier (MIXED/negative, honest)

Goal: show the method generalizes across architectures (needed for a top-tier paper).
- **080 Llama-3.2-3B-Instruct, Qwen harness (`<think>`+`\boxed`, 300 steps)**: NEGATIVE
  transfer. gsm8k −2.24 (slow degradation), math500 +0.16, bigmath −0.18, omnimath −0.04.
- **081 Llama-3.2-3B BASE, Qwen harness**: STOPPED — base model can't bootstrap at all (GRPO
  flat ~0 on all 4; math500 degenerated to len 38). Non-SFT 3B too weak for cold-start here.
- **082 Llama-3.2-3B-Instruct, exp_050 harness** (custom working_out/SOLUTION tags taught in
  system prompt, graded format_exact/approximate + answer_exact/numeric, 500 steps, max_comp
  2048): **GRPO learns** (mid-takeoff by 500, matches exp_050) but **our FIXED shaping
  ANTI-LEARNS on all 4** (format never forms, fmt_approx→−2.5, len→84).
- **083 Llama-3.2-3B-Instruct, exp_050 harness, bigmath, 3-way**: GRPO +0.56 vs
  gtpo_ema_flipped **ORIGINAL +0.98 (BEST)** vs **FIXED −1.07 (COLLAPSES)**. On Llama the
  group-visible FIX HURTS — a REVERSAL of the Qwen story.

### THE cross-model mechanism (exp_082 + exp_083, the key open finding)
Our polarity split uses `advantage > 0` — which is GROUP-RELATIVE. On Qwen (strong base,
format saturates instantly) advantage-polarity ≈ true correctness, so O+ ≈ "correct" and the
shaping semantics hold — the FIX is strictly better. On Llama cold-start NOTHING is correct,
but the graded format reward still creates advantage variance, so O+ = "less-bad junk"; the
1/EMA(C) bonus then reinforces junk exploration and O− punishes the model's fluent (peaked)
text → format never forms → collapse. The ORIGINAL trainer's B=1 microbatch degeneracy
accidentally sidesteps the group-relative polarity split, acting as a mild beneficial
per-sequence bonus (hence it wins on Llama). Note: on THIS stack the original's `_compute_loss`
is NOT bypassed (it diverges from GRPO; step-1 identical) — contra the older
`shaping_bypassed_by_unsloth` memo; verify per run.

---

## 5. NEXT STEPS (prioritized) — what to run in the new session

1. **exp_084 — correctness-grounded polarity (HIGH PRIORITY).** Replace `adv > 0` with a
   polarity keyed to the rollout's ACTUAL answer reward (O+ iff answer_reward>0; groups with
   no correct rollout → plain GRPO). This is the direct fix for the exp_082/083 Llama
   failure and should make the FIXED shaping help on Llama while preserving Qwen gains. Run
   on Llama-3.2-3B-Instruct exp_050 harness (bigmath first, then 4 datasets) AND re-verify on
   Qwen (should be ≈unchanged since there polarity already ≈ correctness). This likely
   becomes the paper's "robust polarity" method upgrade.
2. **α₂ / h-scale sweep for branch_entropy (exp_072)** — its bounded signal has a different
   scale; a tuned α₂ may let the principled estimator beat the heuristic C.
3. **budget + light pos_discount combo** (exp_073 showed budget beats posdisc on gsm8k but
   not math500 — combine them).
4. **Multi-seed + held-out eval accuracy** (not just training reward) on the final method,
   for the paper. Add the GTPO-paper entropy-weighted baseline (arXiv:2508.04349) in our
   codebase for a direct comparison.
5. Optional: Llama-3.1-8B (stronger base than 3B) if a cleaner cross-model positive is wanted.

---

## 6. Paper framing / final goals

Target: top-tier conference; differentiate from GTPO/GRPO-S (arXiv:2508.04349, full-vocab
entropy weighting). Our contributions:
1. **Estimator**: head-truncated confidence (k=3–5) vs full-vocab entropy, justified by
   token-distribution bimodality (exp_067) + the k-sweep sign-reversal + boundedness lemma.
2. **Polarity-flipped, correctness-conditioned per-token credit** (direction depends on
   terminal correctness) + causal EMA — vs polarity-independent entropy weighting.
3. **Algorithm-agnostic**: same shaping boosts GRPO/DAPO/Dr.GRPO (exp_076–079), with a
   convergence-speed story (~⅓–½ steps to baseline peak). GSPO scope boundary is honest.
4. **Stability theory**: boundedness lemma + zero-variance gating + length-invariant budget,
   with 8+ controlled collapses (exp_066/068/069) as the evidence base.
5. **Correctness-grounded polarity** (exp_084, in progress) as the cross-model robustness
   fix — the Llama arc (exp_080–083) is the motivating negative-result story.
6. **Reproducibility**: the B=1 / unsloth-bypass degeneracy (memory
   `shaping_bypassed_by_unsloth`) — practitioners will hit it.

---

## 7. Key files / pointers

- `analysis/exp055-070_deep_analysis.md` — deep analysis + the mechanism/lemma + roadmap.
- `analysis/baseline_tables_qwen3-4b-base.md` — the 4 paper tables (Qwen).
- `experiments/exp_079_drgrpo/README.md` — **the master FINAL RESULTS doc** (all tables,
  method named in full, cross-method takeaways).
- `skills/baseline_peak_table.py` — model-agnostic paper-table generator.
- `src/novel_trainers.py` (GroupShapedBase + PosDiscountTrainer etc.),
  `src/novel_shaping.py` (flipped_advantages, position_discount),
  `src/ema_flipped_utils.py` (confidence, EMA, compute_gtpo_ema_flipped_advantages),
  `src/roadmap_shaping.py` + `src/roadmap_trainers.py` (071–075),
  `src/gtpo_ema_flipped_trainer.py` (ORIGINAL, exp_083),
  `src/gtpo_ema_flipped_fixed_trainer.py` (FIXED) — copied per experiment folder.
- Memory: `paper_roadmap_insights.md` (results + roadmap), `adaptive_posdiscount_proposals.md`
  (exp_065–070), `shaping_bypassed_by_unsloth.md`, `aiim_machine_env.md`, `user-barys.md`.

## 8. Working preferences (memory: user-barys)
Operator is an ML researcher, terse Russian, delegates execution end-to-end. Expected
cadence per experiment: announce setup + hypothesis → run in background → on "как прогресс?"
report numbers → build intermediate `plot_compare.py` and PUSH to GitHub on request →
finalize README + memory when done. Always overlay new methods vs GRPO and the current best.
Report honestly (negatives are results). Push figures directly to main.
