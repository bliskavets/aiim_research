# SAGE — AAAI 2027 resubmission: 6-day strengthening plan

Goal: close the three EMNLP reviews (Y1iM 2, qCe4 2.5, k8B9 2) with a 6-day window.
Bottleneck is wall-clock + new-code time, NOT GPU-hours. So the plan leans on the existing
harness in `rebuttal/` (SEED/EPOCHS/N_GENS already parameterised; a1-c3 already coded) and
keeps new code to three items only.

## Assumptions
- ~4x H100 available; Qwen3-8B-FP8 served on vLLM (32B on 2 GPUs only if pursued).
- Two parallel tracks every day: **Writing** (no GPU, author) and **Compute** (GPU jobs).
- All main runs re-use `rebuttal/experiments/*` and `rebuttal/core/*`; SEED loop = free multi-seed.

## Central thesis to defend
k8B9/qCe4 attack: "self-judge rewards familiarity, not correctness." The killer counter is that
gains on **verifiable** tasks (MATH exact-match, IFEval script, MMLU-Pro MC) cannot be explained
by self-preference. Every experiment below feeds that one argument.

---

## Critical path (fits 6 days)

| ID | Item | Reviewer | Code | GPU-h | Day |
|-|-|-|-|-|-|
| W | Writing fixes (see below) | all | - | 0 | 1-6 (parallel) |
| E1 | **3-seed re-run of main tables** (MATH-500, AlpacaEval-LC, XSTest; both models) + bootstrap CI + significance | k8B9, qCe4 | reuse (SEED loop) | ~40 | 1-2 |
| E2 | **Self-Refine + Reflexion baselines** (same-model, matched budget) on MATH-500 + IFEval | k8B9 | NEW (~1d) | ~10 | 2-3 |
| E3 | **Reward-hacking / self-preference probes**: (a) adversarial judge-context / prompt-injection; (b) MATH judge-selection vs ground-truth cross-check (reuse logged candidates, no new gen); (c) self-vs-other-generation preference | k8B9, qCe4 | NEW-light (~0.5d) + analysis | ~3 | 3 |
| E4 | **IFEval** (verifiable, judge-free -> kills verbosity) | Y1iM, k8B9 | reuse a3 | ~5 | 1-2 (bg) |
| E5 | **MMLU-Pro STEM** (verifiable breadth) | qCe4 | reuse c3 | ~12 | 1-3 (bg) |
| E6 | **Quality-cost Pareto curves**, counting judge+reflection tokens for SAGE/BoN/TPO/SPO/Self-Refine | qCe4, Y1iM | reuse a2 + extend | ~4 | 3-4 |
| E7 | **Grouped-gradient isolation** at fixed budget: {grouped, best-vs-worst (m_min=1), best-only, no-gradient rerank} | qCe4 | reuse b2 + 2 variants | ~10 | 4 |
| E8 | **Calibration**: ECE, Brier, reliability curves from logged margins (replaces NDCG-as-calibration) | qCe4, k8B9 | NEW-light (analysis) | ~1 | 4-5 |
| E9 | **Updated / math-capable verifier** (Skywork-Reward-Qwen2.5-7B or ArmoRM) + MATH exact-match oracle | qCe4, k8B9, pDvu | reuse c2 | ~8 | 3-4 |
| E10 | **XSTest per-category breakdown** (safe-compliance vs unsafe-refusal) | qCe4, k8B9 | reuse core/xstest | ~1 | 5 |

GPU-h total ~= 90-100, i.e. ~1 day of wall-clock on 4 H100 spread across the window. The binding
constraint is E2/E3/E8 code + integration, not compute.

## Day-by-day
- **Day 1**: serve model; kick E1 (seed loop) + E4 (IFEval) + E5 (MMLU-Pro) as background jobs.
  Writing: citations + related-work paragraph on self-judge failure literature.
- **Day 2**: write E2 (Self-Refine/Reflexion on solver scaffolding); E1 finishing. Writing: abstract
  -> AlpacaEval-LC headline + length analysis.
- **Day 3**: run E2; write+run E3 probes; kick E9 (updated RM). Writing: definitions (SAGE+RM,
  m_min, epoch indexing), typo fixes.
- **Day 4**: E6 cost curves; E7 ablation variants; E8 calibration analysis. Resolve the
  SAGE > SAGE+RM anomaly with seeds + E9 verifier.
- **Day 5**: E10 breakdown; bootstrap CI + significance across all tables; build figures.
- **Day 6**: integrate results into paper, finalise tables/figures, tighten claims, artifact package.

---

## Writing track (no GPU, high score-impact, do in parallel)
1. **Multi-seed + significance everywhere**; delete "significantly/substantially" where untested.
2. **Headline = AlpacaEval LC (length-controlled)**, not raw 74.9; add avg-tokens-per-response vs
   baselines; rewrite abstract.
3. **Explain SAGE > SAGE+RM on MATH (92.0 vs 89.8)**: seeds show if noise; else attribute to the
   outdated FsfairX RM on math (E9) and reframe.
4. **Add missing citations + positioning**: Self-Refine (Madaan 2023), Reflexion (Shinn 2023),
   Tian 2023 (calibration), Jiang 2024, Wataoka 2024 (self-preference), Pan 2024 (reward hacking).
5. **Definitions/typos**: define SAGE+RM precisely; Table 1 caption; m_min (min group size vs cap);
   epoch indexing; fix `</asnwer>`, "Optimizational epoch".
6. **Release usable artifact** (candidates, judge prompts, eval harness) — all three gave Datasets=1.

---

## Deferred (out of 6-day scope; camera-ready or note as limitation)
- AlpacaEval human eval (4.1) — do a tiny 50-pair sanity check only if a day frees up, else defer.
- Qwen3-32B scaling (c1) — 20 GPU-h + 3-4 day wall-clock; run as background on 2 spare GPUs if
  available, otherwise defer with the existing 1.7B/8B trend as the scaling argument.
- Aspect sensitivity (b3) and small-model 1.7B (b4) — reuse ICML-cycle results, fold in as-is.
- Signal-form ablation (contrastive margin vs hard label) — stretch.

## Reuse map (already coded in rebuttal/experiments)
a1 baseline · a2 latency · a3 ifeval · b2 m_min · b3 aspect · b4 small · c1 qwen32b · c2 updated_rm ·
c3 mmlu_pro · sage/solver.py · core/{math500,ifeval,xstest,alpaca}_eval.py, vllm_client.py.
New code only: Self-Refine/Reflexion loop (E2), reward-hacking probes (E3), calibration metrics (E8).
