# AAAI-2027 SAGE strengthening: results package

Read this first. This folder is the handoff for the paper-integration step: it contains
every experimental result produced in the July 2026 strengthening cycle, in compact form,
plus the context needed to write them into `papers/icml2026/main.tex` (SAGE paper source).
Produced per the plan in `rebuttal/experiment_plan_aaai2027.md` and the runbook in
`rebuttal/CLAUDE_AAAI2027.md`.

## Setup (applies to every run)

- Model: Qwen/Qwen3-8B-FP8 on vLLM 0.11.0, single H200, `--max-model-len 32768`,
  non-thinking mode via the Qwen3 chat template (see "Harness fixes" below - the paper's
  original ` /nothink` suffix did NOT engage non-thinking; this was fixed and the fixed
  engine used for all runs here).
- SAGE config: m_min=1, 2 optimization epochs x 7 generations (21 solution generations).
- MATH-500 grading: boxed-answer extraction + o3 semantic equivalence (via an
  OpenAI-compatible gateway). MMLU-Pro STEM: exact letter match, judge-free.
  IFEval: official instruction_following_eval verifiers, judge-free.
- Seeds: 42 everywhere; SAGE additionally seeds 7 and 123 on MATH and MMLU-Pro.
  The greedy baseline is seed-invariant; its CI comes from bootstrap over problems.

## Headline results

### MATH-500 (exact-match, N=500)
| Method | Accuracy | Note |
|-|-|-|
| Baseline (greedy, non-thinking) | 83.8 | reproduces paper's 84.4 |
| Self-Refine (budget 21, early-stop) | 86.6 | E2, same-model critique baseline |
| Reflexion (budget 21, early-stop) | 86.4 | E2 |
| SAGE seed42 / 7 / 123 | 88.8 / 87.6 / 88.4 | mean 88.3 +/- 0.6 |

Paired SAGE-vs-baseline (seed42): +5.0pt, 95% CI [+2.2, +7.8], McNemar p=0.0008.
Paper previously claimed SAGE 92.0; the honest multi-seed number with the fixed
non-thinking baseline is 88.3 +/- 0.6. Report the new number.

### MMLU-Pro STEM (letter-match, N=500)
| Method | seed42 | seed7 | seed123 |
|-|-|-|-|
| Baseline | 71.0 | 71.6 | 71.0 |
| SAGE | 75.8 | 78.0 | 78.2 |

SAGE mean 77.3 +/- 1.3 vs baseline ~71.2. Paired deltas +4.8/+6.4/+7.2pt,
McNemar p = 0.006 / 0.00017 / 0.000029. New benchmark for the paper (E5):
verifiable, broad, meaningful headroom.

### IFEval (N=541, prompt-level / instruction-level)
| Method | Prompt-acc | Instr-acc |
|-|-|-|
| Baseline | 73.8 | 79.7 |
| BoN (judge-select, N=7) | 77.4 | 83.1 |
| SAGE | 76.3 | 82.1 |
| Self-Refine | 77.1 | 82.6 |
| Reflexion | (running; see summary_all_runs.json) | |

Honest note for the paper: on IFEval SAGE clearly beats the baseline (+2.6pt prompt)
which counters the verbosity concern (deterministic verifiers), but it does NOT beat
BoN or Self-Refine there; SAGE's edge over selection/critique methods shows on
reasoning (MATH, MMLU-Pro), not instruction-following. Do not overclaim.

### Latency (A2 smoke, N=20, indicative only - rerun at full N if needed)
See `summary_all_runs.json` key `a2_latency_smoke_n20`: baseline 28.5s/problem,
BoN 35-41s, SAGE 328s (sequential refinement). Use only as an order-of-magnitude
statement unless re-run.

## Statistical significance

`rebuttal/analyze_results.py` computes everything: per-run bootstrap 95% CIs,
multi-seed mean +/- sd, paired bootstrap CI on deltas, McNemar exact-ish p.
Run `python analyze_results.py --logs rebuttal/logs` if raw logs are present,
or adapt to `per_problem/` files here (same schema: index + is_correct /
prompt_followed per problem, so all stats are reproducible from this folder alone).

## Files in this folder

- `summary_all_runs.json` - one dict per run: n, accuracy (or prompt/instruction
  accuracy). Runs still in flight at collection time are marked MISSING_OR_RUNNING.
- `per_problem/<run>.jsonl` - compact per-problem records: `index`, `is_correct`
  (or `prompt_followed` + instruction counts for IFEval), plus `pred`/`gt`/`category`
  for MMLU and `num_generations`/`stopped_early` for E2 runs. Full raw outputs
  (all candidate texts) are NOT in git (2.9 GB per SAGE run); they live on the GPU
  box under `rebuttal/logs/` if deeper analysis is needed.
- `INDEX.md` - this file.

## Harness fixes made this cycle (already committed, relevant to paper claims)

1. **Non-thinking bug (critical)**: `AugEngine` appended ` /nothink` to raw
   /v1/completions prompts, which does not engage Qwen3 non-thinking; the model
   emitted full CoT truncated at max_tokens, so ~35% of MATH answers scored wrong.
   Baseline measured 51.4 before the fix, 83.8 after. All numbers above use the fix.
   Any number in the old paper produced by the old harness on a "non-thinking"
   baseline should be treated as suspect and replaced by these.
2. Math judge now works through OpenAI-compatible gateways (chat.completions JSON).
3. A2 BoN generated candidates at temperature 0 with n>1 (rejected by vLLM,
   degenerate anyway); now 0.7.
4. MMLU-Pro SAGE runner batched (was serial).
5. E2 refinement prompts enforce answer-only output + NO_ERRORS early stop
   (naive free-form refinement drifted into boxed commentary).

## Reviewer concern -> evidence map (EMNLP reviews, see rebuttal/Reviews.docx)

- "self-judge rewards familiarity, not correctness" (k8B9, qCe4): the MATH/MMLU-Pro
  gains are exact-match verifiable, cannot come from self-preference. Cite the
  3-seed CIs + McNemar p-values above.
- "verbosity drives AlpacaEval" (qCe4): IFEval gains are deterministic-verifier based.
- "no Self-Refine/Reflexion baselines" (k8B9): E2 rows above, budget-matched (21 gens),
  SAGE beats both on MATH by ~2pt while they beat it slightly on IFEval.
- "benchmarks too easy / need harder" (pDvu-style): MMLU-Pro STEM added (71->77).
- "baseline discrepancy 84.4 vs 87.4": 83.8 reproduced here with the correctly-engaged
  non-thinking template; difference vs the 87.4 tech report is prompt configuration.
- "wall-clock vs BoN" (Y1iM, RD4w): A2 latency table (needs full-N rerun for the paper).

## Still running / pending at collection time

Check `summary_all_runs.json` for MISSING_OR_RUNNING markers. Pending queue:
E2 reflexion IFEval (last E2 job), then TPO (official repo, FsfairX-LLaMA3-RM-v0.1,
D2-N5, non-thinking-patched textgrad) on MMLU-Pro STEM + IFEval - results will be
appended here as `tpo_*` entries when done.

## Style rules for writing these into the paper (author preference)

No em-dashes; no comma thousands (8233 not 8,233); no First/Second/Third scaffolding;
delete "significantly/substantially" unless the test above backs it; tables over prose.
Commit as Barys Liskavets <barys.liskavets@acclaim.ai>; no AI-authorship mentions.
