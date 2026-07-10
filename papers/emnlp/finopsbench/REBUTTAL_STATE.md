# FinOpsBench EMNLP Rebuttal — Full Session State (handoff)

Last updated: session ending 2026-07-10. Repo HEAD at handoff: `1fe252f`.
This file lets a fresh session resume without re-deriving anything. Everything
below is committed under `papers/emnlp/finopsbench/` in `bliskavets/aiim_research`.

---
## 0. Task & context
- Paper: **FinOpsBench** — benchmark for agentic LLMs on financial tool-use. EMNLP submission **5243**.
- Two parts: **v1** (5,979 synthetic SQL-tool tasks, 9-stage LLM pipeline, panel of 3 LLM judges) and **v2** (1,108 FinQA-derived agentic environments, smolagents, custom Python tools + distractors).
- Reviewers & scores: **PVoW** Overall 3.0 / Conf 5 (most constructive); **R2** 2.5 / Conf 4 (novelty/diagnostics); **R3** 2.5 / Conf 4 (contamination, missing models, "outdated smolagents", v1 no hard GT).
- Goal: run experiments + write rebuttal answers. Deliverables live in `answers.md` (per-reviewer, per-question, with drafts + real numbers) and `experiments/`.

## 1. Repositories
- **`bliskavets/aiim_research`** → `papers/emnlp/finopsbench/`: `submission.tex`, `reviews.txt`, **`answers.md`** (the rebuttal), **`experiments/`** (all E1–E11), this file. Commit as `Barys Liskavets <barys.liskavets@acclaim.ai>`, no AI mentions in messages/content. Push blocked in auto-mode sometimes → `git push origin main` may need manual; use `git pull --rebase origin main` before commit (user pushes to same branch).
- **`bliskavets/FinOpsBench`** (release/mirror source for anonymous.4open.science): structured v1/v2 code + data, built earlier this project. **Prompt-leak fixes (see §E11) are staged in the working copy `/tmp/FinOpsBench` but NOT yet pushed to this release repo — must be pushed.**
- **`bliskavets/TTA`** — original messy research code (source for the release).

## 2. Environment / working state (ephemeral — /tmp, will be gone in new session)
- `/tmp/FinOpsBench` — working clone of the release. v1 pool: `v1/data/finopsbench_v1_pool.jsonl.gz` (8,233 items). v2 envs: `v2/finqa_agents/agent_*/` (1,174 usable). **v2 system prompts were edited to remove answer leak; originals in `agent_system_prompt.txt.orig`.**
- `/tmp/finqa_train.json` — FinQA train split (6,251 items), downloaded from czyssrs/FinQA. Mapping **`agent_N` ↔ `train.json[N]`** (verified, 93% answer-match).
- `/tmp/e4venv` — Python venv with `smolagents`, `mlflow`, `openai` (used for agentic v2 runs via subprocess).
- MLflow server on `localhost:7777` (started via `mlflow server`); SA runner requires it.
- Fixed eval subset: `experiments/e8_access_ladder/subset_200.json` (200 v2 agent_ids, seed 13).
- **OpenRouter budget remaining: ~$156.72.** Key `OPENROUTER_API_KEY` was provided in chat (`sk-or-v1-ebd54...`) — **user must revoke after**; also revoke leaked OpenAI key `sk-ccjg...` (in TTA history) and the GitHub token `ghp_iWId...`.

## 3. Reusable infra (patterns for new runs)
- **v2 agentic eval:** `experiments/e4_new_models/run_e4.py` + runner `SA_openrouter.py` (smolagents copy + OpenRouter provider routing via `OPENROUTER_EXTRA_BODY`). Flags: `--model <openrouter id> --runner SA_openrouter.py --python /tmp/e4venv/bin/python --subset_file <ids.json> [--limit N] --concurrency 4-8 --budget_usd N --out_dir <dir>`. Resumable (skips existing `<agent>.txt`). Scores with `v2/compare_outputs.py`.
- **v2 no-tool eval (ladder rungs a/b/d):** `experiments/e8_access_ladder/run_context.py --mode {question_only,finqa_canonical,full_context} --model <id>`. **Percent-robust scoring** needed (see `rescore.py`/`assemble.py`): benchmark's `compare_answers` treats trailing `%` as /100, so a CoT model printing "52.32" for gold "52.32%" is off by 100× — the robust matcher accepts the percent-scaling variant.
- **Known bug fixed:** early `run_context.py` used `zip(todo, asyncio.as_completed(...))` → mispaired predictions with wrong gold. Fixed: `ask()` returns its own item. (E1's `run_closed_book.py` was always correct.)
- Cost tracking: OpenRouter `usage.cost` per request; for smolagents runs, delta of `/api/v1/credits`.

## 4. Experiments — status, results, artifacts
All under `experiments/`. Each has a README.

### E0 — Repo + prompts (PVoW-3/10) ✅
Structured FinOpsBench release built + `PROMPTS.md` index. Prompts were always in code; just surfaced. Scrubbed hardcoded OpenAI key. (Release repo push + anonymous mirror re-sync still TODO by user.)

### E1 — Closed-book contamination, v2 (R3-6) ✅ `e1_closed_book/`
Prompt minus tools, answer from prompt+memory. **GPT-5-mini 14.7%, GPT-4.1 13.3%, Qwen3-30B 13.8%** (vs agentic 67.5/60.6/53.0). Flat ~14% floor → memorization doesn't give the answer.

### E2 — LLM-judge vs deterministic, v1 (PVoW-2/4, R3-1) ✅ `e2_judge_agreement/`
Only **4.4%** (363/8233) of v1 gold answers are single-number (deterministic scoring undefined for the rest → why v1 uses an LLM judge). On the 92 contested (judge≠numeric) cases, expert sides with **judge 82.6%** vs numeric 17.4%, **κ=0.64**. Human labels done by user in a Streamlit viewer.

### E3 — Human eval (PVoW-1/2, R2-4, R3-1) ✅ `e3_human_eval/`
Single expert annotator (report human↔scorer agreement, not inter-annotator κ). Viewers: annotation_viewer.py, viewer.py (v1_judge + v2_validity samples), estimate.py.
- **v1 judge accuracy = 85.1%** stratified (contested 82.6% n=92 + non-contested 85.9% n=78; pooled κ=0.67).
- **v2 dataset cleanliness = 85%** (execution-based: 200 examples, 192 ran, **170 reproduced gold**). 8 flagged (agent_741 etc.), dominated by reference-plan/gold mismatches.
- Total human-verified: **372** (172 v1 + 200 v2).

### E4 — Extra model families, v2 (R3-4, PVoW-8) ✅ `e4_new_models/`
Original leaky-prompt runs: **Claude-Sonnet-4.5 70.5%** (n=139, tops leaderboard), **DeepSeek-V3-0324 57.3%** (n=1,134). Supports "not just tiny open models" + cross-family (generator=GPT-4.1-mini, yet Anthropic tops → no generator-family bias). NOTE: these numbers are on LEAKY prompts (see E11).

### E5 — Failure taxonomy (PVoW-6, R2-3) ✅ `e5_failure_taxonomy/`
779 failing traces, 8 categories, classifier gpt-4.1-mini. v1 failures **semantic not syntactic** (SQL errors≈0; malformed args 36–42%, incomplete retrieval 22–37%; calc only 5–10%). v2 shifts to wrong-tool-selection (20–23%) + DeepSeek round-exhaustion 25%. Process metrics + worked examples + `verify_sample.jsonl` (60 for human check).

### E6 — Diversity stats (PVoW-5, R3-2) ✅ `e6_diversity/`
v1: **742 user roles**, 0 dup queries, SQL: 70% JOIN/35% aggregate/22% subquery; v2: median 5 plan tool-calls, 9 tools/env; op-mix aggregation 51%/diff 41%/ratio 32%. Financial-concept coverage + template diversity added.

### E7 — Construction cost (PVoW-7) ✅ `e7_costs/`
Measured via OpenRouter replay: **v1 $0.037/example (68s)**, **v2 $0.237/example (o3, 112s)**. Extrapolated totals: **v1 ~$450, v2 ~$340, total ~$790**. v1 cost ~81% the 3-judge panel; v2 ~65% the two o3 codegen stages. **API-only, no GPU** (H100 only for open-model eval).

### E8 — Information-access ladder (R2 novelty, R3-6) ✅ `e8_access_ladder/` ⭐ centerpiece
Same 200 v2 items, 4 access modes: **question-only (a)**, **agentic tools (c)**, **FinQA-native = gold facts / qa.model_input (d)**, **full-context whole doc (b)**. Percent-robust scoring. Novel metrics: **tool-use necessity (c−a)**, **agentic gap (d−c)**.
**CRITICAL — leaky vs clean, see E11.** Final 9-model table (`results/clean_vs_leaky.json`, `results/clean_table.md`):

| Model | q-only | agentic LEAKY | agentic CLEAN | FinQA-native | full-ctx | gap(clean) | n(clean) |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 2.5 | 66.5 | **69.9** | 64.5 | 66.5 | −5.4 | 103 |
| Claude-Sonnet-4.5 | 1.5 | 69.2 | **68.6** | 68.5 | 69.5 | −0.1 | 156 |
| GPT-4.1 | 2.0 | 63.5 | **66.0** | 65.5 | 65.0 | −0.5 | 200 |
| Claude-Haiku-4.5 | 0.5 | 67.5 | **65.5** | 67.0 | 69.5 | +1.5 | 200 |
| Qwen3-235B-A22B | 2.5 | 65.0 | **65.0** | 65.0 | 68.0 | 0.0 | 200 |
| GPT-4.1-mini | 1.5 | 61.5 | **60.0** | 60.5 | 64.5 | +0.5 | 200 |
| DeepSeek-V4-Flash | 2.5 | 71.0 | **54.3** | 68.0 | 71.0 | +13.7 | 162 |
| DeepSeek-V3.2 | 4.0 | 48.2 | **38.6** | 69.0 | 69.5 | +30.4 | 158 |
| Llama-3.3-70B | 3.0 | 29.9 | **19.8** | 57.0 | 59.0 | +37.2 | 106 |

Story: tool-use necessity huge for all (q-only≈2%). Agentic gap **splits models 6 (faithful, |gap|≤~2) vs 3 (read-well-act-poorly: DeepSeek-V3.2/V4-Flash, Llama-3.3-70B)** — NOT size-bound; visible across DeepSeek generations. Static benchmarks miss this.
Scripts: `run_context.py`, `run_e4.py` (subset), `rescore.py`, `assemble.py` (rebuilds table; MODELS/NAMES lists).

### E9 — Difficulty control (R2, "tunable difficulty") ✅ `e9_difficulty_control/`
Clean signal: **tool-chain depth** → monotonic accuracy drop (pooled 62%→46% at 8+ hops; DeepSeek 62→43). Distractor-count observational analysis = confounded (no trend). Distractor **ablation** (core-only) reported as INVALID (system prompt still advertises removed tools + plan-external tools are discovery helpers) — honest negative, `analyze_axes.py`, `SA_coreonly.py`.

### E10 — Cross-benchmark competitor (R2) ✅ `e10_cross_benchmark/`
Same model (gpt-4.1-mini): **TAT-QA (external static finance QA) 89%** reading vs FinOpsBench-v2 **1.5% closed-book / ~62% agentic**. Static benchmarks test reading; FinOpsBench tests tool use. `run_tatqa.py`.

### E11 — Prompt-leak audit & fix (found during R3 validation) ✅ `e11_prompt_leak_audit/` ⭐ IMPORTANT
Stage-8 v2 system-prompt generator embedded the **gold answer as the output-format example** (`e.g. "39.1%"`) → leak in **~26%** (345/1174) items. Fixed **305 prompts** (`redact_prompts.py --apply`, neutral placeholder, `.orig` backups); exact-answer-in-prompt **29.4%→6.3%**. Re-ran all 9 models agentic on cleaned prompts (E8 CLEAN column). **Leak effect model-dependent:** strong tool-users unchanged; DeepSeek/Llama drop sharply (gaps grow to +14/+30/+37). Clean runs cost ~10× more (agent can't shortcut). `results/leak_report.json`, `deepseek_v4_clean_vs_leaky.json`.

## 5. Arguments (no experiment) drafted in answers.md
A1 novelty (planning under partial observability; comparison table `tab:benchmark_comparison`), A2 smolagents (current HF lib, v1 doesn't use it, two-protocol consistency), A3 proprietary-model bias (cross-vendor panel + E4 cross-family).

## 6. answers.md status (per reviewer)
- **PVoW:** 1✅ 2✅ 3✅ 4✅ 5🟡(text done, could add table) 6✅ 7✅ 8✅/🟡 9✅ 10✅
- **R2:** 1✅ 2✅[A1+E8] 3✅ 4✅
- **R3:** 1✅ 2–3✅ 4🟡(text done; add DeepSeek-V3.2/Haiku already there) 5✅ 6✅[E1+E8]
- Remaining 🟡 are essentially complete-with-numbers; mainly polish. Pre-post checklist at bottom of answers.md.

## 7. Open TODOs (for user / next session)
1. **Push prompt-leak fix** (`/tmp/FinOpsBench/v2/finqa_agents/*/agent_system_prompt.txt`, redacted) to the FinOpsBench release repo — it's a genuine benchmark bug fix. Backups are `.orig`.
2. **Push FinOpsBench release** + re-sync anonymous.4open.science mirror (PVoW Software:1 / Reproducibility:2).
3. E4 model-coverage numbers (Claude-Sonnet-4.5 70.5%, DeepSeek-V3 57.3%) are on LEAKY prompts → optionally re-run on clean (cheap models ~$7, +Claude ~$15).
4. E5 failure traces were on leaky prompts too (minor; taxonomy shape unaffected).
5. Revoke the 3 leaked credentials.
6. Final read-through of answers.md before posting; convert drafts to final prose, fill any remaining `[...]`.

## 8. Key numbers to remember (quick reference)
- v1 judge accuracy 85.1% (κ0.67); v2 cleanliness 85% (170/200); 372 human-verified.
- Closed-book v2 ~14% vs agentic ~54–70%.
- Construction ~$790 total, API-only.
- Access ladder: agentic gap splits 6 faithful vs 3 read-well-act-poorly.
- Prompt leak: 305 prompts fixed, ~26% affected, corrected agentic column in E8.
- Rebuttal experiments spent so far: OpenRouter budget ~$156.72 remaining.
