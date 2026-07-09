# E5 — Failure-mode taxonomy & process metrics

**Claims addressed:** PVoW-6 (qualitative failure examples beyond accuracy) and
R2-3 (fine-grained diagnostics showing the benchmark yields signal beyond a score).

## Data (clean, no new model runs on v1)
- **v1 (primary):** the archived *evaluated* runs `eval_sample_evaluated_{gpt_5,o4_mini,gpt_4.1,gpt-4.1_mini}.jsonl`
  — each carries the model's own trace (`evaluation.agent_dialog`), the LLM-judge
  verdict, and the judge's reasoning. Full failure counts: GPT-5 342, o4-mini 362,
  GPT-4.1 429, GPT-4.1-mini 413.
  (The raw per-model files for Qwen/Llama were **excluded**: their `agent_dialog`
  is not reliably the model's own run, so they are unsafe for failure analysis.)
- **v2 (cross-version contrast):** the E4 smolagents runs — Claude-Sonnet-4.5 (41
  fails) and DeepSeek-V3 (484 fails), scored by execution against gold.

## Method
1. `extract_failures.py` — pull failing traces + deterministic process metrics
   (tool-call count, SQL/tool errors, round-exhaustion); cap 150/model (seed 13).
2. `classify.py` — one of 8 failure categories per trace via an LLM
   (`openai/gpt-4.1-mini`), given the question, gold, model answer, tool-call
   trace, and (for v1) the judge's reasoning. 779 classified, $0.26.
3. `analyze.py` — category distribution and mean process metrics per model.
4. `verify_sample.jsonl` — 60 random cases with full trace + assigned label
   (`human_agrees` field) for manual spot-check.

## Headline findings (`summary.json`)
- **v1 failures are semantic, not syntactic:** SQL errors ≈ 0, yet
  *malformed_arguments* (36–42%) and *incomplete_retrieval* (22–37%) dominate —
  models write valid SQL with the wrong predicate/threshold or miss required rows.
  Arithmetic errors are minor (5–10%). The bottleneck is precise data selection.
- **v2 shifts the profile:** *wrong_tool_selection* rises to 20–23% (vs 4–10% in
  v1) because of distractor tools; the open-weight DeepSeek-V3 uniquely shows
  *round_limit_exhaustion* 25%.
- **Process metrics track capability:** frontier v1 models fail fast (1.3–1.9 tool
  calls, 0% round-exhaustion) with a single wrong query; the v2 agents make 3.9–4.1
  calls and exhaust the step budget 7–11% of the time.

Run: `python extract_failures.py && OPENROUTER_API_KEY=... python classify.py && python analyze.py`

---

## Summary tables (appended)

Classified **791** failing traces (GPT-5, o4-mini, GPT-4.1, GPT-4.1-mini, Claude-Sonnet-4.5, DeepSeek-V3). v1 = structured-data SQL agent (one tool); v2 = multi-tool agent with distractors. Per-model failure sets capped at 150 (seed 13) for classification; full failure counts and paper accuracies shown for context.

### 1. Overview

| Model | Version | Acc. (paper) | Total failures | Classified |
|---|---|---|---|---|
| GPT-5 | v1 | 68.9% | 342 | 150 |
| o4-mini | v1 | 67.1% | 362 | 150 |
| GPT-4.1 | v1 | 62.4% | 429 | 150 |
| GPT-4.1-mini | v1 | 61.5% | 413 | 150 |
| Claude-Sonnet-4.5 | v2 | 70.5% | 41 | 41 |
| DeepSeek-V3 | v2 | 57.3% | 484 | 150 |

### 2. Failure category distribution (% of classified failures, count in parentheses)

| Category | GPT-5 | o4-mini | GPT-4.1 | GPT-4.1-mini | Claude-Sonnet-4.5 | DeepSeek-V3 |
|---|---|---|---|---|---|---|
| Wrong tool / entity selection | 7% (11) | 8% (12) | 4% (6) | 10% (15) | 20% (8) | 23% (34) |
| Malformed arguments (wrong SQL predicate/filter/threshold) | 41% (61) | 37% (56) | 42% (63) | 36% (54) | 12% (5) | 7% (11) |
| Incomplete retrieval (missing rows/values) | 22% (33) | 31% (47) | 33% (49) | 37% (56) | 15% (6) | 16% (24) |
| Calculation / aggregation error | 7% (10) | 10% (15) | 5% (7) | 5% (8) | 17% (7) | 14% (21) |
| Financial-concept misunderstanding | 13% (20) | 10% (15) | 11% (17) | 11% (17) | 15% (6) | 5% (8) |
| Format / unit / rounding error | 1% (1) | 0% (0) | 2% (3) | 0% (0) | 15% (6) | 7% (11) |
| Round / step-limit exhaustion (no usable answer) | 9% (14) | 3% (4) | 3% (5) | 0% (0) | 7% (3) | 25% (38) |
| Other | 0% (0) | 1% (1) | 0% (0) | 0% (0) | 0% (0) | 2% (3) |
| **n classified** | 150 | 150 | 150 | 150 | 41 | 150 |

### 3. Process metrics on failing traces (mean over classified sample)

| Metric | GPT-5 | o4-mini | GPT-4.1 | GPT-4.1-mini | Claude-Sonnet-4.5 | DeepSeek-V3 |
|---|---|---|---|---|---|---|
| Tool calls / trace | 1.9 | 1.3 | 1.4 | 1.4 | 4.1 | 3.9 |
| Tool/SQL errors / trace | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 1.0 |
| Round/step-limit exhausted | 0% | 0% | 0% | 0% | 7% | 11% |

### 4. Aggregate by benchmark version (% of that version's classified failures)

| Category | v1 (SQL agent) | v2 (multi-tool) |
|---|---|---|
| Wrong tool / entity selection | 7% | 22% |
| Malformed arguments (wrong SQL predicate/filter/threshold) | 39% | 8% |
| Incomplete retrieval (missing rows/values) | 31% | 16% |
| Calculation / aggregation error | 7% | 15% |
| Financial-concept misunderstanding | 12% | 7% |
| Format / unit / rounding error | 1% | 9% |
| Round / step-limit exhaustion (no usable answer) | 4% | 21% |
| Other | 0% | 2% |
| **n** | 600 | 191 |

**Takeaways.** v1 failures are semantic not syntactic (SQL errors ≈ 0; dominated by wrong predicate/threshold and incomplete retrieval → precise data selection is the bottleneck, not arithmetic). v2 shifts to tool-use failures (wrong-tool selection ↑ under distractors; open-weight DeepSeek exhausts its step budget). Frontier v1 models fail fast with one wrong query; v2 agents take more steps and hit the limit more often. Two models at the same accuracy fail for measurably different reasons.

