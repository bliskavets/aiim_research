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
