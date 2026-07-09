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


---

## Coverage of the reviewer's requests (supplement)

This section closes the two items the tables above did not fully address: **qualitative failure examples** (failure ask) and **financial-concept coverage / explicit reasoning-operation mix / template diversity** (diversity ask). SQL complexity, tool-chain depth and numerical operations are in `../e6_diversity/`.

### Qualitative failure examples (one per category)

Each is a real trace; the explanation is the LLM judge's reasoning (v1) or the classifier rationale (v2).

**wrong_tool_selection** — DeepSeek-V3 (v2)
- *Query:* ────────
What was the difference in percentage cumulative total return for the five-year period ending 4/14 between NetApp Inc. and the S&P 500 Information Technology Index?

Your task
─────────
Use w
- *Expected:* -38.54%
- *Model answer:* The entities required for this task (NetApp, Inc., S&P 500 Information Technology Index) could not be found in the database. Therefore, the requested comparison
- *Why it failed:* The agent failed because it could not find the required entities due to using incorrect or incomplete entity resolution methods.

**malformed_arguments** — o4-mini (v1)
- *Query:* Are there invoices marked as paid but where shipment or service delivery confirmation is missing?
- *Expected:* Yes, there are invoices marked as paid but missing shipment or service delivery confirmation. Specifically, invoice 2 is missing service delivery confirmation, 
- *Model answer:* Yes. There is one paid invoice with neither a shipment nor a service‐delivery confirmation:

• Invoice ID: 4
- *Why it failed:* The correct answer states there are three paid invoices missing confirmation: invoice 2 (service missing), invoice 3 (shipment missing), and invoice 4 (both missing). The agent only mentions invoice 4 and ignores invoice

**incomplete_retrieval** — GPT-5 (v1)
- *Query:* What trends exist in delays in payments to our main suppliers compared to the last reporting period?
- *Expected:* For our main suppliers, payment delays show mixed trends compared to the last reporting period. Alpha Supplies experienced a clear increase in payment delays in
- *Model answer:* Here’s what I’m seeing for main suppliers, comparing the latest two payment months (by payment date): 2025-06 vs 2025-05.

Overall
- Share of invoices paid late
- *Why it failed:* The agent’s figures do not match the correct answer’s metrics. The correct answer shows Alpha Supplies’ average delay rising from 0.67 to 5 days (a 4.33-day increase) and Beta Goods improving from a 1.33-day delay to 0.5

**calculation_error** — o4-mini (v1)
- *Query:* Have adjustments in liability recognition impacted our compliance with financial covenants?
- *Expected:* Yes, adjustments in liability recognition have impacted compliance with financial covenants. Initially, the company was compliant with the "Max Liability Total"
- *Model answer:* A review of the covenant‐level snapshots and the timing/magnitude of your liability adjustments shows that the post-recognition adjustments have meaningfully in
- *Why it failed:* The agent’s answer is incorrect. They report liabilities of $950k on 03/31/2025 and $920k on 05/15/2025 yet still call those compliant under a $900k cap, which contradicts the covenant threshold. Their numbers and compli

**financial_misunderstanding** — o4-mini (v1)
- *Query:* Are there any inconsistencies in amortization schedules affecting intangible asset valuations quarter over quarter?
- *Expected:* Yes, there are inconsistencies in the amortization schedules affecting intangible asset valuations quarter over quarter. For example, Patent A (asset_id=1) show
- *Model answer:* Yes. When you match each quarter’s amortization to the change in valuation for the same asset/period, there are several mismatches (i.e. valuation change ≠ amor
- *Why it failed:* The agent’s answer identifies inconsistencies for all three assets (Assets 1, 2, and 3), but the correct analysis states that only Asset 1 shows inconsistencies while Assets 2 and 3 are consistent. Therefore, the agent’s

**format_unit_error** — DeepSeek-V3 (v2)
- *Query:* What is the value of the total premiums and ASO fees, in billions?
- *Expected:* 12.525
- *Model answer:* 3.8701
- *Why it failed:* Agent returned the correct value but failed to convert it into billions as requested.

**round_limit_exhaustion** — DeepSeek-V3 (v2)
- *Query:* What is the percentage change in comprehensive income attributable to NBCUniversal from 2014 to 2015?

Instructions for the assistant
• Think step-by-step, deciding which tool to call and with what ar
- *Expected:* 13%
- *Model answer:* Calling tools:
[{'id': '019f4656f5f9b3c3e6c15c7f6cacd3c6', 'type': 'function', 'function': {'name': 'calculate_percentage_change_tool', 'arguments': {'new_value
- *Why it failed:* The agent reached the maximum step limit before completing the final answer.

### Financial-concept coverage (% of examples mentioning each concept)

| Concept | v1 (n=8233) | v2 (n=548) |
|---|---|---|
| Accounts payable / invoices / vendors | 52.2% | 0.0% |
| Approval / authorization / controls | 18.1% | 1.5% |
| Overdue / aging / late payment | 15.6% | 11.1% |
| Variance / budget vs actual | 12.8% | 0.7% |
| Revenue recognition / deferred | 4.9% | 2.2% |
| Reconciliation / discrepancy | 6.0% | 0.4% |
| Tax / VAT | 3.8% | 4.4% |
| Cash flow / liquidity | 4.0% | 2.0% |
| Fraud / duplicate / anomaly | 2.7% | 0.2% |
| Ratios / financial statement (v2/FinQA) | 7.3% | 76.6% |

### Reasoning operations

**v1 task categories** (keyword-assigned, non-exclusive):

| Category | % of v1 |
|---|---|
| Accounts Payable analysis | 51.9% |
| Variance analysis | 12.8% |
| Data integrity & reconciliation | 6.5% |
| Revenue recognition | 7.7% |
| Financial reporting | 25.2% |

**v1 query operation type** (analyst intent):

| Operation | % of v1 |
|---|---|
| detect/identify (anomaly search) | 16.1% |
| list/retrieve (enumeration) | 22.4% |
| compute/quantify (aggregation) | 1.7% |
| compare (relative reasoning) | 4.0% |

**v2 numerical operations** (from `../e6_diversity/`): aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent-change 11% of reference plans.

### Template diversity

- v1 expansion: **12 seed queries → 8233 examples** (686× expansion) with cosine-0.9 near-duplicate filtering.
- v1 distinct queries: **8233/8233 = 100.0%** (no exact duplicates).
- v1 distinct-token-3-gram ratio: **0.5195**; distinct-4-gram ratio: **0.7395**.
- v1 high-overlap pair rate (token-Jaccard ≥ 0.8 on a 400-query sample): **0.0%** — templated phrasings are rare.
- v2 distinct questions: **548/548** (human-authored FinQA questions).

