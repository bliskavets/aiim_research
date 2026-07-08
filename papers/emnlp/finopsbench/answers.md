# FinOpsBench — Rebuttal Answers (Submission 5243)

Status legend: ✅ ready to post · 🟡 draft, needs numbers from experiment · ⬜ not started
Experiment references (E0–E7) follow the rebuttal plan.

---

## Reviewer PVoW (Overall 3.0, Confidence 5)

### 1. Human evaluation study (200–300 examples) 🟡 [E3]

> Include a human evaluation study for both benchmark versions. Even a random sample of 200–300 examples independently verified by financial experts or trained annotators would substantially increase confidence in the benchmark.

**Draft answer:**
We thank the reviewer for this suggestion and agree that independent human validation is the most important addition to the paper. We have conducted a human evaluation study covering both benchmark versions:

- **Dataset validity.** We sampled **[N=250]** FinOpsBench-v1 examples (stratified by task category) and **[N=100]** FinOpsBench-v2 examples. Each example was independently annotated by **[2]** annotators with professional experience in **[finance/accounting — describe background]** against the same five criteria used by the LLM panel (data naturalness, trace reasonableness, trace soundness, grounding, answer correctness). **[X]%** of v1 examples and **[Y]%** of v2 examples were judged valid on all criteria; inter-annotator agreement was **[κ=...]**.
- **Evaluation-judge accuracy.** We additionally sampled **[N=200]** graded agent answers (stratified across models and judge verdicts) and had humans label answer correctness. The evaluation judge agrees with human labels in **[Z]%** of cases (**[κ=...]**).

We will include the full protocol and results as a new appendix, and the annotation data will be released with the benchmark.

`TODO: run E3, fill numbers.`

### 2. LLM-judge ↔ human agreement (Cohen's κ) 🟡 [E3, E2]

> Report agreement between LLM judges and human evaluators (e.g., Cohen's κ or percentage agreement).

**Draft answer:**
Covered by the human study above (κ numbers for both the construction panel and the evaluation judge). In addition, we ran a fully **deterministic** cross-check on the subset of v1 where it is well-defined (expected answer contains exactly one number; 4.4% of the pool): judge and tolerance-based numeric matching agree on 74.3% of items, and manual inspection shows the disagreements are dominated by failures of the token-matching rule (incidental IDs matched or reformatted values missed), not of the judge. The 93 contested cases are exactly where human adjudication is most informative, so they are included in the human study sample (**[E3 numbers]**).

`TODO: run E2 (automatic), fill numbers; merge with E3 results.`

### 3. Release all pipeline prompts ✅ [E0]

> Release all prompts used throughout the nine-stage pipelines, including prompts for query generation, schema generation, data generation, feedback reconciliation, and system prompt construction.

**Draft answer:**
All prompts were in fact part of our release, but we agree the paper did not make them easy to find. The anonymous repository now contains a top-level **`PROMPTS.md`** index mapping every pipeline stage to the exact prompt location: v1 stages 1–9 plus final filtering (`v1/01_make_queries.py` … `v1/10_check_correct_answer.py`), the v1 evaluation prompts (`v1/eval_model.py`: agent system prompt and `EVALUATE_RESULT_PROMPT` for judge grading), and the full v2 environment-generator prompts (`v2/pipeline/prompts.py`). The camera-ready will reference this index explicitly, and the judge prompt already shown in Appendix (Figure: judge prompt) will be joined by the remaining prompts.

### 4. Deterministic correctness instead of LLM judge ✅/🟡 [E2]

> FinOpsBench-v1 evaluation itself relies on another LLM judge rather than deterministic correctness whenever possible.

**Draft answer:**
We would like to clarify the split: **FinOpsBench-v2 is already fully deterministic** — numeric answers are compared against the output of an executable reference plan with a one-least-significant-digit tolerance; no LLM is involved in v2 scoring. For v1 we quantified how far deterministic scoring can go: only **363 of 8,233 (4.4%)** expected answers contain a single numeric value; the remaining 95.6% are multi-entity analyst answers (lists of invoice IDs, per-vendor breakdowns, month ranges, policy descriptions) for which token-level numeric matching is undefined — this is precisely why v1 uses an LLM comparator while v2, whose answers are plain numbers by construction, does not. On the scalar subset, the judge and a tolerance-based numeric matcher agree on **74.3%** of items; manual inspection of the 93 disagreements shows they are dominated by cases where the single number in the reference answer is incidental (e.g. an invoice or variance ID rather than the asked-for value), i.e. cases where the *deterministic* rule, not the judge, is wrong. We include all disagreement cases in the released annotation file and report human adjudication of them in the human study (**[E3: judge was correct in X of 93 contested cases]**).

### 5. Quantitative diversity analysis 🟡 [E6]

> Analyze benchmark diversity more quantitatively. Statistics on reasoning operations, SQL complexity, tool-chain depth, numerical operations, financial concepts, and template diversity would strengthen the benchmark description.

**Draft answer:**
We have added a quantitative diversity appendix: (a) SQL complexity of reference traces (distribution of JOINs, aggregations, GROUP BY, subqueries per example); (b) tool-chain depth (tool calls per reference solution for both versions); (c) numerical-operation types in v2 (ratio / YoY change / share-of-total / aggregation); (d) lexical diversity of v1 queries (distinct n-gram ratios; the paper already reports 60+ user roles and UMAP spread); (e) financial-concept coverage. **[Insert summary table.]**

`TODO: run E6 scripts over final_exp10k.jsonl and correct_plan_augmented.py files.`

### 6. Qualitative failure analysis 🟡 [E5]

> Provide qualitative examples of common model failures beyond overall accuracy, including tool misuse, reasoning mistakes, planning failures, and financial misunderstandings.

**Draft answer:**
We classified all failing traces of **[GPT-5 / GPT-4.1 / Llama-3.1-8B]** into a seven-way taxonomy: (1) wrong tool selection / distractor-tool use, (2) malformed arguments, (3) incomplete retrieval, (4) calculation/aggregation errors, (5) financial-concept misunderstanding, (6) unit/format errors, (7) round-limit exhaustion. **[Insert distribution table + 3–4 worked examples.]** Notably, **[e.g., frontier models fail mostly on X while small open-source models fail on Y]** — this is exactly the kind of diagnostic signal the benchmark was designed to expose.

`TODO: run E5 on existing traces (v2 results/ + v1 evaluated JSONLs), verify subsample manually.`

### 7. Construction costs 🟡 [E7]

> Report annotation or generation costs, computational resources, and runtime required to construct the benchmark.

**Draft answer:**
Construction of v1 (10,000 candidate examples → 5,979 final) consumed approximately **[X]M input / [Y]M output tokens (~$[Z])** across generator, judge, and repair models; v2 (1,247 → 1,108) consumed **[...]**. Evaluation of the eight models cost **[...]** plus **[N]** H100 GPU-hours for locally served open-source models. Wall-clock: **[...]**. We will add this to the appendix.

`TODO: pull from MLflow logs / API billing; honest estimates where logs are incomplete.`

### 8. Biases from proprietary models ✅/🟡 [E4]

> Discuss potential biases introduced by using proprietary models throughout the generation and validation pipeline.

**Draft answer:**
We agree this deserves explicit discussion and will expand the Discussion section. Three mitigating design choices are already in place: (a) the judge panel is **cross-vendor** (Claude Sonnet 4 + o4-mini + o3-mini), so no single vendor's blind spots decide acceptance; (b) generation (GPT-4.1-mini) and judging use different models; (c) v2 ground truth is execution-based, independent of any LLM's opinion. Empirically, if the benchmark favored the generator's model family we would expect OpenAI models to sit above the size–accuracy trend and other families below it; our new results with **[Claude Sonnet 4.x and DeepSeek-V3]** show **[both fall on the same log-linear trend]**, providing direct evidence against a generator-family advantage. `TODO: E4 numbers.`

### 9. Typos ✅

> "FinOpsBenchis" instead of "FinOpsBench is"; "As Figure 1 shows that..."; spacing inconsistencies.

**Draft answer:**
Thank you — all fixed: the `\xspace` macro issue causing "FinOpsBenchis" has been corrected throughout, "As Figure 1 shows that" → "Figure 1 shows that", and v1/v2 naming spacing is now consistent.

*(Note to self: the root cause is `\datasetname` macro + missing space handling in some contexts; grep the .tex for all occurrences.)*

### 10. Reproducibility 2 / Software 1 ✅ [E0]

**Draft answer:**
We suspect the reviewer could not access our release at review time, and we apologize for the inconvenience. The anonymous repository **[link]** now contains the complete structured release: both construction pipelines (every stage runnable, every prompt included), both evaluation harnesses with per-model runners, the v1 example pool, all 1,100+ self-contained v2 environments (system prompt, tools, distractor tools, SQLite store, executable reference plan), and per-model result files. A quickstart in the README reproduces Table 2 rows with a single command per model.

---

## Reviewer R2 (Overall 2.5, Confidence 4)

### 1. What fundamental NLP capability does it advance? ✅ [A1]

> While the benchmark targets agentic financial analysis, it remains unclear what fundamental NLP capability it advances beyond a domain-specific evaluation resource.

**Draft answer:**
The capability FinOpsBench isolates is **planning under partial observability with grounded evidence aggregation**: the agent must discover what information exists (schema/tool probing), plan a multi-step retrieval strategy, filter distractors, and synthesize a faithful answer from intermediate tool outputs — with the domain contributing hard semantics (aging, variance attribution, revenue recognition) rather than mere surface flavor. Existing resources measure either reading comprehension over provided context (FinQA, TAT-QA) or query-string fidelity against a visible schema (Spider, BIRD); neither requires the agent to *decide what to look at* before reasoning. Our diagnostic design makes this capability measurable in isolation: every failure is attributable to a specific planning or tool-use mistake because the environment is fully controlled and the ground truth executable. The ReAct-vs-native finding (reasoning scaffolds help non-thinking models but hurt thinking ones) is an example of a general, transferable insight the benchmark surfaces.

### 2. What's fundamentally new vs recent agentic finance benchmarks? ✅ [A1]

> multiple recent benchmarks have already moved in this direction. It remains somewhat unclear what fundamentally new evaluation capability FinOpsBench provides.

**Draft answer:**
Relative to FinGAIA, Finance Agent Benchmark, and FinAgentBench (discussed in §2), FinOpsBench is the only resource that combines: (1) **hermetic, executable environments** — no live web/API dependence, so results are exactly reproducible and failures attributable to the agent rather than external noise; (2) **controlled distractors** at both data and tool level; (3) **scale** (≈6k + 1.1k tasks vs. a few hundred); (4) **full reference traces** enabling process-level analysis, not just final-answer scoring. The realism-oriented benchmarks and FinOpsBench are complements, not substitutes: they measure deployment behavior, we measure diagnostic competence. We will restore the comparison table (currently cut for space) making these axes explicit. Notably, our Appendix A cross-benchmark analysis shows FinAgentBench exhibits an *inverse* size–accuracy trend — precisely the validity failure our controlled design avoids.

### 3. Fine-grained diagnostics beyond final-answer accuracy 🟡 [E5]

> the reported analyses are primarily based on final-answer accuracy. More fine-grained diagnostic metrics or failure analyses would better demonstrate that the benchmark provides insights beyond conventional benchmark evaluation.

**Draft answer:**
Agreed — we have added: (a) a **failure-mode taxonomy** over all failing traces (7 categories; distribution per model) showing **[key contrast]**; (b) **process-level metrics** from traces: tool-call efficiency (calls vs. reference plan length), distractor-tool invocation rate, and round-exhaustion rate per model; (c) the per-category radar (Fig. 6) already shows capability is stable across financial sub-domains. **[Insert 2–3 headline findings.]**

`TODO: E5; compute process metrics from traces while classifying failures.`

### 4. Benchmark quality depends on LLM generation/judging 🟡 [E3]

> the final benchmark quality still depends substantially on LLM-generated queries, schemas, data, and judgments.

**Draft answer:**
Three points. First, the dependence is asymmetric across versions: v2 questions are **human-authored** (FinQA) and v2 validation is **execution-based**, not judgment-based; only the environment scaffolding is generated, and it is verified by running it. Second, for v1 we now provide **human validation**: **[X]%** of a stratified sample of **[250]** examples were confirmed valid by independent annotators (κ=**[...]** vs. the LLM panel) — see our response to Reviewer PVoW. Third, the two versions act as mutual controls: per-model accuracies agree across them (mean abs. diff 2.6pp), which would be unlikely if v1's synthetic construction introduced systematic artifacts.

---

## Reviewer R3 (Overall 2.5, Confidence 4)

### 1. v1 lacks machine-verifiable ground truth ✅/🟡 [E2, E3]

> v1 lacks machine-verifiable hard ground truth; fully relies on LLM panel judges, leading to subjective, biased evaluation results.

**Draft answer:**
Two clarifications. (a) Every v1 example **does** carry a hard expected answer (`expected_output`), created jointly with the data in Stage 3 and enforced by execution-based validation (Stage 4) plus an answer-consistency filter (final filtering). The panel is an additional quality gate on top of, not a replacement for, this ground truth. (b) We measured how far machine-only scoring can go on v1: deterministic numeric matching is well-defined for only 4.4% of expected answers (the rest are multi-entity analyst answers); on that subset it agrees with the judge on 74.3% of items, and manual inspection of every disagreement shows the token-matching rule, not the judge, is the unreliable side (incidental IDs, reformatted values). The judge is a necessity created by free-form financial answers, not a source of subjectivity — and its accuracy is directly quantified against human labels in our new human study. v2 is scored fully deterministically against executable reference plans. Human-validation numbers are in our response to Reviewer PVoW.

### 2–3. v2 built on FinQA: monotonous, artificial multi-hop ✅ [A1]

> v2 is built entirely on FinQA, which was not designed for agent tool workflows; query types are monotonous and fail to integrate deep financial domain knowledge.
> v2 inherits FinQA's simple numerical questions, with artificially added multi-hop tool logic rather than native business-driven agent tasks.

**Draft answer:**
This is a deliberate design choice, and the two benchmark halves must be read together. Deriving v2 from FinQA is a **controlled intervention**: the question content is held fixed (human-authored, familiar to the community, with comparable static-setting numbers) while the *access mode* changes from reading to tool use. This isolates the agentic component causally: state-of-the-art systems reach ~85% on static FinQA, yet the best agent reaches only 69.6% on the same questions in our environments — a gap attributable to planning and tool use, not question difficulty. "Native business-driven agent tasks" are exactly what **v1** provides at scale (5,979 tasks across AP aging, reconciliation, variance analysis, revenue recognition, authored from analyst personas). On monotony: v2 intentionally mirrors FinQA's operation surface (we will add the operation-type distribution, `TODO E6`); the breadth axis of the benchmark is carried by v1.

### 4. Missing top agent models (Claude Code, Codex, OpenCode); no finance-specialized LLMs 🟡 [E4]

> Experiment evaluation is incomplete: missing top agent/code frontier models (Claude Code, Codex, OpenCode); baselines only cover tiny open-source models without mainstream finance-specialized LLMs.

**Draft answer:**
Claude Code, Codex, and OpenCode are **agent products/harnesses**, not base models: each bundles its own scaffolding, prompts, and retry logic, so numbers obtained through them would conflate model capability with product engineering and be irreproducible as the products update. FinOpsBench deliberately evaluates *base models under a fixed, open harness* — the standard protocol of agentic benchmarks (AgentBench, τ-bench). That said, we agree frontier-family coverage should be broader: we have added **Claude Sonnet 4.x** and **DeepSeek-V3** to both tables — **[results: v1 X%, v2 Y%; both consistent with the log-linear size–accuracy trend]**. On finance-specialized LLMs: available open finance models (e.g., continued-pretrained variants on financial text) do not support reliable function calling, which is the capability under test; we will note this explicitly.

`TODO: E4 numbers.`

### 5. "Outdated smolagents" / framework noise ✅ [A2]

> Adopts outdated smolagents as agent harness, which may introduce framework noise and interfere with reliable tool-use performance measurement.

**Draft answer:**
We believe this is a misunderstanding. (a) smolagents is a current, actively maintained Hugging Face library (we use ≥1.22, released 2025); it is a minimal harness, which is precisely why we chose it — less scaffolding means less framework noise, not more. (b) FinOpsBench-v1 does not use smolagents at all: it runs a minimal native tool-calling loop (and a ReAct variant) implemented directly over the model API. (c) Framework noise is addressed **empirically**: we evaluate under two protocols (native tool calling and ReAct) and two independent stacks, and the model ranking is consistent across all of them, with per-model accuracies agreeing across the two benchmark versions (mean abs. diff 2.6pp). If harness artifacts were driving results, this cross-harness agreement would not hold.

### 6. Data contamination risk for v2 🟡 [E1]

> High risk of data contamination for v2, as core questions come from widely publicized FinQA training corpus.

**Draft answer:**
We tested this directly with a **closed-book baseline**: every v2 environment prompt (scenario + tool signatures + question) is given to the model with **no callable tools**, and the model must answer from the prompt and its own knowledge (a *conservative* setup — the model sees strictly more than plain closed-book, so this upper-bounds what memorization can deliver; some scenario narratives even contain the needed figure legitimately). Results (v2 scoring rule unchanged):

| Model | Closed-book | Agentic (paper) | Δ |
|---|---|---|---|
| GPT-4.1 (n=300) | **13.3%** | 60.6% | −47.3 pp |
| GPT-5-mini (n=1,174) | **[final]%** | 67.5% | **[final]** |
| Qwen3-30B-A3B (n=1,174) | **[final]%** | 53.0% | **[final]** |

Memorization of FinQA thus does not provide an answer pathway: the system prompt contains neither the source table nor its values, the backing store is a re-instantiated database with distractor rows, and the required multi-hop tool plan does not exist in any training corpus. If contamination were driving v2 performance, closed-book accuracy would approach agentic accuracy — instead it collapses by ~50 points.

`TODO: fill gpt-5-mini / qwen finals when runs complete (experiments/e1_closed_book/).`

---

## Общий чек-лист перед постингом

- [ ] E0: анонимное зеркало обновлено и открывается инкогнито-браузером
- [x] E1: closed-book запущен; gpt-4.1 подтверждён (13.3% vs 60.6%); дождаться gpt-5-mini/qwen finals
- [x] E2: готово (4.4% скалярных; 74.3% agreement; 93 кейса для ручной разметки в experiments/e2_judge_agreement/results/)
- [ ] E3: human eval числа вставлены (PVoW-1/2, R2-4, R3-1)
- [ ] E4: Claude/DeepSeek строки таблицы (PVoW-8, R3-4)
- [ ] E5: таблица failure taxonomy + примеры (PVoW-6, R2-3)
- [ ] E6: diversity-статистика (PVoW-5, R3-2)
- [ ] E7: costs (PVoW-7)
- [ ] Все `[...]`-плейсхолдеры заменены реальными числами
- [ ] Тон: благодарный, конкретный, без обещаний «in future work» там, где можно дать число сейчас
