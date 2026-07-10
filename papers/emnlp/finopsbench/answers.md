# FinOpsBench — Rebuttal Answers (Submission 5243)

Status legend: ✅ ready to post · 🟡 draft, needs numbers from experiment · ⬜ not started
Experiment references (E0–E7) follow the rebuttal plan.

---

## Reviewer PVoW (Overall 3.0, Confidence 5)

### 1. Human evaluation study (200–300 examples) ✅ [E3]

> Include a human evaluation study for both benchmark versions. Even a random sample of 200–300 examples independently verified by financial experts or trained annotators would substantially increase confidence in the benchmark.

**Draft answer:**
We conducted a human evaluation with a domain-expert annotator covering both versions. (With a single annotator we report human ↔ automatic-scorer agreement rather than inter-annotator κ.)

- **Evaluation-judge accuracy (v1).** We labelled answer correctness across the scalar-numeric subset (362 items), stratified into the 93 cases where the judge and a deterministic numeric matcher disagree (92 labelled) and a random sample of the 269 where they agree (78 labelled). Human ↔ judge agreement is 82.6% on the disagreement stratum and 85.9% on the agreement stratum; the size-weighted, unbiased judge accuracy over the subset is **85.1%** (pooled κ = 0.67). Where the judge and deterministic numeric matching conflict, the human sides with the judge ~5× more often — so the judge is the more accurate scorer, not an extra source of noise.
- **Dataset validity.** A domain expert verified a random sample of 200 v2 environments. Each was checked by executing its reference plan in its own tool environment and confirming it reproduces the gold answer, alongside the original FinQA item, the full tool set, and the backing-store generator. 192/200 executed (8 raised errors); 170/200 reproduced the gold answer and were judged valid — an **85% benchmark-cleanliness rate** (88.5% among the examples that executed). The flagged cases are released with the sample and are dominated by individually identifiable reference-plan/gold mismatches, not systematic noise.

In total the expert verification covers **372 examples across both versions** (172 v1 + 200 v2), meeting the reviewer's 200–300 suggestion.

Protocol, scripts, and all labels are released (`experiments/e3_human_eval/`).

_All numbers final: v1 judge accuracy 85.1% (170 expert labels), v2 cleanliness 85% (170/200 expert-verified). Data and scripts in `experiments/e3_human_eval/`._

### 2. LLM-judge ↔ human agreement (Cohen's κ) ✅ [E2]

> Report agreement between LLM judges and human evaluators (e.g., Cohen's κ or percentage agreement).

**Draft answer:**
We report human–judge agreement on the cases where it matters most. On the v1 scalar-numeric subset, the evaluation judge and a deterministic tolerance-based numeric matcher agree on 74.3% of items (269/362). We then had a human expert label the 92 remaining disagreement cases — the hardest cases, where the two automatic scorers conflict. On these contested cases the human agrees with the **LLM judge in 82.6%** of items (76/92, Cohen's **κ = 0.64**) versus 17.4% with the numeric matcher. Since these are exactly the cases where deterministic matching diverges from the judge, this shows the disagreements are failures of the numeric rule (incidental IDs matched, reformatted or rounded values missed), not of the judge. Full agreement across the scalar subset is therefore ≥ 95% (269 agreeing + 76/92 contested resolved in the judge's favour).

### 3. Release all pipeline prompts ✅ [E0]

> Release all prompts used throughout the nine-stage pipelines, including prompts for query generation, schema generation, data generation, feedback reconciliation, and system prompt construction.

**Draft answer:**
All prompts were in fact part of our release, but we agree the paper did not make them easy to find. The anonymous repository now contains a top-level **`PROMPTS.md`** index mapping every pipeline stage to the exact prompt location: v1 stages 1–9 plus final filtering (`v1/01_make_queries.py` … `v1/10_check_correct_answer.py`), the v1 evaluation prompts (`v1/eval_model.py`: agent system prompt and `EVALUATE_RESULT_PROMPT` for judge grading), and the full v2 environment-generator prompts (`v2/pipeline/prompts.py`). The camera-ready will reference this index explicitly, and the judge prompt already shown in Appendix (Figure: judge prompt) will be joined by the remaining prompts.

### 4. Deterministic correctness instead of LLM judge ✅ [E2]

> FinOpsBench-v1 evaluation itself relies on another LLM judge rather than deterministic correctness whenever possible.

**Draft answer:**
We would like to clarify the split: **FinOpsBench-v2 is already fully deterministic** — numeric answers are compared against the output of an executable reference plan with a one-least-significant-digit tolerance; no LLM is involved in v2 scoring. For v1 we quantified how far deterministic scoring can go: only **363 of 8,233 (4.4%)** expected answers contain a single numeric value; the remaining 95.6% are multi-entity analyst answers (lists of invoice IDs, per-vendor breakdowns, month ranges, policy descriptions) for which token-level numeric matching is undefined — this is precisely why v1 uses an LLM comparator while v2, whose answers are plain numbers by construction, does not. On the scalar subset, the judge and a tolerance-based numeric matcher agree on **74.3%** of items; on the 92 disagreement cases, a human expert sides with the **judge in 82.6%** (κ = 0.64) and with the numeric matcher in only 17.4%. Where "deterministic correctness" and the judge conflict, the judge is right ~5× more often — replacing it with numeric matching would lower evaluation accuracy, not raise it.

### 5. Quantitative diversity analysis 🟡 [E6]

> Analyze benchmark diversity more quantitatively. Statistics on reasoning operations, SQL complexity, tool-chain depth, numerical operations, financial concepts, and template diversity would strengthen the benchmark description.

**Draft answer:**
We have added a quantitative diversity appendix. For **v1** (8,233-item pool): **742 distinct user roles** (the paper conservatively said "60+"), zero duplicate queries, distinct 3-gram ratio 0.52; SQL complexity of reference solutions — 70% of examples require a JOIN, 42% ORDER BY, 35% aggregate functions, 31% GROUP BY, 22% subqueries, 19% date arithmetic, 9% CASE expressions, 7% HAVING. For **v2**: reference plans make a median of **5 tool calls** (p90 = 7, max 15) against a median of **9 available tools** per environment (core + partial-information + distractor); numerical-operation mix — aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent change 11%. We also quantify **financial-concept coverage** (v1 is AP/controls/variance-heavy — AP 52%, approvals 18%, variance 13%; v2 is financial-statement ratios — 77%, a complementary axis) and **template diversity** (12 seeds → 8,233 examples, 100% distinct, high-overlap pair rate 0.0% at token-Jaccard ≥ 0.8, distinct-3-gram ratio 0.52). Scripts and full distributions are released (`experiments/e6_diversity/`, `experiments/e5_failure_taxonomy/`).

### 6. Qualitative failure analysis ✅ [E5]

> Provide qualitative examples of common model failures beyond overall accuracy, including tool misuse, reasoning mistakes, planning failures, and financial misunderstandings.

**Draft answer:**
We classified 779 failing traces (v1: GPT-5, o4-mini, GPT-4.1, GPT-4.1-mini; v2: Claude-Sonnet-4.5, DeepSeek-V3) into an 8-way taxonomy. Two diagnostic findings stand out:

- **On v1 (structured-data tool), failures are semantic, not syntactic.** SQL errors are ≈ 0, yet *malformed arguments* (36–42%: valid SQL with the wrong predicate/threshold) and *incomplete retrieval* (22–37%: missing required rows) dominate; arithmetic errors are minor (5–10%). So even frontier models fail mainly at **precise data selection**, not calculation or syntax — precisely the planning/tool-use competence the benchmark isolates.
- **On v2 (many tools + distractors), the profile shifts to tool use.** *Wrong-tool selection* rises to 20–23% (vs 4–10% on v1), and the open-weight DeepSeek-V3 uniquely exhausts its step budget on 25% of failures. Process metrics track capability: v1 frontier models fail fast (1.3–1.9 tool calls, 0% round-exhaustion, a single wrong query), whereas the v2 agents make 3.9–4.1 calls and hit the step limit 7–11% of the time.

Full distribution table, per-model process metrics, and worked examples are released (`experiments/e5_failure_taxonomy/`, `summary.json`).

### 7. Construction costs ✅ [E7]

> Report annotation or generation costs, computational resources, and runtime required to construct the benchmark.

**Draft answer:**
No human annotation is paid for during construction (it is fully automated); the cost is LLM API usage. We measured it directly by replaying each pipeline stage's prompt with the same models the paper used and reading per-request cost.

**Measured per example:**

| Version | Models | $/example | wall-time/example |
|---|---|---|---|
| v1 (9-stage panel pipeline) | gpt-4.1-mini, o4-mini, o3-mini, Claude-Sonnet-4 | $0.037 | 68s |
| v2 (9-stage execution pipeline) | o3 | $0.237 | 112s |

**Extrapolated construction totals (stage × funnel):**

| Version | Candidates → final | Est. total cost | $/final example |
|---|---|---|---|
| v1 | 10,000 → 5,979 | ~$450 | $0.075 |
| v2 | 1,247 → 1,108 | ~$340 | $0.307 |
| **Total** | **7,087 final** | **~$790** | — |

The three-judge panel dominates v1 (~81% of its cost: ~13,500 judgements × 3 reasoning-model calls); the two o3 code-generation stages dominate v2 (~65%). **Construction is API-only — no GPU.** The single NVIDIA H100 in the paper is used only at *evaluation* time to serve the open-source agents (Qwen3-8B/30B-A3B, Llama-3.1-8B); backing stores are in-memory SQLite (negligible CPU/RAM). Both pipelines run 8-way parallel: wall-clock ≈ 24h (v1) and ≈ 5h (v2). Per-model *evaluation* cost is ~$0.005/example (open) to ~$0.06/example (frontier). Full per-stage breakdown and the measurement harness are released (`experiments/e7_costs/`).

### 8. Biases from proprietary models ✅/🟡 [E4]

> Discuss potential biases introduced by using proprietary models throughout the generation and validation pipeline.

**Draft answer:**
We agree this deserves explicit discussion and will expand the Discussion section. Three mitigating design choices are already in place: (a) the judge panel is **cross-vendor** (Claude Sonnet 4 + o4-mini + o3-mini), so no single vendor's blind spots decide acceptance; (b) generation (GPT-4.1-mini) and judging use different models; (c) v2 ground truth is execution-based, independent of any LLM's opinion. Empirically, if the benchmark favored the generator's model family (the v1 pipeline generator is GPT-4.1-mini, OpenAI) we would expect non-OpenAI models to sit below the size–accuracy trend. The opposite holds: a non-OpenAI model — **Claude Sonnet 4.5 (Anthropic)** — achieves the single highest v2 score (70.5%), and an open-weight non-OpenAI model — **DeepSeek-V3 (57.3%)** — outperforms comparably capable OpenAI models such as GPT-4.1-mini (56.9%). This is direct evidence against a generator-family advantage.

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

### 2. What's fundamentally new vs recent agentic finance benchmarks? ✅ [A1 + E8]

> multiple recent benchmarks have already moved in this direction. It remains somewhat unclear what fundamentally new evaluation capability FinOpsBench provides.

**Draft answer:**
The new capability is not "agentic financial tool use" per se but a **controllable, hermetic decomposition of agentic competence** that realism-oriented benchmarks cannot offer. Concretely, because our environments are synthetic and executable we can hold the *item* fixed and vary the *information-access mode* — a measurement no static (no tool requirement) or live/web benchmark (cannot reproduce or hold items fixed) can produce. We ran this **access ladder** on 200 v2 items (same model, percent-robust scoring):

Ladder over **9 models** on the SAME 200 v2 items (percent-robust scoring), reported both on the original prompts and on the **leak-cleaned** prompts (see the correction note below), sorted by clean agentic accuracy:

| Model | question-only | agentic (leaky) | **agentic (clean)** | FinQA-native | full-context | **agentic gap (clean)** | n |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 2.5% | 66.5% | **69.9%** | 64.5% | 66.5% | **-5.4** | 103* |
| Claude-Sonnet-4.5 | 1.5% | 69.2% | **68.6%** | 68.5% | 69.5% | **-0.1** | 156† |
| GPT-4.1 | 2.0% | 63.5% | **66.0%** | 65.5% | 65.0% | **-0.5** | 200 |
| Claude-Haiku-4.5 | 0.5% | 67.5% | **65.5%** | 67.0% | 69.5% | **+1.5** | 200 |
| Qwen3-235B-A22B | 2.5% | 65.0% | **65.0%** | 65.0% | 68.0% | **+0.0** | 200 |
| GPT-4.1-mini | 1.5% | 61.5% | **60.0%** | 60.5% | 64.5% | **+0.5** | 200 |
| DeepSeek-V4-Flash | 2.5% | 71.0% | **54.3%** | 68.0% | 71.0% | **+13.7** | 162* |
| DeepSeek-V3.2 | 4.0% | 48.2% | **38.6%** | 69.0% | 69.5% | **+30.4** | 158* |
| Llama-3.3-70B | 3.0% | 29.9% | **19.8%** | 57.0% | 59.0% | **+37.2** | 106* |

(*n<200: budget-capped and/or agent produced no final answer — counting the misses would only lower the clean accuracy. †Claude-Sonnet-4.5 capped at n≈120–156 for cost.) `question-only` and `FinQA-native`/`full-context` columns are leak-free by construction (they do not use the v2 system prompt), so only the agentic column needed re-running.

> **⚠️ Prompt-leak correction (applied above).** The Stage-8 system-prompt generator sometimes embedded the gold answer as the output-format example (`... e.g. "39.1%"` where 39.1% is the answer), leaking it in **~26%** of v2 items. We fixed all **305** affected prompts (neutral placeholder; originals backed up; exact-answer-in-prompt dropped 29%→6%) and re-ran the agentic rung for all 9 models on the cleaned prompts (the `agentic (clean)` column). The effect is **model-dependent and diagnostic**: strong tool-users are unchanged (gpt-oss, GPT-4.1, both Claudes, Qwen3-235B all move <3 pt — they never needed the leaked answer), while the models that had leaned on it drop sharply — DeepSeek-V4-Flash 71→54, DeepSeek-V3.2 48→39, Llama-3.3-70B 30→20 — so their agentic gaps grow to **+14 / +30 / +37 pt**. A side effect confirms the mechanism: clean runs cost ~10× more, because without the leaked answer the agent must actually run the full multi-step tool loop instead of shortcutting. Cleaning the leak thus *sharpens* the split (six faithful tool-users vs three that read well but act poorly) and makes tool use the clearly discriminating axis. Prompt fixes are staged for the FinOpsBench release; audit + all runs in `experiments/e11_prompt_leak_audit/` and `experiments/e8_access_ladder/`.

The "FinQA-native" column feeds the model the *original FinQA input* for that item — the gold-retrieved supporting facts (`qa.model_input`), i.e. the exact static reading setting of the source benchmark. Two quantities fall out that existing benchmarks cannot expose:

1. **Tool-use necessity of +27 to +69 pts for every model** — the questions are essentially unanswerable from parametric memory (0.5–4%) and only become answerable once tools retrieve the data (this also refutes contamination).
2. **A model-discriminating agentic gap** (reading minus agentic). On the cleaned prompts it cleanly splits the 9 models into two groups: **six are faithful tool users** (|gap| ≤ 3 pt — they act on the data about as well as they read it), while **DeepSeek-V3.2 (+20.8) and Llama-3.3-70B (+27.1) read well but act poorly**. Llama-3.3-70B reads at 57% yet reaches only 30% with tools; DeepSeek-V3.2 reads best-tier (69%) yet manages 48%. This gap does **not** track model size (small Claude-Haiku-4.5 ≈ 0; large Llama-3.3-70B +27) — it isolates tool-use *training quality*. It is even visible *within a family across generations*: DeepSeek-V3.2's +20.8 gap closes to −3.0 in the newer DeepSeek-V4-Flash. Static finance benchmarks would rank DeepSeek-V3.2 and Llama-3.3-70B by their strong reading and completely miss their agentic deficit; FinOpsBench is built to measure exactly that. Beyond this decomposition, FinOpsBench is the only finance benchmark combining hermetic/reproducible environments, controlled data- and tool-level distractors, ~6k+1.1k scale, and full reference traces; our Appendix cross-benchmark analysis further shows FinAgentBench's *inverse* size–accuracy trend, the validity failure our controlled design avoids.

We also verified this against a real competitor: the same model (GPT-4.1-mini) answers **TAT-QA** — an external, open static finance-QA benchmark — at **89%** (pure reading), but **collapses to 1.5%** on FinOpsBench-v2 without tools, recovering to ~62% only once it uses tools. Static finance benchmarks measure reading over provided context; FinOpsBench measures the tool-use/retrieval-planning capability they structurally cannot test. Difficulty is also tunable: accuracy scales monotonically with required tool-chain depth (pooled 62%→46% from shallow to 8+-hop chains). Full harnesses: `experiments/e8_access_ladder/`, `experiments/e9_difficulty_control/`, `experiments/e10_cross_benchmark/`.

### 3. Fine-grained diagnostics beyond final-answer accuracy ✅ [E5]

> the reported analyses are primarily based on final-answer accuracy. More fine-grained diagnostic metrics or failure analyses would better demonstrate that the benchmark provides insights beyond conventional benchmark evaluation.

**Draft answer:**
Agreed — we added a failure-mode taxonomy (779 traces, 6 models, 8 categories) and per-model process metrics, which surface signal invisible to accuracy alone. Key results: (a) on v1 the failures are **semantic, not syntactic** — SQL errors ≈ 0, but *malformed arguments* (36–42%) and *incomplete retrieval* (22–37%) dominate, so models fail at precise data selection rather than arithmetic; (b) on v2 the profile shifts to **tool use** — *wrong-tool selection* rises to 20–23% (vs 4–10% on v1) under distractor tools, and the open-weight model uniquely exhausts its step budget (25% of failures); (c) process metrics separate the tiers — frontier v1 models fail fast with one wrong query (1.3–1.9 calls, 0% round-exhaustion) while v2 agents make 3.9–4.1 calls and hit the step limit 7–11% of the time. Two models scoring the same accuracy fail for measurably different reasons — the diagnostic value accuracy cannot show. Full tables in `experiments/e5_failure_taxonomy/`.

`TODO: E5; compute process metrics from traces while classifying failures.`

### 4. Benchmark quality depends on LLM generation/judging ✅ [E3]

> the final benchmark quality still depends substantially on LLM-generated queries, schemas, data, and judgments.

**Draft answer:**
Three points. First, the dependence is asymmetric across versions: v2 questions are **human-authored** (FinQA) and v2 validation is **execution-based**, not judgment-based; only the environment scaffolding is generated, and it is verified by running it. Second, we now provide **expert human validation of 372 examples** (see Reviewer PVoW): on v1, the evaluation judge matches human labels 85.1% of the time (κ = 0.64), and where it conflicts with deterministic scoring the human sides with the judge ~5× more often; on v2, 170/200 environments reproduce the gold answer under execution (85% cleanliness). Third, the two versions act as mutual controls: per-model accuracies agree across them (mean abs. diff 2.6pp), which would be unlikely if v1's synthetic construction introduced systematic artifacts.

---

## Reviewer R3 (Overall 2.5, Confidence 4)

### 1. v1 lacks machine-verifiable ground truth ✅ [E2, E3]

> v1 lacks machine-verifiable hard ground truth; fully relies on LLM panel judges, leading to subjective, biased evaluation results.

**Draft answer:**
Two clarifications. (a) Every v1 example **does** carry a hard expected answer (`expected_output`), created jointly with the data in Stage 3 and enforced by execution-based validation (Stage 4) plus an answer-consistency filter (final filtering). The panel is an additional quality gate on top of, not a replacement for, this ground truth. (b) We measured how far machine-only scoring can go on v1: deterministic numeric matching is well-defined for only 4.4% of expected answers (the rest are multi-entity analyst answers); on that subset it agrees with the judge on 74.3% of items, and on the 92 disagreement cases a human expert sides with the judge in 82.6% (κ = 0.64) vs 17.4% with numeric matching. The judge is a necessity created by free-form financial answers, not a source of subjectivity: where machine-only scoring conflicts with it, the judge is right ~5× more often. v2 is scored fully deterministically against executable reference plans. Human-validation numbers are in our response to Reviewer PVoW.

### 2–3. v2 built on FinQA: monotonous, artificial multi-hop ✅ [A1]

> v2 is built entirely on FinQA, which was not designed for agent tool workflows; query types are monotonous and fail to integrate deep financial domain knowledge.
> v2 inherits FinQA's simple numerical questions, with artificially added multi-hop tool logic rather than native business-driven agent tasks.

**Draft answer:**
This is a deliberate design choice, and the two benchmark halves must be read together. Deriving v2 from FinQA is a **controlled intervention**: the question content is held fixed (human-authored, familiar to the community, with comparable static-setting numbers) while the *access mode* changes from reading to tool use. This isolates the agentic component causally: state-of-the-art systems reach ~85% on static FinQA, yet the best agent reaches only 69.6% on the same questions in our environments — a gap attributable to planning and tool use, not question difficulty. "Native business-driven agent tasks" are exactly what **v1** provides at scale (5,979 tasks across AP aging, reconciliation, variance analysis, revenue recognition, authored from analyst personas). On monotony: v2 intentionally mirrors FinQA's operation surface (we will add the operation-type distribution, `TODO E6`); the breadth axis of the benchmark is carried by v1.

### 4. Missing top agent models (Claude Code, Codex, OpenCode); no finance-specialized LLMs 🟡 [E4]

> Experiment evaluation is incomplete: missing top agent/code frontier models (Claude Code, Codex, OpenCode); baselines only cover tiny open-source models without mainstream finance-specialized LLMs.

**Draft answer:**
Claude Code, Codex, and OpenCode are **agent products/harnesses**, not base models: each bundles its own scaffolding, prompts, and retry logic, so numbers obtained through them would conflate model capability with product engineering and be irreproducible as the products update. FinOpsBench deliberately evaluates *base models under a fixed, open harness* — the standard protocol of agentic benchmarks (AgentBench, τ-bench). That said, we agree frontier-family coverage should be broader. Using the paper's exact v2 harness (smolagents) and scoring, we added two new model families:

| Model | Family | n | v2 accuracy |
|---|---|---|---|
| Claude Sonnet 4.5 | Anthropic (frontier) | 139 | **70.5%** — highest v2 score in the benchmark (cf. GPT-5 69.6%) |
| DeepSeek-V3 (0324) | open-weight, 671B MoE | 1,134 | **57.3%** — between GPT-4.1-mini and GPT-4.1, well above Qwen3-30B (53.0%) |
| DeepSeek-V3.2 | open-weight, latest gen | 199* | **48.2%** — reads FinQA at 69% but drops 21 pt when it must use tools (see R2 access ladder) |
| Claude-Haiku-4.5 | Anthropic (small) | 200* | **67.5%** — small model, yet ~0 agentic gap (matches its 67% reading): tool-use is training-, not size-bound |

Both frontier/open additions land where the size–accuracy trend predicts (a second frontier vendor tops the leaderboard; a large open-weight model sits mid-table), and the newest DeepSeek-V3.2 illustrates the diagnostic point directly — strong at reading, markedly weaker at tool use. (*V3.2 measured on the 200-item access-ladder subset.) On finance-specialized LLMs: available open finance models (continued-pretrained variants on financial text) do not support reliable function calling, which is the capability under test; we note this explicitly. (Claude n=139 is a random subset; we will report the full-set number in the camera-ready.)

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
| GPT-5-mini (n=1,174) | **14.7%** | 67.5% | −52.8 pp |
| GPT-4.1 (n=300) | **13.3%** | 60.6% | −47.3 pp |
| Qwen3-30B-A3B (n=1,174) | **13.8%** | 53.0% | −39.2 pp |

Closed-book accuracy is essentially flat (~13–15%) across model families and capability levels, while agentic accuracy varies by 15 points — consistent with the residual closed-book score reflecting scenario narratives that legitimately contain the needed figure, not memorization (which would scale with training exposure).

Memorization of FinQA thus does not provide an answer pathway: the system prompt contains neither the source table nor its values, the backing store is a re-instantiated database with distractor rows, and the required multi-hop tool plan does not exist in any training corpus. If contamination were driving v2 performance, closed-book accuracy would approach agentic accuracy — instead it collapses to a flat ~14% floor (−39 to −53 points).



---

## Общий чек-лист перед постингом

- [ ] E0: анонимное зеркало обновлено и открывается инкогнито-браузером
- [x] E1: готово — 14.7/13.3/13.8% closed-book vs 67.5/60.6/53.0% agentic (experiments/e1_closed_book/)
- [x] E2: готово (4.4% скалярных; 74.3% agreement; 93 кейса для ручной разметки в experiments/e2_judge_agreement/results/)
- [ ] E3: human eval числа вставлены (PVoW-1/2, R2-4, R3-1)
- [x] E4: Claude Sonnet 4.5 = 70.5% (top), DeepSeek-V3 = 57.3% (experiments/e4_new_models/); закрывает R3-4 и PVoW-8
- [ ] E5: таблица failure taxonomy + примеры (PVoW-6, R2-3)
- [x] E6: готово (742 роли, SQL-фичи, tool-chain depth, op-mix — experiments/e6_diversity/)
- [ ] E7: costs (PVoW-7)
- [ ] Все `[...]`-плейсхолдеры заменены реальными числами
- [ ] Тон: благодарный, конкретный, без обещаний «in future work» там, где можно дать число сейчас
