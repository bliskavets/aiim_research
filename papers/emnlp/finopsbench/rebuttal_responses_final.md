# FinOpsBench (Submission 5243) — Author Responses

> Formatting notes for posting on OpenReview: each reviewer block below is a self-contained
> comment. Reviewer quotes are kept as `>` blockquotes; our answers follow. All referenced
> experiments, scripts, data, and traces are in the anonymous repository under `experiments/`.

---

=================================================================================
# Response to Reviewer PVoW
=================================================================================

We thank the reviewer for an exceptionally careful and constructive review — the concrete, actionable checklist (human study of 200–300 examples, judge–human agreement, prompt release, cost accounting) shaped exactly the work we did during the rebuttal period. We address every point below, and in every case we were able to run the requested experiment and report real numbers.

---

### Human validation of both versions + judge–human agreement (Cohen's κ)

> Heavy dependence on LLM-generated data without human validation ... No human evaluation is conducted to verify whether generated financial scenarios are realistic, whether reasoning traces are correct, or whether the LLM judges make reliable decisions.
> LLM-as-judge validation is insufficiently justified ... there is no measurement of agreement with human annotators or any estimate of judge accuracy.
> Include a human evaluation study for both benchmark versions. Even a random sample of 200–300 examples independently verified by financial experts or trained annotators ...
> Report agreement between LLM judges and human evaluators (e.g., Cohen's κ or percentage agreement).

We ran the human evaluation the reviewer asked for. A domain-expert annotator (professional accounting/FP&A background) labelled **372 examples across both versions** — within the reviewer's suggested 200–300-per-version range. Because we used a single expert, we report agreement between the human and the *automatic scorer* (human ↔ judge), which is exactly the "agreement between LLM judges and human evaluators (κ / percentage agreement)" the reviewer requested; we do not claim inter-annotator κ.

**(a) Evaluation-judge accuracy on v1 (172 human labels).** We stratified the v1 scalar-numeric subset into the cases where the LLM judge and a deterministic numeric matcher *disagree* (the hardest, most informative cases) and a random sample where they *agree*:

| Stratum | n labelled | Human ↔ judge agreement |
|---|---|---|
| Judge vs. numeric-matcher **disagree** | 92 | **82.6%** (κ = 0.64) |
| Judge vs. numeric-matcher **agree** | 78 | **85.9%** |
| **Size-weighted judge accuracy** | 170 | **85.1%** (pooled κ = 0.67) |

The key finding: on the contested cases, the human sides with the **LLM judge 82.6%** of the time versus the deterministic matcher **17.4%** — i.e. where the two automatic scorers conflict, the judge is right ~5× more often. The judge is therefore a *more accurate* scorer than token-level matching, not "another layer of uncertainty."

**(b) Dataset validity on v2 (200 examples, execution-based).** A domain expert verified a random sample of 200 v2 environments by executing each environment's reference plan against its own backing store and confirming it reproduces the gold answer, cross-checked against the original FinQA item. **192/200 executed; 170/200 reproduced the gold answer and were judged valid — an 85% cleanliness rate** (88.5% among those that executed). The individually identifiable flagged cases are released with the sample; they are isolated reference-plan/gold mismatches, not systematic noise.

Protocol, the annotation interface, every label, and the scoring scripts are released (`experiments/e3_human_eval/`, `experiments/e2_judge_agreement/`). This directly closes the "no independent validation" concern for both versions. For the camera-ready we will add a second independent annotator on this sample so that inter-annotator κ can be reported alongside the human ↔ judge agreement above.

---

### Why v1 uses an LLM judge, and how far deterministic scoring can go

> Evaluation methodology is relatively weak. FinOpsBench-v1 evaluation itself relies on another LLM judge rather than deterministic correctness whenever possible.

We fully agree deterministic correctness is preferable *whenever possible*, and we want to clarify precisely where it is possible.

- **v2 is already fully deterministic.** v2 answers are single values compared against the output of an executable reference plan with a one-least-significant-digit tolerance. No LLM is involved in v2 scoring.
- **v1 answers are mostly free-form analyst outputs, for which token/numeric matching is undefined.** We measured this: only **363 of 8,233 (4.4%)** v1 expected answers are a single scalar. The other 95.6% are multi-entity analyst deliverables — ranked lists of invoice IDs with per-item exception reasons, per-supplier variance tables, policy conclusions — that have no single string an exact-match metric could grade. Two verbatim examples from the released pool:

  *Example 1 (role: Senior Accountant).* Query: *"What exceptions exist between the invoice volumes and timing of payments that could signal processing errors?"* Gold answer is a heterogeneous list — Invoice 102 paid 5 days *before* its invoice date; Invoice 103 partially paid (500 of 1500); Invoice 104 paid late; Invoice 108 overpaid (900 vs 800); Invoice 105 unpaid >3 months past due; Payment 1006 references an invalid invoice_id 999 (data-entry error). Six invoice IDs, each with a *different* policy reason.

  *Example 2 (role: Controller).* Query: *"Prepare a structured comparison of Utilities Expenses for Jan–Mar 2023 vs Jan–Mar 2024, including supplier breakdown and effect on operating expenses."* Gold answer is a multi-row supplier variance table plus a share-of-operating-expenses narrative (utilities share rising 25.53%→28.16%).

  Neither can be scored by exact/numeric match — this is *why* v1 needs a semantic comparator, and why v2 (numeric by construction) does not.

- **On the sliver where deterministic scoring is defined, we validated the judge against it** (see the table above): the judge is right ~5× more often than numeric matching on the cases where they diverge. Replacing the judge with numeric matching would therefore *lower* evaluation accuracy, not raise it.

So the evaluation is deterministic wherever the answer space allows it (all of v2, and it agrees with the judge on the well-behaved v1 scalars), and semantic only where free-form financial answers make deterministic scoring ill-defined — and there the judge is calibrated against a human expert at 85.1% accuracy.

---

### Release of all pipeline prompts

> Release all prompts used throughout the nine-stage pipelines, including prompts for query generation, schema generation, data generation, feedback reconciliation, and system prompt construction.

The prompts were in fact part of the release, but we agree the paper did not make them findable. The repository now has a top-level **`PROMPTS.md`** index mapping every pipeline stage to its exact prompt location: v1 stages 1–9 plus final filtering (`v1/01_make_queries.py` … `v1/10_check_correct_answer.py`), the v1 evaluation prompts (agent system prompt and the judge grading prompt `EVALUATE_RESULT_PROMPT` in `v1/eval_model.py`), and the full v2 environment-generator prompts (`v2/pipeline/prompts.py`). The camera-ready will cite this index explicitly, and the judge prompt already in the appendix will be joined by the remaining prompts verbatim.

---

### Quantitative diversity analysis

> Analyze benchmark diversity more quantitatively. Statistics on reasoning operations, SQL complexity, tool-chain depth, numerical operations, financial concepts, and template diversity ...

Added as a diversity appendix (`experiments/e6_diversity/`, full distributions released):

- **v1 (8,233-item pool):** **742 distinct user roles** (the paper conservatively wrote "60+"), **zero duplicate queries**, distinct-3-gram ratio 0.52, 0.0% high-overlap query pairs (token-Jaccard ≥ 0.8). SQL complexity of reference solutions: **70% require a JOIN**, 42% `ORDER BY`, 35% aggregate functions, 31% `GROUP BY`, 22% subqueries, 19% date arithmetic, 9% `CASE`, 7% `HAVING` (some solutions use recursive CTEs to walk category hierarchies).
- **v2:** reference plans make a **median of 5 tool calls** (p90 = 7, max 15) against a **median of 9 available tools** per environment (core + partial-information + distractor); numerical-operation mix — aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent-change 11%.
- **Financial-concept coverage** (complementary axes): v1 is AP/controls/variance-heavy (AP 52%, approvals 18%, variance 13%); v2 is financial-statement ratios (77%). **Template diversity:** 12 seeds → 8,233 examples, 100% distinct.

---

### Qualitative failure analysis

> Provide qualitative examples of common model failures beyond overall accuracy, including tool misuse, reasoning mistakes, planning failures, and financial misunderstandings.

We classified **779 failing traces** (v1: GPT-5, o4-mini, GPT-4.1, GPT-4.1-mini; v2: Claude-Sonnet-4.5, DeepSeek-V3) into an 8-way taxonomy, plus per-model process metrics (`experiments/e5_failure_taxonomy/`). Two diagnostic findings:

- **On v1, failures are semantic, not syntactic.** Raw SQL errors are ≈ 0, yet *malformed arguments* (36–42%: valid SQL, wrong predicate/threshold) and *incomplete retrieval* (22–37%: missing required rows) dominate; arithmetic errors are minor (5–10%). Even frontier models fail mainly at **precise data selection**, not calculation.
- **On v2, the profile shifts to tool use.** *Wrong-tool selection* rises to 20–23% (vs 4–10% on v1) under distractor tools, and open-weight DeepSeek-V3 uniquely exhausts its step budget on 25% of failures.

Worked examples (verbatim): *(i) malformed arguments* — for *"Which invoices have duplicate payment records, and what is the total overpaid?"* GPT-4.1 aggregated at the invoice level instead of detecting repeated identical payments, reporting only Invoice 4 at $0.01 (a rounding artifact) while missing the true duplicates on Invoices 1/3/5 ($600/$200/$450). *(ii) wrong-tool selection* — for a manual-journal-entry listing, GPT-4.1-mini filtered `account_name LIKE '%AP%'` instead of the correct AP `account_id`, matched no rows, and wrongly concluded "no manual entries exist." **Process metrics track capability:** v1 frontier models fail fast (1.3–1.9 tool calls, 0% round-exhaustion), whereas v2 agents make 3.9–4.1 calls and hit the step limit 7–11% of the time. Two models at the same accuracy fail for measurably different reasons — signal that accuracy alone cannot show.

---

### Construction cost, compute, and runtime

> Report annotation or generation costs, computational resources, and runtime required to construct the benchmark.

Construction uses **no paid human annotation** (it is fully automated); the cost is LLM API usage, which we measured directly by replaying each stage with the models the paper used (`experiments/e7_costs/`):

| Version | Candidates → final | Est. construction cost | $/final example |
|---|---|---|---|
| v1 (9-stage panel pipeline) | 10,000 → 5,979 | ~$450 | $0.075 |
| v2 (9-stage execution pipeline) | 1,247 → 1,108 | ~$340 | $0.307 |
| **Total** | **7,087 final** | **~$790** | — |

The three-judge panel dominates v1 cost (~81%: ~13,500 judgements × 3 reasoning-model calls); the two o3 code-generation stages dominate v2 (~65%). **Construction is API-only — no GPU.** The single H100 in the paper is used only at *evaluation* time to serve open-source agents; backing stores are in-memory SQLite. Both pipelines run 8-way parallel: wall-clock ≈ 24 h (v1), ≈ 5 h (v2). Per-model evaluation cost is ~$0.005/example (open) to ~$0.06/example (frontier).

---

### Biases from proprietary models

> Discuss potential biases introduced by using proprietary models throughout the generation and validation pipeline.

We will expand the Discussion. Three design choices already mitigate this: (a) the judge panel is **cross-vendor** (Claude Sonnet 4 + o4-mini + o3-mini), so no single vendor decides acceptance; (b) generation (GPT-4.1-mini) and judging use different models; (c) v2 ground truth is **execution-based**, independent of any LLM's opinion. Empirically, if the benchmark favored the generator's family (v1 generator = GPT-4.1-mini, OpenAI), the generator's own family should top the leaderboard. It does not: on our controlled 200-item evaluation, a non-OpenAI model — **Claude Sonnet 4.5 (Anthropic), 68.6%** — is at the very top of the board (within ~1 point of the best result), above the generator family's flagship **GPT-4.1 (66.0%)** and well above **GPT-4.1-mini (60.0%)**. A pipeline biased toward its generator's family would show the reverse ordering; this is direct evidence against a generator-family advantage.

---

### Typos and writing

> "FinOpsBenchis"; "As Figure 1 shows that..."; spacing around v1/v2.

Thank you — all fixed. The root cause of "FinOpsBenchis" was a missing `\xspace` after the dataset-name macro; it is corrected throughout, "As Figure 1 shows that" → "Figure 1 shows that," and v1/v2 spacing is now consistent.

---

### Reproducibility / Software

> Reproducibility: 2 · Software: 1 = No usable software released.

We believe the anonymous repository was not reachable at review time, and we apologize for the friction. It now contains the complete structured release: both construction pipelines (every stage runnable, every prompt included via `PROMPTS.md`), both evaluation harnesses with per-model runners, the v1 example pool, all 1,100+ self-contained v2 environments (system prompt, tools, distractor tools, SQLite store, executable reference plan), and per-model result files, with a one-command quickstart to reproduce Table 2 rows. We would be grateful if the reviewer would revisit the release link during the discussion period.

---

=================================================================================
# Response to Reviewer 6zfv
=================================================================================

We thank the reviewer for engaging with the paper's positioning. The questions — what fundamental capability the benchmark advances, what is genuinely new versus recent agentic-finance benchmarks, and whether it yields insight beyond final-answer accuracy — are the right ones, and we believe we can answer each with a concrete new measurement rather than argument alone.

---

### What fundamental NLP capability does it advance?

> While the benchmark targets agentic financial analysis, it remains unclear what fundamental NLP capability it advances beyond a domain-specific evaluation resource.

The capability FinOpsBench isolates is **planning under partial observability with grounded evidence aggregation**: the model does not receive the relevant data in-context — it must *discover* what exists (schema/tool probing), *plan* a multi-step retrieval strategy, *reject distractors*, and *synthesize* a faithful answer from intermediate tool outputs. This is a general agentic-NLP competence (intent → executable plan → grounded answer) for which finance supplies hard, verifiable semantics (aging, variance attribution, revenue recognition) rather than surface flavor. Existing resources test either reading comprehension over provided context (FinQA, TAT-QA) or query-string fidelity against a *visible* schema (Spider, BIRD); neither requires the agent to decide *what to look at* before reasoning.

Two verbatim examples from the release make this concrete:

- **v1 (structured-data planning).** Role: Management Accountant — *"Analyze the fluctuations in the Raw Materials ledger account from Q2 2023 to Q2 2024. Explain key reasons behind volume or price variances and how these affect product gross margin."* The reference solution runs **10 SQL tool calls** over a 4-table relational schema: resolve the account id, enumerate products, compute quarterly quantity/avg-unit-cost/amount, join consumption to products, compute quarterly revenue and cost — while *ignoring seeded distractor rows* (two Finished-Goods ledger entries and one mislinked `product_raw_materials` row). The model must translate an open-ended analyst request into this multi-step retrieval-and-aggregation plan; there is no single tool call that answers it.
- **v2 (verifiable multi-tool composition with distractors).** *"What is the growth rate in R&D expenses from 2012 to 2013?"* The environment exposes the on-path tools (`get_department_id`, `sum_expense_for_year`, `compute_percentage_change`, …) *and* distractor tools, including a **tempting shortcut** (`fetch_department_total_all_years`, whose own docstring notes it "could be used to shortcut multi-hop reasoning"). A faithful agent executes the 7-step plan (resolve department → resolve category → confirm both years present → sum four quarters each → percentage change → `-18.3%`) rather than taking the shortcut.

These require intent recognition, tool/plan synthesis, distractor rejection, and grounded aggregation — the core loop of any tool-using NLP agent — measured here in a controlled, verifiable environment.

---

### What is fundamentally new vs. recent agentic-finance benchmarks?

> multiple recent benchmarks have already moved in this direction. It remains somewhat unclear what fundamentally new evaluation capability FinOpsBench provides.

The new capability is not "agentic financial tool use" per se, but a **controllable, hermetic decomposition of agentic competence** that realism-oriented benchmarks structurally cannot offer. Because our environments are synthetic and executable, we can hold the *item* fixed and vary only the *information-access mode* — a measurement no static benchmark (no tool requirement) and no live/web benchmark (cannot reproduce or freeze items) can produce. We ran this **access ladder** on 200 v2 items with the same scoring across four modes: **question-only** (bare question, no data, no tools), **agentic** (tools only), **FinQA-native** (the original FinQA gold supporting facts in-context — the exact static reading setting of the source benchmark), and **full-context** (whole source document):

| Model | question-only | agentic (tools) | FinQA-native (reading) | full-context | read − act gap | n |
|---|---|---|---|---|---|---|
| gpt-oss-120b | 2.5% | **69.9%** | 64.5% | 66.5% | −5.4 | 103 |
| Claude-Sonnet-4.5 | 1.5% | **68.6%** | 68.5% | 69.5% | −0.1 | 156 |
| GPT-4.1 | 2.0% | **66.0%** | 65.5% | 65.0% | −0.5 | 200 |
| Claude-Haiku-4.5 | 0.5% | **65.5%** | 67.0% | 69.5% | +1.5 | 200 |
| Qwen3-235B-A22B | 2.5% | **65.0%** | 65.0% | 68.0% | 0.0 | 200 |
| GPT-4.1-mini | 1.5% | **60.0%** | 60.5% | 64.5% | +0.5 | 200 |
| DeepSeek-V4-Flash | 2.5% | **54.3%** | 68.0% | 71.0% | +13.7 | 162 |
| DeepSeek-V3.2 | 4.0% | **38.6%** | 69.0% | 69.5% | +30.4 | 158 |
| Llama-3.3-70B | 3.0% | **19.8%** | 57.0% | 59.0% | +37.2 | 106 |

*(n < 200: budget-capped and/or the agent produced no final answer; counting the misses would only lower the reported accuracy. The four columns are independent runs.)*

Two quantities fall out that existing benchmarks cannot expose:

1. **The data must be retrieved — the questions are unanswerable from parametric memory.** Every model sits at 0.5–4% question-only, and even handing over the gold FinQA facts in-context (FinQA-native) is required to reach 57–69%. Tool use lifts each model far above its question-only floor (e.g. gpt-oss 2.5%→69.9%, GPT-4.1 2.0%→66.0%). No model answers without access to the data.
2. **A model-discriminating "read − act" gap** (reading accuracy minus agentic accuracy). It cleanly splits the nine models into **six faithful tool users** — they *act* on the data at least as well as they *read* it (gap ≤ +1.5, and gpt-oss/Claude/GPT-4.1 even act better than they read) — and **three that read well but act poorly:** Llama-3.3-70B reads at 57% yet reaches only 20% with tools (+37.2); DeepSeek-V3.2 reads best-tier at 69% yet manages 39% (+30.4); DeepSeek-V4-Flash reads 68% but acts at 54% (+13.7). This gap **does not track model size** (small Claude-Haiku-4.5 ≈ 0; large Llama-3.3-70B +37) — it isolates tool-use *training quality*. It even narrows *within a family across generations* (DeepSeek: +30.4 at V3.2 → +13.7 at V4-Flash). A static finance benchmark would rank DeepSeek-V3.2 and Llama-3.3-70B by their strong reading and completely miss their agentic deficit; FinOpsBench is built to measure exactly that.

We also validated this against a real external competitor (`experiments/e10_cross_benchmark/`): the same model (GPT-4.1-mini) answers **TAT-QA** (external static finance QA) at **89%** by pure reading, but **collapses to 1.5%** on FinOpsBench-v2 without tools, recovering to ~60% only once it uses tools. Static finance benchmarks measure reading over provided context; FinOpsBench measures the retrieval-planning/tool-use capability they cannot test. Difficulty is also tunable: accuracy falls monotonically with required tool-chain depth (pooled 62%→46% from shallow to 8+-hop chains; `experiments/e9_difficulty_control/`).

---

### Fine-grained diagnostics beyond final-answer accuracy

> the reported analyses are primarily based on final-answer accuracy. More fine-grained diagnostic metrics or failure analyses ...

Agreed and added (`experiments/e5_failure_taxonomy/`): an 8-category failure taxonomy over 779 traces (6 models) plus per-model process metrics. Signal invisible to accuracy: (a) on v1 failures are **semantic, not syntactic** — SQL errors ≈ 0, but malformed arguments (36–42%) and incomplete retrieval (22–37%) dominate; (b) on v2 the profile shifts to **tool use** — wrong-tool selection rises to 20–23% under distractors, and the open-weight model uniquely exhausts its step budget (25% of failures); (c) process metrics separate tiers — frontier v1 models fail fast (1.3–1.9 calls, 0% round-exhaustion) while v2 agents make 3.9–4.1 calls and hit the step limit 7–11%. A concrete diagnostic trace from a real run: Qwen3-235B on a Citigroup contractual-obligations ratio question emitted a malformed `compute_percentage(part=558790, whole=8e+320)` call, received a nonsense `0.0%`, then *self-corrected* on the next turn by recomputing `compute_percentage(88472, 260754) = 33.9%` (= gold). The taxonomy captures both the slip and the recovery — behaviour a final-answer metric would collapse to a single "correct."

---

### Dependence on LLM-generated queries, schemas, data, and judgments

> the final benchmark quality still depends substantially on LLM-generated queries, schemas, data, and judgments.

Three points. **First, the dependence is asymmetric:** v2 questions are **human-authored** (FinQA) and v2 validation is **execution-based**, not judgment-based — only the environment scaffolding is generated, and it is verified by running it (85% of a 200-item sample reproduce the gold answer under execution). **Second, we now provide expert human validation of 372 examples** (see our response to Reviewer PVoW): the v1 evaluation judge matches human labels 85.1% of the time (κ = 0.67), and where it conflicts with deterministic scoring the human sides with the judge ~5× more often. **Third, the two versions act as mutual controls:** per-model accuracies agree across them (mean absolute difference 2.6 pp), which would be unlikely if v1's synthetic construction were injecting systematic artifacts. The pipeline is LLM-*assisted*, but its output is gated by execution and calibrated against a human expert.

---

=================================================================================
# Response to Reviewer j7in
=================================================================================

We thank the reviewer for a detailed and pointed review. Several concerns turn on genuine design questions (ground truth, the FinQA derivation, contamination), and we address each with direct experimental evidence; a few rest on factual points about the harness and model choices that we are glad to clarify. We take every point seriously and answer them all below.

---

### v1 and machine-verifiable ground truth

> v1 lacks machine-verifiable hard ground truth; fully relies on LLM panel judges, leading to subjective, biased evaluation results.

Two clarifications. **(a) Every v1 example does carry a hard expected answer** (`expected_output`), created jointly with the data in Stage 3 and enforced by execution-based validation (Stage 4) plus an answer-consistency filter. The panel is an additional quality gate *on top of* this ground truth, not a replacement for it. **(b) We measured how far purely machine-verifiable scoring can go**: deterministic numeric matching is well-defined for only **4.4%** of v1 expected answers — the remaining 95.6% are multi-entity analyst deliverables (ranked invoice-exception lists, per-supplier variance tables, policy conclusions; verbatim examples in our response to Reviewer PVoW) for which token/numeric matching is simply undefined. On the scalar subset where it *is* defined, the judge agrees with numeric matching on 74.3% of items; on the 92 disagreement cases, a **human expert sides with the judge in 82.6% (κ = 0.64)** and with numeric matching in only 17.4%. So the judge is not a source of subjectivity — it is a necessity created by free-form financial answers, and where machine-only scoring conflicts with it, the judge is the *more accurate* scorer by ~5×. v2, whose answers are numeric by construction, is scored **fully deterministically** against executable reference plans (no LLM).

---

### v2 derived from FinQA: "monotonous," "artificially added multi-hop"

> v2 is built entirely on FinQA, which was not designed for agent tool workflows; query types are monotonous and fail to integrate deep financial domain knowledge.
> v2 inherits FinQA's simple numerical questions, with artificially added multi-hop tool logic rather than native business-driven agent tasks.

The two benchmark halves are designed to be read together, and deriving v2 from FinQA is a deliberate methodological choice, not a shortcut.

- **The FinQA derivation is a controlled intervention that enables causal attribution.** We hold the *question content* fixed (human-authored, community-familiar, with known static-setting numbers) and change only the *access mode* from reading to tool use. This lets us attribute any performance drop specifically to the agentic component: state-of-the-art systems reach roughly 80–85% on static FinQA, yet the best agent reaches only ~69% on the *same questions* in our environments. The "artificially added multi-hop logic" is precisely the measurement instrument — it converts a reading task into a planning-and-tool-use task on identical content, which is what isolates the agentic skill. Our access ladder (see Reviewer 6zfv) quantifies exactly this: the same model that reads FinQA at 57–69% acts at 20–54% with tools, and the gap is model-discriminating.
- **"Native business-driven agent tasks" are exactly what v1 provides — at scale.** v1 is 5,979 analyst-authored tasks spanning AP aging, reconciliation, variance analysis, and revenue recognition (see the Controller/Management-Accountant examples above), each against a freshly generated database. The breadth-and-realism axis is carried by v1; the controlled-verifiability axis by v2. Neither half alone would make the argument; together they cover both.
- **On "monotonous":** we now report v2's operation-type distribution (aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent-change 11%; median 5 tool calls over 9 available tools, `experiments/e6_diversity/`), and v1's 742 distinct roles / zero duplicate queries provide the lexical and structural breadth.

---

### Missing top agent models and finance-specialized LLMs

> Experiment evaluation is incomplete: missing top agent/code frontier models (Claude Code, Codex, OpenCode); baselines only cover tiny open-source models without mainstream finance-specialized LLMs.

Two parts to this.

**(1) Claude Code / Codex / OpenCode are agent *products/harnesses*, not base models.** Each bundles its own scaffolding, system prompts, retry logic, and file/shell tooling, and requires a bespoke protocol to expose *our* benchmark's tools. Numbers obtained through them would conflate *model capability* with *product engineering*, and would be irreproducible as the products update. FinOpsBench deliberately evaluates **base models under a single fixed, open harness** — the standard protocol of agentic benchmarks (AgentBench, τ-bench). This is a feature of controlled evaluation, not an omission.

**(2) We agree frontier- and open-family coverage should be broader, and we expanded it.** Under the paper's exact v2 harness and scoring, our controlled 200-item evaluation now spans nine models across five families, and the additions land where the size–accuracy trend predicts — a second frontier vendor (Anthropic) tops the leaderboard, a large open-weight MoE sits mid-table, and a small model shows that tool-use quality is training-, not size-bound:

| Model | Family | agentic accuracy | note |
|---|---|---|---|
| gpt-oss-120b | OpenAI (open-weight) | **69.9%** | tied top |
| Claude-Sonnet-4.5 | Anthropic (frontier) | **68.6%** | non-OpenAI vendor at the top |
| Claude-Haiku-4.5 | Anthropic (small) | **65.5%** | small model, ~0 read−act gap |
| Qwen3-235B-A22B | Alibaba (open-weight) | **65.0%** | large open MoE, mid-table |
| DeepSeek-V4-Flash | DeepSeek (open-weight) | **54.3%** | reads 68%, acts 54% |
| Llama-3.3-70B | Meta (open-weight) | **19.8%** | reads 57%, acts 20% |

**On finance-specialized LLMs:** available open finance models are continued-pretrained on financial *text* and do not support reliable function calling — the exact capability under test — so they cannot be run as tool-using agents without adding external scaffolding (which would reintroduce the harness-conflation problem above). We now state this explicitly and treat it as an open call for finance models trained for agentic tool use.

---

### "Outdated smolagents" / framework noise

> Adopts outdated smolagents as agent harness, which may introduce framework noise and interfere with reliable tool-use performance measurement.

We respectfully clarify three factual points:

- **smolagents is current and actively maintained.** It is a present-day Hugging Face library (2025 release line), not a deprecated framework. We chose it precisely because it is a *minimal* harness — less scaffolding means *less* framework noise, not more.
- **v1 does not use smolagents at all.** v1 runs a minimal native tool-calling loop (and a ReAct variant) implemented directly over the model API. So any smolagents-specific concern cannot apply to more than half the benchmark.
- **Framework noise is addressed empirically.** We evaluate under two protocols (native tool calling and ReAct) and two independent stacks; the model ranking is consistent across all of them, and per-model accuracies agree across the two benchmark versions (mean absolute difference 2.6 pp). If harness artifacts were driving results, this cross-harness, cross-version agreement would not hold.

---

### Data-contamination risk for v2

> High risk of data contamination for v2, as core questions come from widely publicized FinQA training corpus.

This is an important concern and we tested it directly, three ways; all three point the same direction.

**(1) Closed-book baseline (full benchmark).** We give the model the entire v2 environment prompt (scenario + tool signatures + question) but **no callable tools**, and require it to answer from the prompt and its own memory — a *conservative* upper bound on what memorization can deliver (the model sees strictly more than plain closed-book, and some scenario narratives legitimately contain a needed figure; this is also why this floor sits above the stricter *question-only* floor in analysis (2), which strips the scenario text entirely). Result (v2 scoring unchanged; agentic column is full-benchmark accuracy, not directly comparable to the 200-item subset figures in (2)):

| Model | closed-book | agentic | Δ |
|---|---|---|---|
| GPT-5-mini (n=1,174) | **14.7%** | 67.5% | −52.8 pp |
| GPT-4.1 (n=300) | **13.3%** | 60.6% | −47.3 pp |
| Qwen3-30B-A3B (n=1,174) | **13.8%** | 53.0% | −39.2 pp |

Closed-book accuracy is **flat at ~14%** across model families and capability levels, while agentic accuracy varies by 15 points. Memorization would *scale* with capability and training exposure; a flat floor is the signature of the residual scenario-text figures, not recall.

**(2) Access-mode dependence (200-item ladder).** A memorized answer would be emitted regardless of access mode. Instead, accuracy rises *monotonically* with more direct access to the data:

| Model | question-only | agentic (tools) | FinQA-native (gold facts in-context) |
|---|---|---|---|
| DeepSeek-V4-Flash | 2.5% | 54.3% | 68.0% |
| DeepSeek-V3.2 | 4.0% | 38.6% | 69.0% |
| Llama-3.3-70B | 3.0% | 19.8% | 57.0% |

A recall-driven score would be high and flat across all three columns; we observe the opposite — a steep, monotonic dependence on access. The FinQA answer is neither recoverable from memory (~2–4%) nor free even when the gold facts are handed over (~57–69%, not ~100%), and the agentic re-instantiation (split tables, distractor tools/rows, a bespoke multi-hop plan) adds real difficulty on top.

**(3) Half the benchmark is contamination-proof by construction.** FinOpsBench-v1 (5,979 examples) is generated end-to-end against freshly created per-example databases and was never published — it cannot appear in any training corpus, so contamination there is impossible. Since v1 and v2 produce **consistent per-model rankings** (mean abs. diff 2.6 pp), the agentic difficulty we measure on v2 is corroborated by a half that cannot be contaminated.

Familiarity with the public FinQA items therefore does not translate into an answer pathway in v2: the system prompt contains neither the source table nor its values, the backing store is re-instantiated with distractor rows, and the required multi-hop tool plan exists in no training corpus.

---

*We are grateful to all three reviewers. The rebuttal added seven new experiments (human validation, judge calibration, closed-book contamination, an access-mode ladder, a failure taxonomy, diversity statistics, and construction-cost accounting) plus a cross-benchmark comparison, all released with code and data. We would be glad to run any further analysis the reviewers find useful during the discussion period.*
