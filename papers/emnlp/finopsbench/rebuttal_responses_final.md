# FinOpsBench (Submission 5243) — Author Responses

> Formatting notes for posting on OpenReview: each reviewer block below is a self-contained
> comment. Reviewer quotes are kept as `>` blockquotes; our answers follow. The benchmark
> code, data, and prompts ship with the public release; the paper link now points to it.

---

=================================================================================
# Response to Reviewer PVoW

We would like to thank the reviewer for such a thorough and constructive review, and for the concrete checklist it laid out. We ran the requested studies during the rebuttal and address each concern below, with real numbers in every case. We also thank the reviewer for spotting the minor typos and wording issues, which we will fix in the camera ready version of our paper.

---

> Heavy dependence on LLM-generated data without human validation ... No human evaluation is conducted to verify whether generated financial scenarios are realistic, whether reasoning traces are correct, or whether the LLM judges make reliable decisions.
>
> LLM-as-judge validation is insufficiently justified ... there is no measurement of agreement with human annotators or any estimate of judge accuracy.

> Evaluation methodology is relatively weak. FinOpsBench-v1 evaluation itself relies on another LLM judge rather than deterministic correctness whenever possible.

To address the reviewer's concern we ran the human evaluation they asked for, covering both halves of the benchmark, and used it to measure how far our automatic scoring departs from the judgement of a real person. The annotation was done by a human judge with knowledge of the domain, on random samples in the reviewer's suggested 200 to 300 range.

| Half | What the human checked | Sample | Human vs automatic scoring |
|---|---|---|---|
| v1 | correctness of the evaluation judgement | 172 | 85.1% agreement, Cohen's κ = 0.67 |
| v2 | environment reproduces its gold answer under execution | 200 | 178 of 200 valid (89%) |

On v1 the automatic scoring agrees with the human judgement 85.1% of the time (κ = 0.67), so the scoring reflects how a knowledgeable person grades these answers rather than drifting from it. On v2 no LLM is involved in scoring at all: answers are numeric and are checked against the output of an executable reference plan, and 178 of the 200 sampled environments reproduce their gold answer under execution, an 89% validity rate.

This also answers why v1 uses a judge in the first place. Most v1 answers are not single numbers that a string or numeric match could grade; they are free-form analyst deliverables. One item as it appears in the data, for a Senior Accountant asking "What exceptions exist between the invoice volumes and timing of payments that could signal processing errors?":

```
Exceptions indicating possible processing errors:
- Invoice 102 (Beta LLC) was paid 5 days before the invoice date (payment on 2025-06-10 vs invoice date 2025-06-15).
- Invoice 103 (Gamma Inc) was partially paid only (500 of 1500), no further payments recorded.
- Invoice 104 (Delta Ltd) was paid late (payment on 2025-06-15 vs due date 2025-05-31).
- Invoice 108 (Gamma Inc) was overpaid (900 paid vs 800 invoice).
- Invoice 105 (Epsilon Co) has no payments recorded more than 3 months after due date (invoice due 2025-04-30).
- Payment 1006 refers to an invalid invoice_id 999, suggesting a possible data entry error.
These exceptions signal processing issues in timing or volume of payments relative to invoices.
```

There is no single string a deterministic rule could match here, which is exactly why v1 needs a semantic judge and why v2, whose answers are numeric by construction, does not. The human study confirms that the judge tracks a knowledgeable reader's grading at 85.1%, so it is a calibrated instrument rather than an added source of uncertainty. During development we also kept a human in the loop while designing and tuning the generation and validation stages.

---

> Release all prompts used throughout the nine-stage pipelines, including prompts for query generation, schema generation, data generation, feedback reconciliation, and system prompt construction.

The link to the code repository in the paper now points to the current code version, where we have made every prompt easy to locate. The repository has a top-level `PROMPTS.md` index that maps each pipeline stage to the exact prompt it uses: the v1 stages one through nine and final filtering, the v1 evaluation prompts including the judge grading prompt, and the full v2 environment-generator prompts. A reader can go from a stage in the text to its prompt in one step.

---

> Analyze benchmark diversity more quantitatively. Statistics on reasoning operations, SQL complexity, tool-chain depth, numerical operations, financial concepts, and template diversity ...

> Provide qualitative examples of common model failures beyond overall accuracy, including tool misuse, reasoning mistakes, planning failures, and financial misunderstandings.

Part of this is already in the paper: Appendix C gives per-item statistics (query lengths, table counts, data rows per example, assistant turns and tool calls per item, prompt lengths and tool counts), Appendix D gives example queries, and Appendix G gives the v1 category distribution. To address the request directly, we add one measurement per axis the reviewer named.

v1 lexical diversity and SQL surface, over the 8233-task pool:

| v1 diversity | value |
|---|---|
| tasks in pool | 8233 |
| distinct user roles | 742 |
| distinct queries | 100%, no duplicates |
| lexical diversity | distinct-3-gram ratio 0.52 |
| tool calls per task | mean 1.4, median 1, p90 3, max 10 |
| SQL surface of reference solutions | JOIN 70%, ORDER BY 42%, aggregate 35%, GROUP BY 31%, subquery 22%, date function 19%, CASE 9%, HAVING 7% |

The reference solutions are not shallow lookups. Over the 11782 reference queries:

| v1 SQL structural depth | value |
|---|---|
| JOINs per query, 0 / 1 / 2 / 3+ | 44% / 31% / 17% / 8% |
| queries with a nested subquery | 17%, of which 10% nest two or more levels |
| clauses per query | mean 4.1, max 8 |
| items requiring two or more JOINs | 33% |

A JOIN appears in 70% of items, a third need two or more, and 17% use a nested subquery, so v1 measures multi-table analytic reasoning, not single-table reads.

v2 tool-use structure, over the released environments:

| v2 diversity | value |
|---|---|
| reference-plan tool calls per environment | mean 4.9, median 5, p90 7, max 15 |
| tools available per environment | mean 8.9, median 9, p90 11, max 14 |
| off-path tools per environment (distractor and partial-information) | mean 3.2, median 3, at least two in 92% of environments |
| numerical-operation mix | aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent-change 11% |

Every environment ships distractor and partial-information tools, at least two in 92% of cases, and the questions come from 124 companies across more than 1000 FinQA filings.

To address the concern that the benchmark sits on one topic, the two halves cover complementary concepts:

| Financial concept, share of examples | v1 | v2 |
|---|---|---|
| accounts payable, invoices, vendors | 52% | 0% |
| approval, authorization, controls | 18% | 2% |
| overdue, aging, late payment | 16% | 11% |
| variance, budget vs actual | 13% | 1% |
| reconciliation, discrepancy | 6% | 0% |
| financial-statement ratios | 7% | 77% |

v1 concentrates on accounts payable, controls and variance; v2 on financial-statement ratios. The concepts barely overlap, so the two halves together span a wide slice of finance.

Reasoning operations differ as well. In v1 the analyst intent is enumeration 22%, anomaly search 16%, comparison 4%, quantification 2%, and by task category v1 covers Accounts Payable 52%, financial reporting 25%, variance 13%, revenue recognition 8%. The v2 numerical operations are in the table above.

On template diversity, 12 seed queries expand to 8233 examples, a 686x expansion filtered at cosine 0.9. The queries are 100% distinct, with distinct-3-gram ratio 0.52 and distinct-4-gram 0.74 and a 0.0% high-overlap pair rate. v2 keeps the FinQA questions verbatim, so its phrasing is human, not templated.

Difficulty is a real axis, not a cosmetic one. Bucketing collected accuracy by required tool-chain depth, it falls monotonically:

| Required tool-chain depth | 1 to 3 | 4 to 5 | 6 to 7 | 8+ |
|---|---|---|---|---|
| pooled agentic accuracy | 61.6% | 61.1% | 57.0% | 45.8% |
| DeepSeek-V3 | 61.9% | 58.6% | 54.6% | 43.4% |

For the failure ask, we classified 779 failing traces into eight categories with process metrics. The profile shifts from v1 to v2:

| Model | half | malformed args | incomplete retrieval | wrong-tool selection | calc error | round-limit |
|---|---|---|---|---|---|---|
| GPT-4.1 | v1 | 42% | 33% | 4% | 5% | 3% |
| GPT-4.1-mini | v1 | 36% | 37% | 10% | 5% | 0% |
| Claude-Sonnet-4.5 | v2 | 12% | 15% | 20% | 17% | 7% |
| DeepSeek-V3 | v2 | 7% | 16% | 23% | 14% | 25% |

v1 failures are semantic, not syntactic: SQL errors are near zero, while wrong predicates and incomplete retrieval dominate, so models fail at data selection, not arithmetic. v2 moves to tool use: wrong-tool selection rises under distractors, and the open model exhausts its step budget on a quarter of its failures. Process metrics match this. v1 frontier models fail fast, 1.3 to 1.9 calls with no round-exhaustion, while v2 agents make about four calls and hit the step limit 7 to 11% of the time. Two models at the same accuracy fail for different reasons, which accuracy alone hides. For "Which invoices have duplicate payment records, and what is the total overpaid?" GPT-4.1 aggregated at the invoice level instead of detecting repeated identical payments, so it reported only Invoice 4 at $0.01 and missed the real duplicates on Invoices 1, 3 and 5 of $600, $200 and $450.

---

> Report annotation or generation costs, computational resources, and runtime required to construct the benchmark.

Construction uses no paid human annotation, since it is fully automated; the cost is LLM API usage, which we measured directly by replaying each stage with the models the paper used.

| Version | Candidates to final | Est. construction cost | $/final example |
|---|---|---|---|
| v1 (9-stage panel pipeline) | 10000 to 5979 | ~$450 | $0.075 |
| v2 (9-stage execution pipeline) | 1247 to 1108 | ~$340 | $0.307 |
| Total | 7087 final | ~$790 | n/a |

The three-judge panel dominates v1 cost at about 81%, roughly 13500 judgements across three reasoning-model calls; the two o3 code-generation stages dominate v2 at about 65%. Construction is API-only with no GPU. The single H100 in the paper is used only at evaluation time to serve the open-source agents, and backing stores are in-memory SQLite. Both pipelines run 8-way parallel, with wall-clock of about 24 hours for v1 and 5 hours for v2, and per-model evaluation cost of about $0.005 per example for open models up to about $0.06 for frontier ones. The construction and evaluation code is reproducible, and we plan to release all of it as open source with the benchmark.

---

> Discuss potential biases introduced by using proprietary models throughout the generation and validation pipeline.

We would like to thank the reviewer for a fair concern for any pipeline built with proprietary models, and we will expand the Discussion around it. Several design choices already reduce single-vendor influence. The construction quality panel is cross-vendor, so no one vendor decides acceptance. Generation and judging use different models. The v2 ground truth is execution-based and independent of any model's opinion. And we kept a human in the loop while designing and tuning the generation and validation stages. The human study above supports this as well: our automatic scoring agrees with a human judge with knowledge of the domain 85.1% of the time (κ = 0.67), so acceptance is not an artefact of one model's preferences.

There is also direct empirical evidence against a generator-family advantage. If the benchmark favoured its generator's family, since the v1 generator is GPT-4.1-mini from OpenAI, that family should top the leaderboard. It does not:

| Model | Family | v2 accuracy |
|---|---|---|
| Claude Sonnet 4.5 | Anthropic | 68.6% |
| GPT-4.1 | OpenAI (generator family) | 66.0% |
| GPT-4.1-mini | OpenAI (generator) | 60.0% |

A non-OpenAI model sits at the very top, above the generator's own family (GPT-4.1 is used for generation by default settings). A pipeline biased toward its generator would show the opposite ordering.

---

# Response to Reviewer 6zfv
=================================================================================

We thank the reviewer for these positioning questions, and we address each of them below with concrete measurements.

---

### What fundamental NLP capability does it advance?

> While the benchmark targets agentic financial analysis, it remains unclear what fundamental NLP capability it advances beyond a domain-specific evaluation resource.

To address this, FinOpsBench measures three capabilities that a domain QA resource does not, and we back each with a statistic rather than an assertion.

First, multi-step planning under partial observability. The model gets no data in context; it has to probe the schema or the tools, plan a retrieval path, and aggregate the result. Difficulty scales with the length of that path. Bucketing collected agentic accuracy by the required tool-chain depth, accuracy falls monotonically:

| Required tool-chain depth | 1 to 3 | 4 to 5 | 6 to 7 | 8+ |
|---|---|---|---|---|
| agentic accuracy, pooled across models | 61.6% | 61.1% | 57.0% | 45.8% |

Second, writing correct database queries in v1. The reference solutions are real analytic SQL, not single-table lookups:

| v1 SQL surface | share of items |
|---|---|
| uses a JOIN | 70% |
| uses an aggregate | 35% |
| uses GROUP BY | 31% |
| uses a subquery | 22% |
| needs two or more JOINs | 33% |

Third, turning an open-ended request into an analytical answer through a sequence of tool calls. For example, a Management Accountant asks "Analyze the fluctuations in the Raw Materials ledger account from Q2 2023 to Q2 2024, and explain the volume and price variances and their effect on gross margin." The gold answer is a written variance analysis. The reference solution reaches it in 10 SQL calls: resolve the account id, list the products, compute quarterly quantity, average unit cost and amount, join consumption to products, then compute quarterly revenue and cost, all while ignoring seeded distractor rows from a Finished-Goods account. No single call answers this, so the model has to plan the whole chain.

Together these are the core loop of any tool-using agent: read the intent, synthesize a plan, reject distractors, aggregate grounded evidence. FinOpsBench measures that loop in a controlled and verifiable financial setting.

---

### What is fundamentally new vs. recent agentic-finance benchmarks?

> multiple recent benchmarks have already moved in this direction. It remains somewhat unclear what fundamentally new evaluation capability FinOpsBench provides.

What is new is not agentic financial tool use itself, but a controllable, hermetic decomposition of agentic competence that realism-oriented benchmarks cannot offer. The environments are synthetic and executable, so every item is reproducible and difficulty is tunable, and the benchmark is built through several generation stages followed by explicit difficulty-raising stages. We also release the construction code, so the community can regenerate harder environments instead of consuming a fixed set.

We ran the models on a uniformly-sampled subset and observed that even the strongest agent stays below 70%, so the benchmark is far from saturated:

| Model | agentic accuracy |
|---|---|
| gpt-oss-120b | 69.9% |
| Claude-Sonnet-4.5 | 68.6% |
| GPT-4.1 | 66.0% |
| Claude-Haiku-4.5 | 65.5% |
| Qwen3-235B-A22B | 65.0% |
| GPT-4.1-mini | 60.0% |
| DeepSeek-V4-Flash | 54.3% |
| DeepSeek-V3.2 | 38.6% |
| Llama-3.3-70B | 19.8% |

The headroom is real and the difficulty is controllable, since accuracy drops from 61.6% at shallow depth to 45.8% at eight or more tool calls.

The clearest way to see what FinOpsBench adds is to compare it against an open static finance benchmark on the same model. GPT-4.1-mini answers TAT-QA, an external reading benchmark, at 89%, yet the same model scores 1.5% on FinOpsBench-v2 without tools and recovers to about 60% only once it uses them:

| Same model, GPT-4.1-mini | accuracy |
|---|---|
| TAT-QA, reading (external static finance QA) | 89% |
| FinOpsBench-v2, no tools | 1.5% |
| FinOpsBench-v2, agentic | ~60% |

Static finance benchmarks measure reading over provided context; FinOpsBench measures the retrieval-planning and tool-use capability they cannot reach. [PLACEHOLDER E12: extend this comparison to FinQA, ConvFinQA and MultiHiertt in reading mode against FinOpsBench-v2 agentic, to show the reading-to-tool-use gap holds across the open finance-QA landscape and not only against TAT-QA.]

---

### Fine-grained diagnostics beyond final-answer accuracy

> the reported analyses are primarily based on final-answer accuracy. More fine-grained diagnostic metrics or failure analyses ...

We thank the reviewer, and we agree the benchmark should yield more than a single accuracy number. The scoring it rests on is validated first: our automatic scoring matches a human judge with knowledge of the domain 85.1% of the time on v1 (κ = 0.67), and v2 is scored by execution, so the diagnostics below sit on calibrated ground. On top of that we add two diagnostic layers.

First, a failure-mode taxonomy over 779 failing traces in eight categories. Each cell is the share of that model's failing traces, and the profile shifts clearly from v1 to v2:

| Model | half | malformed args | incomplete retrieval | wrong-tool selection | calc error | round-limit |
|---|---|---|---|---|---|---|
| GPT-4.1 | v1 | 42% | 33% | 4% | 5% | 3% |
| GPT-4.1-mini | v1 | 36% | 37% | 10% | 5% | 0% |
| Claude-Sonnet-4.5 | v2 | 12% | 15% | 20% | 17% | 7% |
| DeepSeek-V3 | v2 | 7% | 16% | 23% | 14% | 25% |

Second, per-model process metrics, which separate models that land on the same accuracy:

| Model | avg tool calls per task |
|---|---|
| GPT-4.1 (v1) | 1.4 |
| Claude-Sonnet-4.5 (v2) | 4.1 |
| DeepSeek-V3 (v2) | 3.9 |

On v1 the failures are semantic, not syntactic: SQL errors are near zero, while wrong predicates and incomplete retrieval dominate, so models fail at data selection, not arithmetic. On v2 the profile moves to tool use under distractors. One trace makes the diagnostic value concrete: Qwen3-235B on a Citigroup contractual-obligations ratio first emitted a malformed percentage call, got a nonsense 0.0%, then self-corrected and computed compute_percentage(88472, 260754) = 33.9%, the gold answer. The taxonomy records both the slip and the recovery, which a final-answer metric would collapse into a single "correct".

---

### Dependence on LLM-generated queries, schemas, data, and judgments

> the final benchmark quality still depends substantially on LLM-generated queries, schemas, data, and judgments.

We appreciate this concern and share the goal of keeping benchmark quality independent of any single model's behaviour. Three points address it directly.

First, the dependence is asymmetric across the two halves. The v2 questions are human-authored, taken from FinQA, and v2 is validated by execution rather than by judgement: only the environment scaffolding is generated, and it is accepted only if running the reference plan reproduces the gold answer. In a 200-item sample, 178 of 200 do so.

Second, the LLM judgements are aligned with human perception, which we measured rather than assumed:

| Half | check | agreement |
|---|---|---|
| v1 | judge vs a human judge with domain knowledge | 85.1%, Cohen's κ = 0.67 |
| v2 | reference plan reproduces gold under execution | 178 of 200 |

So the judgement tracks a knowledgeable reader rather than adding noise. [PLACEHOLDER: inter-annotator κ from a second independent annotator, to report human-human agreement next to the human vs judge number above.]

Acceptance also does not rest on one model's opinion. The construction panel is three independent judges from two vendors, and on the released items they agree as follows:

| Panel criterion | judges unanimous |
|---|---|
| data is natural | 97% |
| reasoning is grounded | 90% |
| trace is reasonable | 90% |
| trace is sound | 83% |
| answer is sound | 62% |

The judges converge on the objective criteria and split most on the subjective one, answer soundness, which is exactly why acceptance is a majority vote of three models rather than a single call.

Third, the two halves act as mutual controls: per-model accuracies agree across them within 2.6 points on average, which would be unlikely if the synthetic construction of v1 were injecting systematic artifacts. The pipeline is LLM-assisted, but its output is gated by execution and calibrated against a human judge.

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
- **"Native business-driven agent tasks" are exactly what v1 provides — at scale.** v1 is 5979 analyst-authored tasks spanning AP aging, reconciliation, variance analysis, and revenue recognition (see the Controller/Management-Accountant examples above), each against a freshly generated database. The breadth-and-realism axis is carried by v1; the controlled-verifiability axis by v2. Neither half alone would make the argument; together they cover both.
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
| GPT-5-mini (n=1174) | **14.7%** | 67.5% | −52.8 pp |
| GPT-4.1 (n=300) | **13.3%** | 60.6% | −47.3 pp |
| Qwen3-30B-A3B (n=1174) | **13.8%** | 53.0% | −39.2 pp |

Closed-book accuracy is **flat at ~14%** across model families and capability levels, while agentic accuracy varies by 15 points. Memorization would *scale* with capability and training exposure; a flat floor is the signature of the residual scenario-text figures, not recall.

**(2) Access-mode dependence (200-item ladder).** A memorized answer would be emitted regardless of access mode. Instead, accuracy rises *monotonically* with more direct access to the data:

| Model | question-only | agentic (tools) | FinQA-native (gold facts in-context) |
|---|---|---|---|
| DeepSeek-V4-Flash | 2.5% | 54.3% | 68.0% |
| DeepSeek-V3.2 | 4.0% | 38.6% | 69.0% |
| Llama-3.3-70B | 3.0% | 19.8% | 57.0% |

A recall-driven score would be high and flat across all three columns; we observe the opposite — a steep, monotonic dependence on access. The FinQA answer is neither recoverable from memory (~2–4%) nor free even when the gold facts are handed over (~57–69%, not ~100%), and the agentic re-instantiation (split tables, distractor tools/rows, a bespoke multi-hop plan) adds real difficulty on top.

**(3) Half the benchmark is contamination-proof by construction.** FinOpsBench-v1 (5979 examples) is generated end-to-end against freshly created per-example databases and was never published — it cannot appear in any training corpus, so contamination there is impossible. Since v1 and v2 produce **consistent per-model rankings** (mean abs. diff 2.6 pp), the agentic difficulty we measure on v2 is corroborated by a half that cannot be contaminated.

Familiarity with the public FinQA items therefore does not translate into an answer pathway in v2: the system prompt contains neither the source table nor its values, the backing store is re-instantiated with distractor rows, and the required multi-hop tool plan exists in no training corpus.

---

*We are grateful to all three reviewers. The rebuttal added seven new experiments (human validation, judge calibration, closed-book contamination, an access-mode ladder, a failure taxonomy, diversity statistics, and construction-cost accounting) plus a cross-benchmark comparison, all released with code and data. We would be glad to run any further analysis the reviewers find useful during the discussion period.*
