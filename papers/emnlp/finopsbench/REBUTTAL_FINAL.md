=================================================================================
# Reviewer PVoW

We thank the reviewer for the thorough review. We ran additional analyses and address each concern below; we will fix the minor typos and wording in the camera-ready.

### Human validation and LLM-judge reliability

> Heavy dependence on LLM-generated data without human validation ... no human evaluation that scenarios are realistic, traces correct, or the LLM judges reliable, and no measured agreement with human annotators or estimate of judge accuracy.

> Evaluation is relatively weak: FinOpsBench-v1 relies on an LLM judge rather than deterministic correctness whenever possible.

To address the reviewer's concern, we ran the human evaluation they asked for on both halves, measuring how far our automatic scoring departs from a human's. A human judge with domain knowledge annotated random samples, in the reviewer's suggested 200 to 300 range.

|Part|assessment criteria|n|human vs auto scoring|
|-|-|-|-|
|v1|human eval vs auto scoring|172|85.1% agreement, Cohen's κ = 0.67|
|v2|exec. of the ref. plan reproduces the gold answer|200|178 of 200 valid (89%)|

For v2, scoring is deterministic, so the audit checked environment validity, not the scorer.

This is also why v1 uses an LLM judge: most v1 answers are free-form analyst deliverables, not single numbers a string or numeric match could grade. One item as it appears in the data, for a Senior Accountant asking "What exceptions exist between the invoice volumes and timing of payments that could signal processing errors?":

```
Exceptions indicating possible processing errors:
- Invoice 102 (Beta LLC): paid 5 days before the invoice date.
- ... 4 more, each a different type: partial payment, late payment, overpayment, missing payment ...
- Payment 1006: refers to invalid invoice_id 999, likely a data-entry error.
These signal processing issues in the timing or volume of payments relative to invoices.
```

With no unique target string for such answers, semantic evaluation is necessary for v1, and the study above shows the judge tracks a domain-knowledgeable human closely. A human was also in the loop during pipeline design and validation.

### Releasing the pipeline prompts

> Release all prompts used throughout the nine-stage pipelines (query, schema, and data generation, feedback reconciliation, system-prompt construction).

We release all of them. The paper's repo link now points to the current version, with a top-level `PROMPTS.md` mapping each stage to its exact prompt: v1 stages one to nine and final filtering, the v1 evaluation prompts including the judge grading prompt, and the full v2 environment-generator prompts.

### Benchmark diversity and failure analysis

> Analyze benchmark diversity more quantitatively: reasoning operations, SQL complexity, tool-chain depth, numerical operations, financial concepts, template diversity ...

> Provide qualitative examples of common model failures beyond accuracy: tool misuse, reasoning mistakes, planning failures, financial misunderstandings.

Part of this is in the paper: Appendix C gives per-item statistics (query/table/row counts, turns, tool calls, prompt lengths and tool counts), Appendix D example queries, Appendix G the v1 category distribution. To address the request directly, we add one measurement per axis the reviewer named.

v1 lexical diversity and SQL surface, over the 8233-task pool:

|v1 diversity|value|
|-|-|
|tasks in pool|8233|
|distinct user roles|742|
|distinct queries|100%, no duplicates|
|lexical diversity|distinct-3-gram ratio 0.52|
|tool calls per task|mean 1.4, median 1, p90 3, max 10|
|ref.-solution SQL surface|JOIN 70%, ORDER BY 42%, aggregate 35%, GROUP BY 31%, subquery 22%, date function 19%, CASE 9%, HAVING 7%|

The reference solutions need more than simple lookups. Their structure:

|v1 SQL structural depth|value|
|-|-|
|queries with a nested subquery|17%, of which 10% nest two or more levels|
|clauses per query|mean 4.1, max 8|
|items requiring two or more JOINs|33%|

So a substantial portion of v1 needs multi-table analytical reasoning, not single-table retrieval.

v2 tool-use structure, over the released environments:

|v2 diversity|value|
|-|-|
|reference-plan tool calls per env.|mean 4.9, median 5, p90 7, max 15|
|tools available per env.|mean 8.9, median 9, p90 11, max 14|
|off-path tools per env. (distractor + partial-info)|mean 3.2, median 3, at least two in 92% of envs|
|numerical-operation mix|aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent-change 11%|

Questions come from 124 companies across more than 1000 FinQA filings.

To address the concern that the benchmark sits on one topic, the two halves cover complementary concepts:

|Financial concept, share of examples|v1|v2|
|-|-|-|
|accounts payable, invoices, vendors|52%|0%|
|approval, authorization, controls|18%|2%|
|overdue, aging, late payment|16%|11%|
|variance, budget vs actual|13%|1%|
|reconciliation, discrepancy|6%|0%|
|financial-statement ratios|7%|77%|

v1 concentrates on accounts payable, controls and variance; v2 on financial-statement ratios: largely complementary.

Reasoning operations differ too. v1 analyst intent: enumeration 22%, anomaly search 16%, comparison 4%, quantification 2%; by task category: Accounts Payable 52%, financial reporting 25%, variance 13%, revenue recognition 8%.

On template diversity, 12 seed queries expand to 8233 examples (686x, filtered at cosine 0.9), 100% distinct: distinct-3-gram ratio 0.52, distinct-4-gram 0.74, 0.0% high-overlap pairs. v2 keeps the FinQA questions verbatim, so its phrasing is human, not templated.

Difficulty also rises with required tool-chain depth: bucketed by depth, accuracy falls monotonically:

|Required tool-chain depth|1 to 3|4 to 5|6 to 7|8+|
|-|-|-|-|-|
|pooled agentic acc.|61.6%|61.1%|57.0%|45.8%|
|DeepSeek-V3|61.9%|58.6%|54.6%|43.4%|

For the failure ask, we classified 779 failing traces into eight categories with process metrics; the profile shifts from v1 to v2:

|Model|half|malformed args|incomplete retrieval|wrong-tool selection|calc error|round-limit|
|-|-|-|-|-|-|-|
|GPT-4.1|v1|42%|33%|4%|5%|3%|
|GPT-4.1-mini|v1|36%|37%|10%|5%|0%|
|Claude-Sonnet-4.5|v2|12%|15%|20%|17%|7%|
|DeepSeek-V3|v2|7%|16%|23%|14%|25%|

v1 failures are primarily semantic, not syntactic: SQL errors near zero, while wrong predicates and incomplete retrieval dominate, so models fail at data selection, not arithmetic. v2 shifts to tool use: wrong-tool selection rises under distractors, and DeepSeek-V3 hits the step limit in a quarter of its failures. Process metrics match: v1 frontier models fail fast, 1.3 to 1.9 calls with no round-exhaustion, while v2 agents make about four calls and hit the step limit 7 to 11% of the time. This separates models at similar final-answer accuracy but different failure profiles. On "Which invoices have duplicate payment records, and what is the total overpaid?" GPT-4.1 aggregated at the invoice level instead of detecting repeated identical payments, reporting only Invoice 4 at $0.01 and missing the real duplicates on Invoices 1, 3 and 5 of $600, $200 and $450.

### Construction cost, compute, and runtime

> Report annotation or generation costs, computational resources, and runtime required to construct the benchmark.

Construction is fully automated, with no paid human annotation; the cost is LLM API usage, measured by replaying each stage with the paper's models.

|Version|Candidates to final|Est. construction cost|$/final example|
|-|-|-|-|
|v1 (9-stage panel pipeline)|10000 to 5979|~$450|$0.075|
|v2 (9-stage execution pipeline)|1247 to 1108|~$340|$0.307|
|Total|7087 final|~$790|n/a|

The three-judge panel dominates v1 cost (about 81%, roughly 13500 judgements across three reasoning-model calls); the two o3 code-generation stages dominate v2 (about 65%). Construction is API-only, no GPU: the single H100 serves the open-source agents only at evaluation; backing stores are in-memory SQLite. Both run 8-way parallel, wall-clock about 24 hours (v1) and 5 hours (v2); per-model evaluation costs about $0.005 per example for open models, up to $0.06 for frontier ones. We release all construction and evaluation code with the benchmark, so these measurements are reproducible.

### Potential bias from proprietary models

> Discuss potential biases introduced by using proprietary models throughout the generation and validation pipeline.

Several design choices reduce single-vendor influence. The construction quality panel is cross-vendor, so no one vendor decides acceptance; generation and judging use different models; the v2 ground truth is execution-based, independent of any model's opinion; and we kept a human in the loop while designing and tuning the stages. The study above supports this: acceptance tracks a domain-knowledgeable human, not one model's preferences.

Direct evidence against a generator advantage is best read off v1, the half the generator (GPT-4.1-mini) produced: if the pipeline rewarded its generator, GPT-4.1-mini should top v1; instead it is the lowest frontier model on v1 (Table 2):

|Model (v1 frontier tier)|v1 acc.|
|-|-|
|GPT-5|68.9%|
|o4-mini|67.1%|
|GPT-5-mini|65.8%|
|GPT-4.1|62.4%|
|GPT-4.1-mini (the generator)|61.5%|

The generator gains no advantage on the data it built, at the bottom of the frontier tier. The remaining differences track base-model capability (the log-linear size-accuracy relationship in the paper), not vendor identity, and the cross-vendor panel means no single provider decides acceptance.

---

# Reviewer 6zfv
=================================================================================

We thank the reviewer for raising important questions about the benchmark’s scope, novelty, and diagnostic value.

---

### Fundamental NLP capability advanced

> While the benchmark targets agentic financial analysis, it remains unclear what fundamental NLP capability it advances beyond a domain-specific evaluation resource.

FinOpsBench measures capabilities a domain QA resource does not, each backed by a statistic. One is multi-step planning under partial observability: with no data in context, the model must probe the schema, plan a retrieval path, and aggregate the result. Difficulty tracks that path's length:

|Required tool-chain depth|1 to 3|4 to 5|6 to 7|8+|
|-|-|-|-|-|
|agentic acc., pooled across models|61.6%|61.1%|57.0%|45.8%|

It also requires writing real analytic SQL, not single-table lookups:

|v1 SQL surface|share of items|
|-|-|
|uses a JOIN|70%|
|uses an aggregate|35%|
|uses a subquery|22%|
|needs two or more JOINs|33%|

And it has to turn an open-ended request into an analysis over many tool calls. One v1 item:

```
Prompt (Management Accountant): analyze the fluctuations in the Raw Materials
ledger account from Q2 2023 to Q2 2024, and explain the volume and price
variances and their effect on gross margin.

Golden answer: unit cost is stable near 10.0, spikes to 12.0 in Q3 2023, then
moves between 11.0 and 12.5 through Q2 2024; rising cost squeezes gross margin,
falling cost improves it. The Finished-Goods rows and a mislinked
product_raw_materials row are distractors.

Model's SQL (10 calls; quarter = a strftime date bucket), e.g.:
  SELECT quarter, p.name, AVG(s.unit_price) FROM sales s
    JOIN products p ON s.product_id = p.id GROUP BY quarter, p.name;
  ... other calls: id lookups and per-quarter revenue/cost/quantity totals ...

Model's response (abridged): per-quarter ledger, consumption and per-product
sales tables, concluding the Q1 2024 cost peak compresses margin while stable
selling prices cushion it.
```

No single call answers this: the model must pick the right account, drop the seeded Finished-Goods distractors, and chain ten queries into one analysis.

None of this is finance-specific. Parsing an ambiguous instruction, planning, writing queries, calling tools correctly, ignoring distractors, and composing a grounded answer are the core loop of any tool-using agent. Finance just supplies executable semantics; those abilities are broadly applicable, so FinOpsBench evaluates planning, retrieval, and tool use on top of finance-specific knowledge.

---

### Novelty versus recent agentic-finance benchmarks

> multiple recent benchmarks have already moved in this direction. It remains somewhat unclear what fundamentally new evaluation capability FinOpsBench provides.

We agree recent benchmarks also study financial agents. Our contribution differs: the environments are synthetic and executable, so every item is reproducible and difficulty tunable, built through several generation stages plus explicit difficulty-raising stages. We also release the construction code, so the community can regenerate harder environments instead of consuming a fixed set.

On a uniformly-sampled subset, even the strongest agent stays below 70%, so the benchmark is far from saturated:

|Model|agentic acc.|
|-|-|
|gpt-oss-120b|69.9%|
|Claude-Sonnet-4.5|68.6%|
|Claude-Haiku-4.5|65.5%|
|Qwen3-235B-A22B|65.0%|
|GPT-4.1-mini|60.0%|
|DeepSeek-V4-Flash|54.3%|
|DeepSeek-V3.2|38.6%|
|Llama-3.3-70B|19.8%|

This is substantial headroom, and the depth table above shows difficulty is controllable through tool-chain length.

Another comparison runs one model across both settings. Static finance QA hands the model the relevant table and text in the prompt, so reading scores well; FinOpsBench withholds the data, so without tools the same model collapses and must retrieve through tool calls to recover:

|Same model, GPT-4.1-mini|acc.|
|-|-|
|TAT-QA, reading (data in prompt)|89%|
|FinQA, reading (data in prompt)|67%|
|FinOpsBench-v2, no tools (data withheld)|1.5%|
|FinOpsBench-v2, agentic (retrieves via tools)|61.5%|

Reading a static benchmark and operating in FinOpsBench are different skills: static benchmarks measure reading over provided context; FinOpsBench also measures retrieval planning and tool use, unneeded when the evidence is given directly.

The reviewer rightly points to recent agentic finance benchmarks, so for comparison we looked at three of the newest, FinAgentBench, FinGAIA, and Herculean, which emphasize different goals: FinAgentBench retrieval and ranking, FinGAIA and Herculean realistic interactions with external systems. That realism is useful but costs reproducibility and control. FinOpsBench keeps both: hermetic synthetic environments that rerun identically, answers scored against an executable ground truth rather than a ranking or rubric, and difficulty exposed as a knob via released construction code. The two are complementary: realism-oriented benchmarks test deployment-like behavior, FinOpsBench enables controlled, reproducible diagnosis.

---

### Diagnostics beyond final-answer accuracy

> the reported analyses are primarily based on final-answer accuracy. More fine-grained diagnostic metrics or failure analyses ...

The benchmark already reports metrics beyond final-answer accuracy; we add a human-grounded check and two diagnostic layers built on it.

Following the reviewer's request, we ran a human annotation study with a human judge who has domain knowledge; it confirms the diagnostics align closely with human judgement:

|Half|check|agreement|
|-|-|-|
|v1|scoring vs a human judge w/ domain knowl.|85.1%, Cohen's κ = 0.67|
|v2|ref. plan reproduces gold under execution|89%|

Anchored this way, the diagnostics are meaningful, not circular. A failure-mode taxonomy over 779 failing traces in eight categories shows the profile shifting from v1 to v2; each cell is the share of that model's failing traces:

|Model|half|malformed args|incomplete retrieval|wrong-tool selection|calc error|round-limit|
|-|-|-|-|-|-|-|
|GPT-4.1|v1|42%|33%|4%|5%|3%|
|GPT-4.1-mini|v1|36%|37%|10%|5%|0%|
|Claude-Sonnet-4.5|v2|12%|15%|20%|17%|7%|
|DeepSeek-V3|v2|7%|16%|23%|14%|25%|

Per-model process metrics separate models at the same accuracy that behave differently:

|Model|avg tool calls per task|
|-|-|
|GPT-4.1 (v1)|1.4|
|Claude-Sonnet-4.5 (v2)|4.1|
|DeepSeek-V3 (v2)|3.9|

On v1, failures are primarily semantic, not syntactic (SQL errors near zero, wrong predicates and incomplete retrieval dominant), so models fail at data selection, not arithmetic; on v2 the profile moves to tool use under distractors. One trace makes this concrete: Qwen3-235B on a Citigroup contractual-obligations ratio first emitted a malformed percentage call, got a nonsense 0.0%, then self-corrected to compute_percentage(88472, 260754) = 33.9%, the gold answer. The taxonomy records both the slip and the recovery, which a final-answer metric would collapse into a single "correct".

---

### Dependence on LLM-generated data and judgments

> the final benchmark quality still depends substantially on LLM-generated queries, schemas, data, and judgments.

We share the goal of keeping quality independent of any single model. The dependence is asymmetric across the halves: the v2 questions are human-authored, from FinQA, and v2 is validated by execution, not judgement. Only the environment scaffolding is generated, accepted only if running the reference plan reproduces the gold answer. The human validation:

|Half|check|agreement|
|-|-|-|
|v1|judge vs a human judge w/ domain knowl.|85.1%, Cohen's κ = 0.67|
|v2|ref. plan reproduces gold under execution|89%|

The judge therefore agrees substantially with the domain-knowledgeable human. The released set is also what survives the funnel, not raw generation: about 40% of v1 and 11% of v2 candidates are discarded by execution checks, the answer-consistency filter, and the panel:

|Version|generated|released|
|-|-|-|
|v1|10000|5979|
|v2|1247|1108|

Acceptance also does not rely on one model's opinion. The construction panel is three independent judges from two vendors; on the released items they agree as follows:

|Panel criterion|judges unanimous|
|-|-|
|data is natural|97%|
|reasoning is grounded|90%|
|trace is reasonable|90%|
|trace is sound|83%|
|answer is sound|62%|

The judges converge on the objective criteria and split most on the subjective one, answer soundness, which is why acceptance is a majority vote of three models, not a single call. The scoring judge is similarly robust on hard cases: on the 92 items where it and a strict exact-match check disagree, a human sides with the judge on 82.6% (Cohen's κ = 0.64), so even where scorers conflict, the retained judgement is the one a knowledgeable reader endorses.

The two halves also act as mutual controls: per-model accuracies agree within 2.6 points on average, unlikely if v1's synthetic construction were injecting systematic artifacts. The pipeline is LLM-assisted, but its output is gated by execution and calibrated against a human judge.

The dependence on specific models is not fixed either: we release the full construction and extension code, so the community can swap the generator or judge models, adjust any stage, and regenerate or extend the benchmark rather than rely on our choices.

---

=================================================================================
# Response to Reviewer j7in
=================================================================================

We thank the reviewer for the detailed review. Several concerns are about design choices (ground truth, the FinQA derivation, contamination), which we address with direct experiments; we also clarify points on the harness and model selection.

---

### v1 and machine-verifiable ground truth

> v1 lacks machine-verifiable ground truth; relies on LLM panel judges, giving subjective, biased results.

Two clarifications. **(a) Every v1 example does carry a hard expected answer** `expected_output`, created jointly with the data in Stage 3 and enforced by execution-based validation (Stage 4) plus an answer-consistency filter. The panel is a quality gate *on top of* this ground truth, not a replacement. **(b) We measured how far machine-verifiable scoring can go**: deterministic numeric matching is well-defined for only **4.4%** of v1 expected answers; the other 95.6% are multi-entity analyst deliverables (ranked invoice-exception lists, per-supplier variance tables, policy conclusions; examples in our PVoW response) for which token/numeric matching is undefined. On the scalar subset where it *is* defined, the judge agrees with numeric matching on 74.3% of items; on the 92 disagreements a **human judge with domain knowledge sides with the judge, not numeric matching (only 17.4%)**. We also compared the judge against the same human on a stratified v1 sample:

|v1 evaluation check|agreement with the human|
|-|-|
|judge vs a human judge w/ domain knowl. (overall)|85.1%, Cohen's κ = 0.67|
|the 92 items where judge and numeric matching disagree|82.6%, Cohen's κ = 0.64|

This supports semantic evaluation for v1; where machine-only scoring conflicts, the judge is the *more accurate* scorer by ~5×. v2, whose answers are numeric by construction, is scored **fully deterministically** against executable reference plans (no LLM).

---

### v2 derived from FinQA: "monotonous," "artificially added multi-hop"

> v2 is built entirely on FinQA, not designed for agent tool workflows; query types are monotonous, with artificially added multi-hop tool logic rather than native business-driven agent tasks.

The two halves are designed to be read together, and deriving v2 from FinQA is a deliberate choice, not a shortcut.

- **The FinQA derivation enables a controlled comparison between reading-based and tool-mediated access.** Holding the *question content* fixed and varying only the access mode isolates the difficulty from planning, retrieval, and tool use and attributes any drop to the agentic component: state-of-the-art systems reach roughly 80-85% on static FinQA, yet the best agent reaches only ~69% on the *same questions* here. The "artificially added multi-hop logic" is the measurement instrument: it turns a reading task into a planning-and-tool-use task on identical content. The access ladder (see Reviewer 6zfv) quantifies this: a model that reads FinQA at 57-69% acts at 20-54% with tools, and the gap is model-discriminating.
- **"Native business-driven agent tasks" are what v1 provides at scale.** v1 has 5979 analyst-style tasks spanning AP aging, reconciliation, variance analysis, and revenue recognition (see the Controller/Management-Accountant examples above), each against a freshly generated database. v1 carries the breadth-and-realism axis, v2 the controlled-verifiability axis; neither half alone makes the argument, together they cover both.
- **On "monotonous":** we report v2's operation-type distribution (aggregation 51%, difference/YoY 41%, ratio 32%, average 11%, percent-change 11%; median 5 tool calls over 9 tools), while v1's 742 distinct roles and zero duplicate queries provide lexical and structural breadth.

---

### Missing top agent models and finance-specialized LLMs

> Experiment evaluation is incomplete: missing top agent/code models (Claude Code, Codex, OpenCode); baselines only cover tiny open-source models without finance-specialized LLMs.

Two parts to this.

**(1) Claude Code, Codex and OpenCode are agent products, not base models, and the base models they run on are already in our evaluation.** Claude Code runs on Claude, Codex on OpenAI models, and we evaluate exactly those under our fixed harness (Claude Sonnet 4.5, GPT-5, GPT-4.1, and others). These products add scaffolding: their own system prompts, retry logic, file and shell tooling, plus a bespoke protocol to expose our tools. A score through them measures product engineering, not the model, and is not reproducible as products update. The paper already shows the harness alone moves the number: on v1, switching only the tool-calling protocol from native to ReAct shifts accuracy up to 6.4 pp and can flip its sign, helping non-thinking models and hurting thinking ones; a full product harness adds far more. Evaluating base models under one fixed, open harness is standard for agentic benchmarks (AgentBench, τ-bench) and keeps the comparison controlled and reproducible.

**(2) The paper already evaluates frontier models, not only small open ones, and we broadened coverage.** Table 2 already includes GPT-5, o4-mini and GPT-4.1 alongside open-source models; across the paper and this rebuttal we report more than a dozen models across five vendors (OpenAI, Anthropic, Alibaba, DeepSeek, Meta). Under the paper's exact v2 harness and scoring, our controlled 200-item evaluation spans five families; the additions below land where the size-accuracy trend predicts, and a small model (Haiku) shows tool-use quality is training-bound, not size-bound:

|Model|Family|agentic acc.|note|
|-|-|-|-|
|gpt-oss-120b|OpenAI (open-weight)|**69.9%**|tied top|
|Claude-Sonnet-4.5|Anthropic (frontier)|**68.6%**|non-OpenAI vendor at the top|
|Claude-Haiku-4.5|Anthropic (small)|**65.5%**|small model, ~0 read-act gap|
|Qwen3-235B-A22B|Alibaba (open-weight)|**65.0%**|large open MoE, mid-table|
|DeepSeek-V4-Flash|DeepSeek (open-weight)|**54.3%**|reads 68%, acts 54%|
|Llama-3.3-70B|Meta (open-weight)|**19.8%**|reads 57%, acts 20%|

**On finance-specialized LLMs:** open finance models are continued-pretrained on financial *text* and do not support reliable function calling, the exact capability under test, so they cannot run as tool-using agents without external scaffolding (which reintroduces the harness-conflation problem above). We state this explicitly and treat it as an open call for finance models trained for agentic tool use.

---

### "Outdated smolagents" / framework noise

> Adopts outdated smolagents as agent harness, which may introduce framework noise and interfere with reliable tool-use performance measurement.

Three clarifications:

- **smolagents is current and actively maintained.** It is a 2025 Hugging Face library, not a deprecated framework, and we chose it because it is a *minimal* harness.
- **v1 does not use smolagents at all.** v1 runs a minimal native tool-calling loop (and a ReAct variant) directly over the model API, so any smolagents-specific concern cannot apply to more than half the benchmark.
- **Framework noise is addressed empirically.** We evaluate under two protocols (native tool calling and ReAct) and two independent stacks; the model ranking is consistent across all, and per-model accuracies agree across the two versions (mean absolute difference 2.6 pp). If harness artifacts drove results, this cross-harness, cross-version agreement would not hold.

---

### Data-contamination risk for v2

> High risk of data contamination for v2, as core questions come from widely publicized FinQA training corpus.

This is an important concern, which we evaluate directly in three ways:

**(1) Closed-book baseline (full benchmark).** We give the model the entire v2 environment prompt (scenario + tool signatures + question) but **no callable tools**, requiring it to answer from the prompt and its memory: a *conservative* upper bound on memorization (it sees strictly more than plain closed-book, and some scenarios contain a needed figure, which is why this floor sits above the stricter *question-only* floor in (2)). Result (v2 scoring unchanged; the agentic column is full-benchmark accuracy, not comparable to the 200-item figures in (2)):

|Model|closed-book|agentic|Δ|
|-|-|-|-|
|GPT-5-mini (n=1174)|**14.7%**|67.5%|−52.8 pp|
|GPT-4.1 (n=300)|**13.3%**|60.6%|−47.3 pp|
|Qwen3-30B-A3B (n=1174)|**13.8%**|53.0%|−39.2 pp|

Closed-book accuracy is **flat at ~14%** across model families and capability levels, while agentic accuracy varies by 15 points, so memorization is not what drives the agentic differences. We attribute the residual closed-book score partly to figures exposed in the scenario text, though this cannot rule out all contamination.

**(2) Access-mode dependence (200-item ladder).**
If memorization were the primary driver, question-only performance would be much higher and less dependent on access mode. Instead, accuracy rises *monotonically* with more direct access:

|Model|question-only|agentic (tools)|FinQA-native (gold facts in-context)|
|-|-|-|-|
|DeepSeek-V4-Flash|2.5%|54.3%|68.0%|
|DeepSeek-V3.2|4.0%|38.6%|69.0%|
|Llama-3.3-70B|3.0%|19.8%|57.0%|

A recall-driven score would be high and flat across all three columns; instead we see a steep monotonic dependence on access. The FinQA answer is neither recoverable from memory (~2-4%) nor free even when the gold facts are handed over (~57-69%, not ~100%), and the agentic re-instantiation (split tables, distractor tools/rows, a bespoke multi-hop plan) adds real difficulty on top.

**(3) Half the benchmark is substantially less exposed to contamination by construction.**
FinOpsBench-v1 (5979 examples) is generated end-to-end against freshly created per-example databases and was never published, so direct instance-level contamination is unlikely. Since v1 and v2 give **consistent per-model rankings** (mean abs. diff 2.6 pp), the agentic difficulty we measure on v2 is corroborated by a half far less exposed to contamination.

Familiarity with public FinQA items thus gives no answer pathway in v2: the prompt contains neither the source table nor its values, the store is re-instantiated with distractor rows, and the required multi-hop plan exists in no training corpus.
