## E7 — Construction cost, resources, runtime

Per-example figures are **measured** by replaying each pipeline stage's real prompt through OpenRouter with the same models the paper used (v1: gpt-4.1-mini + o4-mini + o3-mini + Claude-Sonnet-4; v2: o3); cost is OpenRouter's per-request `usage.cost`. Totals weight each stage by the construction funnel.

### Measured per-example cost & runtime

| | Model(s) | $/example | wall-time/example |
|---|---|---|---|
| v1 (9-stage panel pipeline) | gpt-4.1-mini, o4-mini, o3-mini, Claude-Sonnet-4 | $0.037 | 68s |
| v2 (9-stage exec pipeline) | o3 | $0.237 | 112s |

(n=2 examples each; v1 variance is driven by the SQL-repair trigger and agent-loop length.)

### v1 per-stage cost (mean of measured runs)

| Stage | Model | $/call |
|---|---|---|
| 1 Query gen (amortised /20) | openai/gpt-4.1-mini | $0.0000 |
| 2 Schema gen | openai/gpt-4.1-mini | $0.0008 |
| 3 Data gen | openai/gpt-4.1-mini | $0.0017 |
| 4 SQL repair (conditional) | openai/o4-mini | $0.0097 |
| 5 Agent trace (loop) | openai/gpt-4.1-mini | $0.0008 |
| 6 Judge — Claude-Sonnet-4 | anthropic/claude-sonnet-4 | $0.0095 |
| 6 Judge — o4-mini | openai/o4-mini | $0.0074 |
| 6 Judge — o3-mini | openai/o3-mini | $0.0099 |
| Final answer check | openai/o4-mini | $0.0018 |

### Extrapolated construction totals (stage × funnel)

| Version | Candidates processed | Final examples | Est. total cost | $/final example |
|---|---|---|---|---|
| v1 | 10,000 → 8,233 passed → 5,979 filtered | 5,979 | **~$449** | $0.075 |
| v2 | 1,247 attempted | 1,108 | **~$340** | $0.307 |
| **Total construction** | | **7,087** | **~$789** | |

**Where the cost goes.** In v1 the three-judge panel (run on 9,557 first-pass + 3,967 second-pass examples = ~13,500 judgements, each 3 reasoning-model calls) is ~81% of v1 construction cost; generation (schema+data) and the agent traces are comparatively cheap. v2 cost is dominated by the two o3 code-generation stages (initial + augmented DB/tools), ~65% of its 10 stages.

### Runtime & compute

- Serial LLM wall-time: v1 ~190 API-hours, v2 ~39 API-hours; both pipelines run 8-way parallel, so real wall-clock is ~1/8 of that (v1 ≈ 24h, v2 ≈ 5h).
- Construction uses **API models only — no GPU**. The single NVIDIA H100 reported in the paper is used at **evaluation** time to serve the open-source agents (Qwen3-8B/30B-A3B, Llama-3.1-8B); frontier models are API-served.
- Backing stores are in-memory SQLite; CPU/RAM footprint is negligible.

### Evaluation cost (for completeness)

Measured via the same OpenRouter accounting in our rebuttal runs: a full agentic pass over v2 costs ~$0.005/example for an open model (DeepSeek-V3, $5.18 for 1,134) and ~$0.06/example for a frontier model (Claude-Sonnet-4.5). A closed-book v2 pass is ~$0.6–0.9 per model over the full set. These are the per-model evaluation costs; the benchmark itself is reusable at no regeneration cost.

