# E7 — Construction cost, resources, runtime

**Claim addressed:** PVoW-7 ("Report annotation or generation costs, computational
resources, and runtime required to construct the benchmark").

Construction is fully automated (no paid human annotation); the cost is LLM API
usage. We **measure** it directly rather than estimating: each pipeline stage's
real prompt is replayed through OpenRouter with the same models the paper used,
and cost is taken from OpenRouter's per-request `usage.cost`.

## Method
- `measure_v1.py` — replays the v1 stages (query/schema/data = gpt-4.1-mini;
  SQL-repair/answer-check = o4-mini; agent trace = gpt-4.1-mini tool-calling loop
  against a real in-memory SQLite; panel = Claude-Sonnet-4 + o4-mini + o3-mini).
  Prompts copied verbatim from `v1/finopsbench_v1/pipeline/*.py`.
- `measure_v2.py` — reconstructs the 10-stage o3 generator prompt exactly as
  `create_dataset_v3.make_long_prompt` builds it (cumulative past-stage files from
  a real agent dir), calls o3 per stage.
- `extrapolate.py` — weights measured per-stage cost by the construction funnel
  (paper's stage counts) to get full-benchmark totals; writes `E7_ANSWER.md`.

## Measured results (n=2 examples each)
| Version | $/example | wall-time/example |
|---|---|---|
| v1 | $0.037 | 68s |
| v2 | $0.237 | 112s |

## Extrapolated totals
| Version | Candidates → final | Est. total | $/final |
|---|---|---|---|
| v1 | 10,000 → 5,979 | ~$450 | $0.075 |
| v2 | 1,247 → 1,108 | ~$340 | $0.307 |
| **Total** | 7,087 | **~$790** | |

v1 cost is ~81% three-judge panel; v2 is ~65% the two o3 code-gen stages.
Construction is API-only (no GPU); the H100 is used only for open-source
evaluation. Full breakdown in `E7_ANSWER.md`, raw runs in `results/`.

Run: `OPENROUTER_API_KEY=... python measure_v1.py --n 2 && python measure_v2.py --n 1 && python extrapolate.py`
