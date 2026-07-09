# E4 results

Evaluated with the paper's own v2 harness (smolagents `SA.py`, `max_steps=10`)
and scoring (`compare_outputs.compare_answers`, one-least-significant-digit
tolerance), via OpenRouter.

| Model | Family | n | Accuracy v2 (%) | Spend (USD) |
|---|---|---|---|---|
| Claude Sonnet 4.5 | Anthropic (frontier) | 139 | 70.5 | 7.83 |
| DeepSeek-V3-0324 | DeepSeek (open-weight, 671B MoE) | 1,134 | 57.3 | 5.18 |

Paper v2 leaderboard for context: GPT-5 69.6, GPT-5-mini 67.5, o4-mini 67.3,
GPT-4.1 60.6, GPT-4.1-mini 56.9, Qwen3-30B-A3B 53.0, Qwen3-8B 44.1,
Llama-3.1-8B 16.3.

Notes / limitations:
- Claude Sonnet 4.5 was budget-stopped at n=139 of a planned 250-item random
  sample (seed 13); the number is a subset estimate, not the full v2 set.
- DeepSeek: 1,134 of 1,165 attempted environments scored; 31 failed on
  transient OpenRouter provider / usage-metadata errors and were dropped.
- Model versions and provider routing are pinned via OpenRouter; small
  provider-side variance is possible.

Total E4 spend (incl. pilots): ~$13.4. Cost measured via OpenRouter's
`/api/v1/credits` delta around each run (see `costs.json`).
