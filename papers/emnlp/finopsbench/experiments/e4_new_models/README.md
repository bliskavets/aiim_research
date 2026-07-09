# E4 — Additional model families on FinOpsBench-v2 (Claude, DeepSeek)

**Claim tested:** "Experiment evaluation is incomplete: missing top agent/code
frontier models ...; baselines only cover tiny open-source models" (Reviewer R3);
cross-family check against generator-family bias (Reviewer PVoW).

**Design.** Models are evaluated with the benchmark's own smolagents harness:
`SA_openrouter.py` is a verbatim copy of the benchmark's `agent_runners/SA.py`
plus OpenRouter provider routing via the `OPENROUTER_EXTRA_BODY` env var
(needed because one DeepSeek provider rejects smolagents' request format).
`run_e4.py` orchestrates per-environment subprocess runs, resumable, and
scores with the benchmark's standard `compare_outputs.compare_answers`.

**Cost control.** OpenRouter's `/api/v1/credits` endpoint is snapshotted
before/after each run and every 25 items; a run aborts if the projected total
exceeds `--budget_usd`. Per-run spend is appended to `results/costs.json`.
(The credits endpoint lags in-flight requests by a few seconds; final spend is
taken from the post-run snapshot.)

**Run:**
```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_EXTRA_BODY='{"provider":{"ignore":["Novita"]}}'
python run_e4.py --model deepseek/deepseek-chat-v3-0324 --runner SA_openrouter.py \
    --python /tmp/e4venv/bin/python --budget_usd 6
python run_e4.py --model anthropic/claude-sonnet-4.5 --runner SA_openrouter.py \
    --python /tmp/e4venv/bin/python --sample 250 --budget_usd 13   # seed 13
```

Requires: a venv with `smolagents`, `mlflow`, `openai`; an MLflow server on
`localhost:7777` (the benchmark's `mlflow.sh`).

Results: `results/<model>.json` (same format as the benchmark's `v2/results/`),
`results/costs.json` (spend ledger), per-item raw outputs in `results/<model>/`.
