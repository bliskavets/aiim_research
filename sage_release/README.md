# SAGE — Self-judged Aspect-Guided Exploration

Test-time alignment for large language models: refine a **frozen** model's
outputs at inference time using the *same* model as both generator and soft
judge — **no retraining, no external reward model**.

SAGE (i) samples a small batch of candidate responses, (ii) scores each one with
a **contrastive label margin** read directly from a judging prompt
(`logprob(yes) − logprob(no)`), (iii) aggregates per-aspect margins into a single
calibrated preference score, (iv) forms *best* and *worst* groups and distills a
concise textual **"gradient"** describing what separates them, and (v) conditions
the next round of sampling on that guidance — all under a fixed token budget.

```
                 ┌── generate ──┐   ┌──── judge ────┐   ┌──── reflect ────┐
   prompt  ──▶   │  N hypotheses │─▶ │ contrastive   │─▶ │ best/worst sets │─┐
                 │  (temp. sched)│   │ label margins │   │ textual gradient│ │
                 └───────────────┘   └───────────────┘   └─────────────────┘ │
        ▲                                                                     │
        └──────────────────  ×T optimization epochs  ◀───────────────────────┘
                                     │
                                     ▼
                         argmax score  ──▶  selected answer
```

## Repository layout

```
sage/                     core library
├── solver.py             the SAGE algorithm (process_query + scoring/grouping/refinement)
├── engine.py             vLLM completions client + non-thinking / thinking engines
├── config.py             benchmark presets and the paper's inference-time budget
├── __main__.py           CLI:  python -m sage --preset ... --input ... --output ...
├── prompts/              judge prompt templates
│   ├── math_verification.txt
│   ├── instruction_following.txt
│   ├── alpaca_attributes.txt
│   └── alpaca_verification.txt
└── configs/              judge scoring configs (aspect tags, polarity, weights)
    ├── math500.json
    ├── general_verdict.json
    ├── alpacaeval.json
    └── verification.json
scripts/                  serve_vllm.sh, run_benchmark.py
examples/                 quickstart.py
tests/                    unit tests for the scoring primitives (no server needed)
```

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
```

## Serve a model

SAGE reads token log-probabilities from an OpenAI-compatible `/v1/completions`
endpoint (vLLM exposes it by default):

```bash
bash scripts/serve_vllm.sh Qwen/Qwen3-8B-FP8       # [model] [port] [max_len] [gpu_util]
```

## Run

```bash
# math (best settings loaded automatically from the preset)
python -m sage --preset math500 --input data/math500.jsonl --output out/math500.jsonl

# open-ended preference (multi-aspect judge)
python -m sage --preset alpacaeval --input data/alpaca.jsonl --output out/alpaca.jsonl

# reasoning-mode benchmark (e.g. AIME): switch the generator to thinking mode
python -m sage --preset math500 --engine-type think --input data/aime.jsonl --output out/aime.jsonl
```

Programmatic use:

```python
import asyncio
from sage import process_query, load_preset

kwargs = load_preset("math500")            # judge prompt + config + best settings
out = asyncio.run(process_query(
    "What is 2+2? Put the final answer in \\boxed{}.",
    model_name="Qwen/Qwen3-8B-FP8",
    base_url="http://localhost:9090/v1",
    **kwargs,
))
print(out["output"])
```

## Best settings (as used in the paper)

`sage.config.BEST_SETTINGS` (loaded by every preset):

| setting | value | notes |
|---|---|---|
| hypotheses per stage `N` | 7 | temperature schedule 0.7 / 0.8 / 0.9 |
| optimization epochs `T` | 2 | initial + 2 rounds → 21 candidates total |
| group cap `m_min` | 1 | automatic best/worst group formation |
| judge temperature | 0.1 | low-temperature verification, `logprobs=20` |
| generator engine | `aug` | Qwen3 non-thinking; use `think` for reasoning mode |

Per-benchmark judge recipes live in `sage/config.py::PRESETS`. Verifiable tasks
(math) use a single **verification** aspect; open-ended preference tasks use
**multi-aspect** tagging (usefulness / completeness / non-repetition) with the
weights in `configs/alpacaeval.json`.

## Tests

```bash
python -m pytest tests/            # or: python tests/test_scoring.py
```

## License

MIT — see [LICENSE](LICENSE).
