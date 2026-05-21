# Figures

Source data and reproducible scripts for the dataset paper.

## What's here

| File | Source data | What it shows |
|---|---|---|
| `fig_accuracy_bars.{png,pdf}` | paper §6 Tables 3 & 4 | Per-model accuracy on FinAgent-Synth vs FinAgent-Curated |
| `fig_native_vs_react.{png,pdf}` | paper §6 Table 5 | Native vs ReAct on FinAgent-Synth (thinking vs non-thinking) |
| `fig_synth_vs_curated_scatter.{png,pdf}` | paper §6 Tables 3 & 4 | Per-model agreement between the two benchmarks |
| `fig_pipeline_funnel.{png,pdf}` | paper §5.1 Table 2 | Construction attrition 10,000 → 5,979 |
| `fig_synth_categories_radar.{png,pdf}` | per-model eval JSONLs (see below) | Per-category accuracy on FinAgent-Synth, 8 models |

## Scripts

- `make_figures.py` — generates the first four figures from data inlined in the script (matches paper tables).
- `make_radar.py` — generates the category-radar. Requires the per-model eval JSONLs (not committed).

## Reproducing `make_radar.py`

The radar needs:

- `eval-queries.txt` — list of 1,000 candidate test queries
- `11_check_no_tools.jsonl` — 5,803 valid filtered dataset items (the 729-query test set = the intersection of these two)
- `eval_sample_*_<model>.jsonl` — per-model agent outputs with `evaluation.correct` (or `output_matches_expected.is_correct`) for each query
- `seed-queries.jsonl` — 13 seed queries with `category` field, used to label each test query by nearest-seed TF-IDF similarity

These files live in the collaborator's `experiments.zip` under `exp/expand-10k/` and are not committed (200 MB+). To regenerate the radar, extract them into `papers/aiim_research/.tta_eval_extracted/` and run:

```bash
cd papers/emnlp2026/figures
python3 make_radar.py
```

The script will print per-model per-category accuracy then save the radar.

## Category labels

The seed file contains 4 unique categories (the §3.1 paper text mentions 5 categories but the seed set used during construction has only 4):

- Accounts Payable (AP) Analysis — 9 of 13 seeds (largest category by design)
- Data Integrity & Reconciliation — 2 seeds
- Financial Analysis (variance-focused) — 1 seed
- Revenue Recognition Analysis — 1 seed

Each of the 729 test queries is assigned to the category of its nearest seed by TF-IDF cosine similarity (computed without sklearn — manual TF-IDF over docs).

## Notes on accuracy reconciliation

Filtering the eval JSONLs by the 729-query test set produces accuracies close to (but not identical to) the paper's headline numbers:

| Model | Paper (§6 Table 3) | Recomputed | Δ |
|---|---:|---:|---:|
| GPT-5 | 68.9% | 69.5% | +0.6 |
| GPT-5-mini | 65.8% | 66.0% | +0.2 |
| o4-mini | 67.1% | 68.7% | +1.6 |
| GPT-4.1 | 62.4% | 60.8% | −1.6 |
| GPT-4.1-mini | 61.5% | 63.4% | +1.9 |
| Qwen3-30B-A3B | 50.5% | 49.2% | −1.3 |
| Qwen3-8B | 47.6% | 54.2% | +6.6 |
| Llama-3.1-8B | 21.9% | 27.0% | +5.1 |

The biggest gaps are on the open-source models, suggesting the paper might have used a different eval run or unanswered-query handling. For the radar's purposes, the per-category *relative* pattern is what matters and that holds.
