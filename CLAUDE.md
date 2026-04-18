# CLAUDE.md — GRPO/GTPO Research Repository

## Project Overview

Research on per-token reward shaping in GRPO-based RL fine-tuning of LLMs.
Core idea: inject token-level reward signals (based on log probabilities / confidence / entropy)
into GRPO to improve training stability and sample efficiency.

Main methods developed:
- **GTPO** — Group Token-level Policy Optimization (per-token entropy/confidence-weighted advantages)
- **GRPO-S** — Sequence-level entropy/confidence-weighted advantages
- **EMA variant** — exponential moving average over confidence signal (λ=0.9), current best method

## Repository Structure

```
experiments/
  exp_NNN_short_name/   ← one folder per experiment
    README.md           ← required: experiment description + results
    requirements.txt    ← required: pip dependencies
    train*.py           ← training script(s)
    test_*.py OR tests/ ← required: pytest unit tests
    plot_metrics.py     ← required: metrics visualization script
    *.log               ← training logs
    figures/            ← saved plots
    src/                ← reusable modules (if experiment has custom logic)
    metrics*.json       ← saved metrics
skills/                 ← reusable procedures and Claude Code commands
```

## Credentials

Tokens are stored in Claude's persistent memory (check memory at session start).
Never hardcode tokens in committed files.

- **GitHub token**: stored in Claude memory as `github_token`
  - Usage: `git remote set-url origin https://<token>@github.com/bliskavets/aiim_research.git`
- **HuggingFace token**: stored in Claude memory as `hf_token`
  - Usage: set as `HF_TOKEN` env var before training
  - Or pass as `token=os.environ["HF_TOKEN"]` to `from_pretrained()`

## Infrastructure

- **Volume**: experiments run from `/mnt/data/aiim_research` (mounted volume, lots of space)
  - Main disk is small — never install large packages to main disk
- **Docker image**: `unsloth/unsloth` (unsloth>=2026.3.7)
- **Hardware**: NVIDIA A100 80GB PCIe
- **Workspace inside Docker**: `/workspace/`
- **Python env**: use `uv` — `uv venv && uv pip install -r requirements.txt`

## Running an Experiment

### Before each experiment run:
1. Clear Docker caches to free space:
   ```bash
   for cid in $(docker ps -q); do
     docker exec $cid bash -c "rm -rf ~/.cache/huggingface/hub ~/.cache/torch 2>/dev/null || true"
   done
   ```

2. Announce in chat:
   - Summary of the experiment
   - Launch command
   - Hypothesis being tested

### Launch command pattern:
```bash
docker run --rm --gpus all \
  -v /mnt/data:/mnt/data \
  -v /mnt/data/aiim_research/experiments/exp_NNN_name:/workspace/exp_NNN_name \
  -e HF_TOKEN=$HF_TOKEN \
  unsloth/unsloth bash -c "
    cd /workspace/exp_NNN_name &&
    uv venv /tmp/venv_expNNN &&
    source /tmp/venv_expNNN/bin/activate &&
    uv pip install -r requirements.txt &&
    python train.py 2>&1 | tee train.log
  "
```

## Required Files Per Experiment

Every experiment MUST contain:
1. `README.md` — description, config table, results table, observations
2. `requirements.txt` — all pip dependencies (base: unsloth, trl, torch, datasets)
3. `train*.py` — training script
4. `test_*.py` or `tests/test_*.py` — pytest unit tests for core functionality (no training)
5. `plot_metrics.py` — script to generate figures from metrics.json/log

## Experiment Naming Convention

Format: `exp_NNN_short_description`
- NNN = three-digit zero-padded number (001, 002, ..., 014, ...)
- Short description in snake_case

Sub-experiments: `exp_NNNb_...`, `exp_NNNc_...`

## Key Findings Summary (as of exp_016)

**Next experiment number: exp_017** (exp_015 was skipped)

| Exp | Model | Dataset | Method | Final Reward | Notes |
|-----|-------|---------|--------|-------------|-------|
| 001 | Llama-3.2-3B | GSM8K | GRPO | 3.0 | Baseline, peak 8.4@step169 |
| 002 | Llama-3.2-3B | GSM8K | GTPO/GRPO-S entropy | 0.0 / 3.0 | GTPO fails on GSM8K; GRPO-S ✅ |
| 003 | Llama-3.2-3B | MATH-500 | GRPO | 0.62 | Peak 10.0@150, collapses (small dataset) |
| 004 | Llama-3.2-3B | MATH-500 | GTPO/GRPO-S entropy | 2.38 / 2.38 | **GTPO works on MATH-500**; KL 10× lower |
| 005 | Llama-3.2-3B | GSM8K | GTPO/GRPO-S confidence | 2.375 / 0.0 | Token confidence > seq-level |
| 006 | Llama-3.2-3B | GSM8K | GTPO-EMA / GRPO-S-EMA | 3.0 / 3.0 | **EMA breakthrough** — both methods ✅ |
| 007 | Llama-3.2-3B | MATH-500 | GTPO-EMA / GRPO-S-EMA | 2.38 / 2.38 | Peak faster (steps ~97-104 vs ~150) |
| 008 | Qwen3-4B | GSM8K | GRPO | 3.0 | **6× faster convergence**, KL=0.006 |
| 009 | Qwen3-4B | GSM8K | GTPO-EMA / GRPO-S-EMA | 3.0 / 3.0 | EMA works with Qwen thinking mode |
| 009b | Qwen3-4B | GSM8K | GTPO-EMA + 4096ctx | 4.44 | max_seq=4096, no completion clipping |
| 010 | Qwen3-4B | GSM8K | EMA v2 (bug fix) | - | **Fixed z-score norm bug** destroying EMA signal |
| 011 | Qwen3-4B | MATH-500 | EMA v2 | -2.5 | alpha2=0.5; clipping issue |
| 012 | Qwen3-4B | MATH-500 | GRPO | - | Qwen3+MATH500 baseline |
| 013 | Qwen3-4B | MATH-500 | EMA v2 + 4096ctx | -0.875 | GRPO-S early stop |
| 014 | Llama-3.2-3B | Big-Math | GRPO + EMA variants | -1.84 | Multiple variants; needs curriculum |
| 016 | Llama-3.2-3B | Big-Math | GRPO (clean baseline) | - | Integer-filtered Big-Math (~1000 problems) |

## Key Insights for Future Experiments

1. **EMA-smoothed confidence (v2) is current best method** — no z-score normalization
2. **Qwen3-4B generates very long completions** (thinking mode) → need max_seq ≥ 6144 for MATH-500
3. **Big-Math dataset** is promising but needs more steps (≥1000) and possibly curriculum
4. **max_seq=4096 is better than 2048** for Qwen3 on GSM8K
5. **MATH-500 collapse** is dataset size problem — 500 examples = 1 epoch = collapse; need larger split
6. **Format hacking** occurs when format reward > answer reward weight; reduce format weight ratio

## Plotting Progress

```bash
# Plot single experiment
python experiments/exp_006_ema_confidence/plot_metrics.py

# Compare multiple experiments (see skills/compare_experiments.py)
python skills/compare_experiments.py --experiments exp_001 exp_006 exp_009b
```

When asked for progress: extract metrics from metrics.json or last N lines of *.log,
generate plots with plot_metrics.py, report key numbers (reward@current, peak_reward, KL, format).
