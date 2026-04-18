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

Tokens are stored in environment variables on the host — never hardcode them in files.

- **GitHub token**: stored in env var `GITHUB_TOKEN`
  - Usage: `git clone https://$GITHUB_TOKEN@github.com/bliskavets/aiim_research.git`
  - Push: configure remote with token in URL
- **HuggingFace token**: stored in env var `HF_TOKEN`
  - Usage: `export HF_TOKEN=$HF_TOKEN` before training (already set in Docker via -e flag)
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

## Key Findings Summary (as of exp_014)

| Exp | Model | Dataset | Method | Final Reward | Notes |
|-----|-------|---------|--------|-------------|-------|
| 001 | Llama-3.2-3B | GSM8K | GRPO | 3.0 | Baseline, format hacking |
| 002 | Llama-3.2-3B | GSM8K | GTPO/GRPO-S entropy | 0.0 / 3.0 | GTPO fails on GSM8K |
| 003 | Llama-3.2-3B | MATH-500 | GRPO | 0.62 | Collapse on small dataset |
| 004 | Llama-3.2-3B | MATH-500 | GTPO/GRPO-S entropy | 2.38 | Entropy = regularization |
| 005 | Llama-3.2-3B | GSM8K | GTPO/GRPO-S confidence | 2.375 / 0.0 | Confidence > entropy at token level |
| 006 | Llama-3.2-3B | GSM8K | GTPO-EMA / GRPO-S-EMA | 3.0 / 3.0 | **EMA breakthrough** |
| 007 | Llama-3.2-3B | MATH-500 | GTPO-EMA | 2.38 | Faster convergence |
| 008 | Qwen3-4B | GSM8K | GRPO | 3.0 | 6× faster, KL=0.006 |
| 009 | Qwen3-4B | GSM8K | GTPO-EMA | 3.0 | Format hacking at max_seq=2048 |
| 009b | Qwen3-4B | GSM8K | GTPO-EMA | 4.44 | max_seq=4096 helps |
| 010 | Qwen3-4B | GSM8K | EMA v2 (fixed) | - | Fixed z-score bug |
| 011 | Qwen3-4B | MATH-500 | EMA v2 | -2.5 | Clipping issue, need >4096 |
| 012 | Qwen3-4B | MATH-500 | GRPO | - | Baseline for Qwen3+MATH500 |
| 013 | Qwen3-4B | MATH-500 | EMA v2 | -0.875 | max_seq=4096, still clipping |
| 014 | Llama-3.2-3B | Big-Math | GRPO | -1.84 | Failed: too large dataset |

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
