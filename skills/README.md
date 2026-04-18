# Skills

Reusable procedures and scripts for experiment management.

## Files

| File | Purpose |
|------|---------|
| `new_experiment.md` | Checklist for creating a new experiment (read before each new exp) |
| `check_progress.md` | How to check training progress and report metrics |
| `run_experiment.sh` | Shell script to launch experiment in Docker with uv setup |
| `compare_experiments.py` | Plot reward/KL curves from multiple experiments on one chart |
| `plot_progress.py` | Parse a training log and generate quick dashboard figures |

## Quick Usage

```bash
# Launch experiment
bash skills/run_experiment.sh exp_015_my_new_exp train.py

# Compare multiple experiments
python skills/compare_experiments.py --experiments exp_001 exp_006 exp_009b --metric reward kl
python skills/compare_experiments.py --all --output all_experiments.png

# Plot progress from log
python skills/plot_progress.py experiments/exp_009b_qwen3_ema_gsm8k_4096/train_gtpo_ema.log
python skills/plot_progress.py experiments/exp_001_grpo_llama32_gsm8k/metrics.json
```
