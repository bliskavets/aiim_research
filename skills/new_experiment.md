# Skill: Create New Experiment

Use this skill when asked to create a new experiment.

## Checklist

1. **Determine next experiment number** from existing `experiments/` folders
2. **Create folder**: `experiments/exp_NNN_short_name/`
3. **Required files** — create ALL of these:
   - `README.md` — config table, hypothesis, expected results
   - `requirements.txt` — pip deps (base: `unsloth>=2026.3.7`, `trl>=0.23.0`, `torch>=2.9.0`, `datasets`)
   - `train*.py` — training script
   - `tests/test_*.py` — pytest unit tests for core logic (no actual training)
   - `plot_metrics.py` — reads metrics.json / .log and generates figures/

4. **Base your trainer on**: copy closest previous experiment's src/ and adapt
   - Current best trainer: `exp_010_ema_conf_fixed/src/ema_confidence_utils_v2.py`

5. **Before announcing run**: state in chat
   - Experiment summary (1-2 sentences)
   - Hypothesis being tested
   - Launch command

6. **Clear Docker caches** before run:
   ```bash
   for cid in $(docker ps -q); do
     docker exec $cid bash -c "rm -rf ~/.cache/huggingface ~/.cache/torch 2>/dev/null || true"
   done
   ```

7. **Launch in Docker**:
   ```bash
   docker run --rm --gpus all \
     -v /mnt/data:/mnt/data \
     -v /mnt/data/aiim_research/experiments/exp_NNN:/workspace/exp_NNN \
     -e HF_TOKEN=$HF_TOKEN \
     unsloth/unsloth bash -c "
       cd /workspace/exp_NNN &&
       uv venv /tmp/venv &&
       source /tmp/venv/bin/activate &&
       uv pip install -r requirements.txt &&
       python train.py 2>&1 | tee train.log
     "
   ```

8. **After run**: update README.md with actual results, push to GitHub

## Common Configs

### GSM8K
```python
DATASET = "openai/gsm8k"
TAGS = ("<start_working_out>", "<end_working_out>", "<SOLUTION>")
MAX_STEPS = 500
```

### MATH-500 (Qwen3-4B needs max_seq ≥ 6144)
```python
DATASET = "HuggingFaceH4/MATH-500"
TAGS = ("<think>", "</think>", "<answer>", "</answer>")
MAX_STEPS = 500
ANSWER_BONUS = 5.0
```

### Models
- `meta-llama/Llama-3.2-3B-Instruct` — fast (4s/step), needs more steps
- `Qwen/Qwen3-4B` — strong prior (converges fast), long completions (thinking mode)
  - GSM8K: max_seq=4096, ~15s/step
  - MATH-500: max_seq=6144+, ~30s/step
