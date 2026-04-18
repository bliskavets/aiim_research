# Skill: Check Experiment Progress

Use this skill when asked about current progress / metrics of a running or completed experiment.

## Steps

1. **Find latest metrics** from log file:
   ```bash
   # Last N logged steps
   grep "^{'loss'" experiments/exp_NNN_name/*.log | tail -20
   # Or: last line with reward
   grep "'reward'" experiments/exp_NNN_name/*.log | tail -5
   ```

2. **Extract key numbers** to report:
   - Current step / total steps
   - `reward` (main metric)
   - `rewards/reward_format_exact/mean`
   - `rewards/reward_answer_*/mean`
   - `kl`
   - `completions/clipped_ratio` (>0 = context overflow problem)

3. **Generate plots** if plot_metrics.py exists:
   ```bash
   cd experiments/exp_NNN_name && python plot_metrics.py
   ```

4. **Report in chat**:
   ```
   Step X/Y (Z%):
   - reward: A → current B (peak C @ step D)
   - format_exact: E
   - answer_numeric: F
   - KL: G
   - clipped_ratio: H
   ```

## Reading metrics.json

```python
import json
with open("metrics.json") as f:
    metrics = json.load(f)
# metrics is list of dicts with keys: step, reward, kl, etc.
steps = [m["step"] for m in metrics]
rewards = [m["reward"] for m in metrics]
```
