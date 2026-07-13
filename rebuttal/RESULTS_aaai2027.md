# SAGE AAAI-2027 strengthening: results log

Runs on a single NVIDIA H200 (143 GB), Qwen3-8B-FP8 served on vLLM 0.11.0
(`--max-model-len 32768`). MATH-500 graded by exact-answer equivalence (o3 via an
OpenAI-compatible endpoint); MMLU-Pro STEM graded by exact letter match (judge-free).
Raw per-problem outputs under `rebuttal/logs/` (gitignored; regenerate with the
commands below).

## Verifiable-task results (seed 42, N=500)

| Benchmark | Baseline | SAGE | Delta |
|-----------|----------|------|-------|
| MATH-500 (exact-match) | 83.8 (419/500) | 88.8 (444/500) | +5.0 |
| MMLU-Pro STEM (letter-match) | 71.0 (355/500) | 75.8 (379/500) | +4.8 |

Both are verifiable tasks where a self-preference bias cannot manufacture the gain,
so the consistent ~+5 point lift supports the paper's central claim. The MATH
baseline (83.8) reproduces the paper's reported 84.4 non-thinking baseline. SAGE MATH
(88.8) trails the paper's 92.0; this is a single seed and uses an o3 grader through
an OpenAI-compatible gateway, so it needs multi-seed CIs before any headline claim.

Caveats: single seed only (headline numbers still need >=3 seeds + bootstrap CI per
the plan). N=500 for MMLU-Pro STEM (not the full 2000).

## Harness fixes required to get correct numbers

1. **Non-thinking mode was not engaged.** `AugEngine` appended a bare ` /nothink` to
   raw `/v1/completions` prompts. Qwen3's non-thinking switch only works through the
   chat template, so the model emitted full chain-of-thought that truncated at
   max_tokens before `\boxed{}`; ~35 percent of MATH answers were scored wrong from
   truncation alone (baseline measured 51.4 instead of 83.8). Fixed by wrapping the
   prompt in the Qwen3 template with a prefilled empty `<think></think>` block and
   stopping at `<|im_end|>`, keeping the completions endpoint so SAGE keeps token
   logprobs.
2. **Math judge client.** `core/helpers.py` used the OpenAI-native Responses API;
   switched to a lazily-constructed client plus `chat.completions` JSON output so the
   judge runs through OpenAI-compatible gateways.
3. **BoN sampling temperature.** A2 generated BoN candidates at temperature 0 with
   n>1, which vLLM rejects (greedy) and which collapses the candidates; set to 0.7.
4. **MMLU-Pro SAGE batching.** `run_mmlu_pro_sage.py` evaluated serially; batched it
   with `asyncio.gather` like the baseline script (about 5x faster on one GPU).

## Reproduce

```bash
# vLLM server
vllm serve Qwen/Qwen3-8B-FP8 --host 0.0.0.0 --port 9090 \
  --max-model-len 32768 --gpu-memory-utilization 0.90

# MATH-500 baseline (N=500)
python experiments/a1_baseline_clarification/run_math500_baseline.py \
  --num-samples 500 --seed 42 --batch-size 16 --output-path logs/a1_baseline_full

# MATH-500 SAGE (m_min=1, 2 epochs x 7 gens)
python experiments/b2_mmin_ablation/run_mmin_sweep.py \
  --num-samples 500 --seed 42 --batch-size 16 --m-min-values 1 \
  --judge-prompt configs/math500_judge_prompt.txt \
  --judge-config configs/math500_judge_config.json \
  --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
  --output-path logs/sage_math_full_s42

# MMLU-Pro STEM baseline + SAGE (N=500)
python experiments/c3_mmlu_pro/run_mmlu_pro_baseline.py \
  --num-samples 500 --seed 42 --batch-size 16 --output-path logs/c3_mmlu_baseline_s42
python experiments/c3_mmlu_pro/run_mmlu_pro_sage.py \
  --num-samples 500 --seed 42 --batch-size 16 \
  --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
  --output-path logs/c3_mmlu_sage_s42
```
