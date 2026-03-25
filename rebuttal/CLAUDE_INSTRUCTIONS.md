# SAGE Rebuttal — Instructions for Claude on GPU Server

> **How to use this file:** Clone the repo, `cd rebuttal/`, read this file fully, then
> follow the steps in order. All paths below are relative to this `rebuttal/` directory.

---

## 1. Context: What Is This?

This is the rebuttal codebase for the paper **"SAGE: Self-Assessed Gradient-based Enhancement
for Test-Time Alignment of Large Language Models"** (ICML 2026 submission).

SAGE improves LLM outputs at inference time without retraining or an external reward model.
The model acts as its own judge via contrastive log-probability scoring, partitions candidates
into best/worst groups, generates a "textual gradient" (improvement recommendations), and
iteratively refines its output. Main results: Qwen3-8B on MATH-500 → 92.0%, AlpacaEval
win rate 74.9%, XSTest 93.5%.

**Current situation:** Four reviewers gave weak-reject scores citing specific empirical gaps.
Your job is to run the experiments that fill those gaps. The rebuttal draft with placeholders
is in `rebuttal_responses.txt`. After running experiments, fill in every `[PLACEHOLDER]`
in that file with actual numbers.

---

## 2. Reviewer Concerns You Are Addressing

| ID | Reviewer | Concern | Covered by |
|----|----------|---------|------------|
| YbZE-Q1 | R1 | Small models (<3B): does the textual gradient become noise? | B4 |
| YbZE-Q2 | R1 | Wall-clock vs. Best-of-N under same compute budget | A2 |
| aHH9-Q1 | R2 | Wall-clock SAGE vs. BoN+RM | A2 |
| aHH9-Q2 | R2 | Tested on 70B+? | C1 (32B proxy) |
| aHH9-W4 | R2 | AlpacaEval gains may be verbosity-driven | A3 |
| aHH9-Q3 | R2 | Self-congratulatory loop? | text response only |
| RD4w-Q1 | R3 | GPT-4.1 for SPO unfair? | text response only |
| RD4w-Q2 | R3 | Aspect sensitivity? | B3 |
| RD4w-Q3 | R3 | How to pick m_min? | B2 |
| RD4w-Q4 | R3 | Cost/time vs. baselines? | A2 |
| pDvu-W1 | R4 | Baseline discrepancy 84.4 vs 87.4 vs 97.4 | A1 |
| pDvu-W2 | R4 | Benchmarks too easy | A3 + C3 |
| pDvu-W3 | R4 | Outdated RM baseline | C2 |

Priorities: **A1 → A2 → A3** (these three cover 7/13 concerns and can be submitted alone).
Then B2, B3, B4. Then C1, C2, C3 if time allows.

---

## 3. Environment Setup

### Python dependencies
```bash
pip install vllm transformers datasets torch openai diskcache tqdm aiohttp \
            instruction-following-eval pydantic
```

For the updated RM baseline (C2):
```bash
pip install transformers accelerate  # already above, just noting it's needed
```

### Environment variables
Set these before running any script — they control the OpenAI-compatible client:

```bash
export OPENAI_API_KEY="your-key-here"   # only needed for A1 math judge (o3) and helpers.py
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Start the vLLM inference server
Before running any experiment, the vLLM server must be running in a separate terminal.

**Qwen3-8B (default, Plans A and B):**
```bash
vllm serve Qwen/Qwen3-8B-FP8 \
    --host 0.0.0.0 --port 9090 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92
```

**Qwen3-1.7B (Plan B4):**
```bash
vllm serve Qwen/Qwen3-1.7B-FP8 \
    --host 0.0.0.0 --port 9091 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7
```

**Qwen3-32B (Plan C1, needs 2× H100 or bfloat16 quantized on 1):**
```bash
vllm serve Qwen/Qwen3-32B \
    --host 0.0.0.0 --port 9092 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90
```

Wait for the server to print `Application startup complete` before running experiments.
Verify it's up: `curl http://localhost:9090/health`

---

## 4. Running Experiments

### Quick start — run everything in Plan A (recommended first)
```bash
bash run_experiments.sh A
```

Results land in `./logs/`. The script skips already-completed experiments (resume-safe).

### Override server settings if needed
```bash
IP=localhost PORT=9090 MODEL=Qwen/Qwen3-8B-FP8 bash run_experiments.sh A
```

### Run individual experiments
```bash
bash run_experiments.sh a1          # just A1
bash run_experiments.sh a2 a3       # A2 and A3 together
bash run_experiments.sh B           # all of Plan B (includes A)
bash run_experiments.sh all         # everything
```

---

## 5. Experiment-by-Experiment Details

### A1 — Baseline Clarification (`logs/a1_baseline/`)
**Goal:** Confirm the 84.4% figure is the correct non-thinking baseline.

```bash
python experiments/a1_baseline_clarification/run_math500_baseline.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 500 --seed 42 \
    --output-path logs/a1_baseline
```

**What to record:** The final accuracy printed at the end. It should be approximately 84.4%.
If it differs by more than 1 point, check the prompt format — the discrepancy with the Qwen3
tech report (87.4%) is expected and comes from different system prompt configurations.

**Placeholder to fill:** `[A1_BASELINE]` in `rebuttal_responses.txt`

---

### A2 — Latency Table (`logs/a2_latency/`)
**Goal:** Compare SAGE, BoN (no RM), and BoN+RM on wall-clock time AND accuracy on 100 problems.

```bash
python experiments/a2_latency_table/run_latency_comparison.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 100 --seed 42 \
    --n-candidates 7 --sage-epochs 2 \
    --judge-prompt configs/math500_judge_prompt.txt \
    --judge-config configs/math500_judge_config.json \
    --output-path logs/a2_latency
```

**What to record:** The summary table printed at the end. Collect:
- For each method: accuracy (%), mean time per problem (seconds), mean tokens generated

**Placeholders to fill:**
```
[A2_TABLE]        — paste the full markdown table
[A2_SAGE_TIME]    — e.g. "18.4"
[A2_BON_TIME]     — e.g. "11.2"
[A2_BONRM_TIME]   — e.g. "13.0"
[A2_SAGE_ACC]     — e.g. "91.0"
[A2_BON_ACC]      — e.g. "87.0"
[A2_BONRM_ACC]    — e.g. "88.0"
```

**Important note for C2:** When you later run C2 (updated RM), rerun A2 with the new RM
to get `[A2_BONRM_TIME]` and `[A2_BONRM_ACC]` with Skywork RM. For now, the A2 script
uses the older RM; that's fine — use C2 results to update those two values.

---

### A3 — IFEval (`logs/a3_ifeval_*/`)
**Goal:** Evaluate SAGE on IFEval — a judge-free benchmark with deterministic verification.
This directly refutes the verbosity accusation (aHH9-W4) and shows headroom (pDvu-W2).

Run all three variants:
```bash
# Baseline (single generation)
python experiments/a3_ifeval/run_ifeval_baseline.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 541 --seed 42 \
    --output-path logs/a3_ifeval_baseline

# Best-of-N
python experiments/a3_ifeval/run_ifeval_bon.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 541 --seed 42 --n-candidates 7 \
    --judge-prompt configs/ifeval_judge_prompt.txt \
    --judge-config configs/ifeval_judge_config.json \
    --output-path logs/a3_ifeval_bon

# SAGE
python experiments/a3_ifeval/run_ifeval_sage.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 541 --seed 42 \
    --judge-prompt configs/ifeval_judge_prompt.txt \
    --judge-config configs/ifeval_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/a3_ifeval_sage
```

**What to record:** Each script prints prompt-level and instruction-level accuracy at the end.
Also look for the per-category breakdown — specifically find the length-constrained category
(instructions like "respond in under 100 words"). Report those numbers separately.

**Placeholders to fill:**
```
[A3_SAGE_PROMPT]  — e.g. "84.7"
[A3_BASE_PROMPT]  — e.g. "79.3"
[A3_BON_PROMPT]   — e.g. "81.5"
[A3_SAGE_INSTR]   — e.g. "88.1"
[A3_BASE_INSTR]   — e.g. "84.2"
[A3_LEN_GAIN]     — SAGE minus baseline on length-constrained instructions specifically
```

---

### B2 — m_min Ablation (`logs/b2_mmin_ablation/`)
**Goal:** Show how performance varies with group size. m_min=1 is the degenerate single-pair case.

```bash
python experiments/b2_mmin_ablation/run_mmin_sweep.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 500 --seed 42 \
    --m-min-values "1,2,4,6,8" \
    --judge-prompt configs/math500_judge_prompt.txt \
    --judge-config configs/math500_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/b2_mmin_ablation
```

If you have 5 GPUs, run each m_min value in parallel on a separate GPU and separate port:
```bash
for m in 1 2 4 6 8; do
    PORT=$((9090 + m)) python experiments/b2_mmin_ablation/run_mmin_sweep.py \
        --m-min-values "$m" --port $((9090 + m)) \
        --output-path logs/b2_mmin_${m} &
done
wait
```

**Placeholder to fill:** `[B2_TABLE]` — a markdown table with columns m_min and MATH-500 accuracy.

---

### B3 — Aspect Sensitivity (`logs/b3_aspect_*/`)
**Goal:** Show SAGE is not brittle to the exact aspect formulation.

```bash
for aspect in default generic task_specific; do
    for benchmark in math500 ifeval; do
        python experiments/b3_aspect_sensitivity/run_aspect_ablation.py \
            --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
            --aspect-config $aspect --benchmark $benchmark \
            --num-samples 500 --seed 42 \
            --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
            --output-path logs/b3_aspect_${aspect}_${benchmark}
    done
done
```

**Placeholder to fill:** `[B3_TABLE]` — a 3×3 markdown table (aspect config × benchmark).
The numbers should show that the generic single-aspect config is slightly worse than default,
but task-specific matches or beats it. If results differ from this expectation, report honestly.

---

### B4 — Small Model (`logs/b4_small_*/`)
**Goal:** Empirically show what happens below 3B. Expected: modest gains on IFEval, minimal
gains on hard MATH-500 — not catastrophic failure.

First, make sure the Qwen3-1.7B vLLM server is running on port 9091 (see section 3).

```bash
python experiments/b4_small_model/run_small_model_math500.py \
    --ip localhost --port 9091 --model-name Qwen/Qwen3-1.7B-FP8 \
    --num-samples 500 --seed 42 \
    --judge-prompt configs/math500_judge_prompt.txt \
    --judge-config configs/math500_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/b4_small_math500

python experiments/b4_small_model/run_small_model_ifeval.py \
    --ip localhost --port 9091 --model-name Qwen/Qwen3-1.7B-FP8 \
    --num-samples 541 --seed 42 \
    --judge-prompt configs/ifeval_judge_prompt.txt \
    --judge-config configs/ifeval_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/b4_small_ifeval
```

**If Qwen3-1.7B is not available** on HuggingFace, use `Qwen/Qwen2.5-3B-Instruct` as fallback.

**Placeholders to fill:**
```
[B4_MATH_SAGE]    [B4_MATH_BASE]    [B4_MATH_BON]
[B4_IFEVAL_SAGE]  [B4_IFEVAL_BASE]
[B4_VARIANCE]     — a sentence describing the log-prob margin variance plot printed at the end
```

---

### C1 — Qwen3-32B (`logs/c1_qwen32b_*/`)
**Goal:** Show SAGE scales to a stronger model. Start the 32B server on port 9092 first.

```bash
python experiments/c1_qwen32b/run_qwen32b.py \
    --ip localhost --port 9092 --model-name Qwen/Qwen3-32B \
    --benchmark math500 --num-samples 500 --seed 42 \
    --judge-prompt configs/math500_judge_prompt.txt \
    --judge-config configs/math500_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/c1_qwen32b_math500

python experiments/c1_qwen32b/run_qwen32b.py \
    --ip localhost --port 9092 --model-name Qwen/Qwen3-32B \
    --benchmark ifeval --num-samples 541 --seed 42 \
    --judge-prompt configs/ifeval_judge_prompt.txt \
    --judge-config configs/ifeval_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/c1_qwen32b_ifeval
```

**Placeholders:** `[C1_32B_MATH]`, `[C1_32B_BASE]`

---

### C2 — Updated RM Baseline (`logs/c2_updated_rm_*/`)
**Goal:** Compare against a modern Qwen3-era reward model, not the outdated Llama3-based one.

```bash
python experiments/c2_updated_rm/run_updated_rm.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --benchmark math500 \
    --reward-model Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2 \
    --num-samples 500 --n-candidates 7 --seed 42 \
    --output-path logs/c2_updated_rm_math500

python experiments/c2_updated_rm/run_updated_rm.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --benchmark ifeval \
    --reward-model Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2 \
    --num-samples 541 --n-candidates 7 --seed 42 \
    --output-path logs/c2_updated_rm_ifeval
```

Note: The reward model runs on CPU or a second GPU — the script handles loading automatically.
If Skywork RM is unavailable, use `ArmoRM-Llama3-8B-v0.1` as fallback.

**Placeholder:** `[C2_RM_MATH]`

---

### C3 — MMLU-Pro STEM (`logs/c3_mmlu_pro_*/`)
**Goal:** Demonstrate gains on a broad, multi-domain, judge-free benchmark with meaningful headroom.

```bash
python experiments/c3_mmlu_pro/run_mmlu_pro_baseline.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 2000 --seed 42 \
    --output-path logs/c3_mmlu_pro_baseline

python experiments/c3_mmlu_pro/run_mmlu_pro_sage.py \
    --ip localhost --port 9090 --model-name Qwen/Qwen3-8B-FP8 \
    --num-samples 2000 --seed 42 \
    --judge-prompt configs/math500_judge_prompt.txt \
    --judge-config configs/math500_judge_config.json \
    --num-optimization-epochs 2 --number-of-gens-per-epoch 7 \
    --output-path logs/c3_mmlu_pro_sage
```

**Placeholders:** `[C3_MMLU_SAGE]`, `[C3_MMLU_BASE]`

---

## 6. Filling In the Rebuttal

Once experiments are done, open `rebuttal_responses.txt` and replace every `[PLACEHOLDER]`
with the actual number. To see all remaining placeholders:

```bash
grep -n '\[' rebuttal_responses.txt
```

For table placeholders like `[B2_TABLE]`, replace the entire token with a properly
formatted markdown table.

For the conditional in the C2 response:
- If SAGE > BoN+Skywork RM: replace `[matches / exceeds]` with `exceeds`, and remove
  the "slightly trails" sentence.
- If SAGE < BoN+Skywork RM: replace with `trails`, and keep the honest framing sentence.

---

## 7. Debugging Tips

**Script exits immediately / ImportError:**
```bash
# Run from the rebuttal/ directory so Python can find the core/ package:
cd /path/to/aiim_research/rebuttal
python experiments/a1_baseline_clarification/run_math500_baseline.py ...
```

**vLLM server connection refused:**
- Check the server is on the right port: `curl http://localhost:9090/v1/models`
- Check `--ip` and `--port` match what you started the server with

**Low accuracy on MATH-500 baseline (well below 84%):**
- Check `--model-name` matches the actually loaded model exactly
- The judge prompt format matters — use the configs in `configs/`

**OOM on 32B model:**
- Reduce `--gpu-memory-utilization` to 0.85, or add `--quantization bitsandbytes`
- Alternatively use `Qwen/Qwen3-32B-FP8` instead of bf16

**IFEval evaluation crashes (missing `instruction_following_eval`):**
```bash
pip install instruction-following-eval
# or from source:
git clone https://github.com/google-research/google-research
cd google-research && pip install -e instruction_following_eval
```

---

## 8. File Map

```
rebuttal/
├── CLAUDE_INSTRUCTIONS.md          ← you are here
├── rebuttal_responses.txt          ← rebuttal draft; fill in placeholders
├── run_experiments.sh              ← main runner script
├── experiment_plan.tex             ← detailed experiment rationale
├── core/                           ← evaluation library
│   ├── helpers.py                  ← math judge, file utils
│   ├── vllm_client.py              ← AugEngine (non-thinking), EngineNoThink
│   ├── math500_eval.py             ← MATH-500 evaluator
│   ├── ifeval_eval.py              ← IFEval evaluator (deterministic checker)
│   ├── alpaca_eval.py              ← AlpacaEval evaluator
│   └── xstest_eval.py              ← XSTest evaluator
├── experiments/
│   ├── sage/solver.py              ← core SAGE algorithm (used by all experiments)
│   ├── a1_baseline_clarification/
│   ├── a2_latency_table/
│   ├── a3_ifeval/
│   ├── b2_mmin_ablation/
│   ├── b3_aspect_sensitivity/
│   ├── b4_small_model/
│   ├── c1_qwen32b/
│   ├── c2_updated_rm/
│   └── c3_mmlu_pro/
└── configs/
    ├── math500_judge_prompt.txt    ← judge prompt for MATH-500 and MMLU-Pro
    ├── math500_judge_config.json   ← scoring config (correct/incorrect labels)
    ├── ifeval_judge_prompt.txt     ← judge prompt for IFEval
    └── ifeval_judge_config.json    ← scoring config (yes/no labels)
```

---

## 9. Priority Order If Time Is Short

1. **A2** — one run, answers 4 reviewer questions simultaneously
2. **A3 baseline** — fast single-generation run
3. **A3 SAGE** — the main new benchmark result
4. **A1** — validates the baseline number (very fast)
5. **B4** — fast at 1.7B, answers YbZE
6. **B2** — 5 runs but parallelizable
7. **B3** — 6 runs
8. **C2** — moderate cost, fixes pDvu-W3
9. **C1** — expensive, nice to have
10. **C3** — expensive, secondary
