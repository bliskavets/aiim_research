# HANDOFF — exp_055 replication & context (read me first)

You are a Claude Code session picking up the aiim_research project on a (possibly
different) machine. This file is written by the Claude session that built and ran
exp_055 so you can reproduce it and act with the same context. Read it top to
bottom; it is self-contained.

---

## 0. Who you're working with

- ML researcher, works on RL fine-tuning of LLMs (GRPO/GTPO reward-shaping research).
- Communicates in **Russian, very terse** ("запушь", "как там прогресс?", "построй график"). Mirror that: short, lowercase, concrete file paths and numbers, no decorative prose, no LLM-boilerplate. Honest about negative results.
- Delegates execution. When they say "запускай"/"построй"/"сделай", do it end-to-end, commit, push, report. Don't ask permission for reversible local actions; do confirm before destructive/shared-state actions.
- Deliverables (plots, READMEs, decks) must not "look auto-generated": real numbers, honest negatives, terse engineer-in-lab-notebook voice.

## 1. The research, in one paragraph

We inject per-token reward shaping into GRPO RL fine-tuning to see if it beats a plain GRPO baseline. Three shaped methods, all implemented as TRL `GRPOTrainer` subclasses overriding `_compute_loss`:
- **GRPO-S** (`grpo_s_entropy`): sequence-level entropy weighting of the advantage.
- **GTPO-Conf** (`gtpo_conf`): per-token bonus from top-k confidence `C = -mean_topk(log p)`.
- **GTPO-EMA-flipped** (`gtpo_ema_flipped`): per-token EMA(λ=0.9)-smoothed confidence, O+/O- advantage-sign split (α1=0.9, α2=0.1).
The per-token methods also apply a **tag-mask**: on structural format tokens the per-token shaped advantage is replaced by the seq-level GRPO advantage, so shaping doesn't distort the gradient on highly-peaked control tokens.

## 2. What exp_055 is and what it found

4 methods (grpo + the 3 shaped) on Big-Math-RL-Verified integer-answer subset, first 2000 (seed 3407), Qwen3-4B, **Qwen3 native format** (`<think>...</think>` + `\boxed{}`). See README.md for the full config + results table.

**Finding:** all methods land within ±0.13 reward — the easy subset saturates Qwen3-4B (~82% of strict-answer ceiling at step 0), so shaping has no headroom. Honest null result. (gtpo_ema_flipped was skipped; exp_054 already showed it ties/loses on Qwen3 native.)

## 3. The research arc (why exp_055 exists)

- exp_049/050 (Llama-3.2-3B, Big-Math int-2000): tag-masked shaping made GTPO-Conf and GTPO-EMA-flipped beat the GRPO baseline (Δ +0.40/+0.46). Headline win. Key mechanism: per-token shaping was distorting gradients on format tokens; masking it there let format learning proceed (exact_top 0.06→0.30).
- exp_051 (first Qwen3 port): **confounded** — used custom `<start_working_out>`/`<SOLUTION>` tags while `apply_chat_template` defaulted `enable_thinking=True`, so Qwen3 emitted its native `<think>` AND tried our format → fought itself, heavy completion clipping.
- exp_052/053 (Qwen3, harder Big-Math subsets via `llama8b_solve_rate` filter): mixed; per-token shaping ≈ no-op or slight regression on Qwen3.
- exp_054 (Qwen3 NATIVE format, extra-hard subset): the fix — stop fighting Qwen3, use `<think>` + `\boxed{}`. gtpo_ema_flipped lagged grpo step-matched.
- **exp_055** (this one): exp_054's native format applied back to the EASY int-2000 slice, all 4 methods → saturation, null result.
- exp_056 (Search-R1, in progress on the origin machine): same shaping toolkit ported to retrieval-augmented multi-turn QA.

Cross-experiment takeaway: tag-masked **gtpo_conf** is the only shaping that consistently helped, and only on **weaker, non-saturated baselines** (Llama). On strong/saturated Qwen3 baselines shaping has no room to act.

## 4. Infrastructure (CRITICAL to replicate exactly)

- **Hardware:** single NVIDIA A100 80GB.
- **All training runs in the `unsloth/unsloth` docker image.** Never run training on the host.
- Launch pattern (see `run_055.sh`): `docker run --rm --gpus all --entrypoint /bin/bash --user root -v /mnt/data:/mnt/data -v <expdir>:/workspace/<exp> -e HF_TOKEN=... unsloth/unsloth -c '...'`
- **Python env inside the container:** `source /opt/venv/bin/activate` then overlay `uv pip install -r requirements.txt` (numpy<2.3) and `uv pip install --no-deps unsloth==2026.3.7 unsloth_zoo`. DO NOT use `uv venv --system-site-packages` — on this image it resolves the base interpreter as `/usr/bin/python3` and misses `/opt/venv`'s packages (symptom: `ModuleNotFoundError: packaging` at unsloth import).
- **Stack (observed):** unsloth 2026.3.7, trl 0.23.1, torch 2.10.0+cu128, vllm 0.16.1.dev0, transformers 4.57.6, numpy 2.2.6.
- Benign startup error to ignore: `gpt_oss_triton_kernels_moe.py ... No module named 'triton_kernels.routing'` (gpt-oss MoE path, unused for dense Qwen).
- **Credentials are NOT in the repo.** You need `HF_TOKEN` (HuggingFace, to pull Qwen/Qwen3-4B + the dataset) exported in the environment before launching. The operator has it in their Claude memory under `hf_token`; ask them for it or have them set `HF_TOKEN` in the shell. GitHub push uses a PAT embedded in `.git/config` remote URL — do not rewrite that URL.

## 5. Exact reproduction

```bash
cd <repo>/experiments/exp_055_qwen3_native_int2k_4way
# one method, inside the unsloth container (see run_055.sh for the full docker wrapper):
#   source /opt/venv/bin/activate
#   uv pip install -r requirements.txt --quiet
#   uv pip install --no-deps --quiet unsloth==2026.3.7 unsloth_zoo
#   python train.py --method grpo

# or the whole sweep from the host:
HF_TOKEN=<token> bash run_055.sh > run_055.console.log 2>&1
```

`train.py` is fully self-contained: config dicts at the top, `prepare_dataset()`, the 3 reward functions, the trainer factory, and `main()`. `src/` has the shaping trainers + `format_tag_mask.py`. `tests/test_methods.py` (run with pytest inside the container) validates the shaping math; expect 6 passing.

Step time on A100: ~17-65 s/step depending on completion lengths. 1000 steps ≈ a few hours per method. We stop early once the reward plateau is visible.

## 6. Pitfalls we already hit (don't repeat)

1. **SEED / group diversity (the big one).** GRPO needs the `num_generations` rollouts of a prompt to DIFFER. If you reuse our exp_056 multi-turn rollout code or write any custom vLLM sampling, do NOT pass a fixed per-request seed to `SamplingParams` — vLLM seeds deterministically per request, making all G samples byte-identical → zero within-group advantage → zero gradient (`frac_reward_zero_std=1.0` every step, `completions min==max==mean length`, `grad_norm ~1e-4`, flat reward). exp_055's `train.py` uses TRL's own generation (not affected), but this killed the first exp_056 run for 100 steps. Sanity-check early: `frac_reward_zero_std` should be < 1.0 on at least some steps, and `grad_norm` should spike (0.1-0.3) on steps with mixed reward.
2. **gpu_memory_utilization vs backward OOM.** vLLM grabs most of the card; long completions (esp. with retrieved context) OOM the training backward. Levers: lower `gpu_memory_utilization` (0.40→0.32), lower `num_generations`, lower `max_completion_tokens`, and set `PYTORCH_ALLOC_CONF=expandable_segments:True`.
3. **NVIDIA driver module mismatch** after host apt upgrades: new `--gpus all` containers fail with NVML "driver/library version mismatch". Fix = reload kernel modules (`modprobe -r nvidia_*` then re-add) or reboot.
4. **Don't `cat a b > big`** to assemble multi-part downloads — needs 2× peak disk. Use `mv a big; cat b >> big; rm b`.

## 7. Memory-system convention (please keep it up)

The operator runs a file-based memory at `<claude-config>/projects/-mnt-data-aiim-research/memory/` with a `MEMORY.md` index pointing to per-topic `.md` files. The canonical research-state file is `project_aiim_research.md`. **After analyzing any exp_NNN (logs, metrics, plots), update `project_aiim_research.md`** with the last experiment number, a results summary, and next steps. This is an enforced workflow rule. On a fresh machine that memory dir won't exist yet — this HANDOFF.md plus the repo's per-exp READMEs are your substitute; recreate a project memory file from them if the operator wants persistence.

## 8. Repo conventions

- Branch `main`, push direct (no PRs). Conventional-ish commit prefixes: `exp_NNN:`, `presentations:`, `add`, `fix`.
- `.gitignore` covers training artefacts: `outputs_*/`, `grpo_trainer_lora_model/`, `unsloth_compiled_cache/`, `*.log`, big data dirs. Commit source + plots + READMEs, not checkpoints/logs.
- Per-exp READMEs are the durable record; logs/outputs are disposable.
