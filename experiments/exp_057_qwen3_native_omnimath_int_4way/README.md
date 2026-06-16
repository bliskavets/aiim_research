# exp_057 — Qwen3-4B native format, 4 methods on Omni-MATH integer subset

Exact re-run of **exp_055** (same model, same 4 methods, same hyperparameters,
same Qwen3 native `<think>...</think>` + `\boxed{}` format) on a **different,
harder dataset**: [KbsdJames/Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH),
restricted to its integer-answer subset.

## ⚠️ CRITICAL FINDING (2026-06-12) — the per-token shaping was NOT being applied

During this experiment the grpo and gtpo_ema_flipped reward curves were
suspiciously identical. An audit (`diagnose_shaping.py`, `diagnose_real.py`,
`tests/test_shaping_diagnostics.py`) found the shaping never ran:

1. **unsloth bypasses `_compute_loss`.** The shaping trainers
   (`GRPOSTrainer` / `GTPOConfTrainer` / `GTPOEMAFlippedTrainer`) inject their
   per-token shaping by overriding **`_compute_loss`**. But unsloth replaces
   `trl.GRPOTrainer` with a compiled `_UnslothGRPOTrainer` whose `compute_loss`
   is **self-contained and never calls `_compute_loss`**. HF Trainer calls
   `compute_loss`, so the shaping override is dead code. Proof: (a)
   `GTPOEMAFlippedTrainer.compute_loss.__qualname__ == _UnslothGRPOTrainer.compute_loss`;
   (b) the real gtpo_ema_flipped run logged **zero** `gtpo_ema_flipped/mean_ema`
   metrics. **gtpo_ema_flipped @921 and grpo @492 were BOTH plain GRPO** —
   hence the identical curves.
2. **`self.top_k` collision.** `GRPOTrainer.__init__` overwrites the shaping
   trainers' `self.top_k` to `None` (vLLM sampling top-k), which crashes the
   shaping the moment it runs. Fixed: shaping params set after `super().__init__`.
3. **`_compute_loss` was stale vs trl 0.23.1 — now FIXED by injection.** The
   hand-written loss recomputed logps on a grid incompatible with the stored
   old/ref (left-pad), and a memory-safe grad-logps can't be recomputed by hand
   (full logits with grad OOMs ~28 GB; `_get_per_token_logps_and_entropies`
   returns detached logps). **Fix:** each shaping trainer now computes the
   per-token shaped advantage in `compute_loss` and **injects** it (as a 2-D
   `advantages` tensor, left-padded to the loss grid) into unsloth's compiled
   loss, which owns the memory-efficient chunked gradient (`src/shaped_loss.py`).
   `grpo_compute_loss` consumes a 2-D advantages tensor per-token natively.
   GRPO-S injects a seq-level (B,) advantage. **Verified live** (2026-06-12):
   each method now logs its `<method>/...` shaped metrics, gradients flow
   (grad_norm ~0.03–0.05), no OOM (peak ~132 GB/143). GRPO-S now uses real
   per-token entropy (the chunked helper returned None → constant before).
4. **Method-design caveat (independent of the stack).** `_znorm_over_active`
   centers each polarity at 0 and washes out `alpha1/alpha2` (0.9/0.1 vs 0.1/0.9
   ⇒ Δ 3e-5), and the shaped advantage correlates with the GRPO seq-advantage at
   only ~0.05 — i.e. the sequence-reward magnitude is discarded by construction.

The tag mask is **not** the culprit: it covers ~2.5% of a real completion
(single special-token ids only).

**Consequence:** the FIRST exp_057 "shaped" runs (grpo @492, gtpo_ema_flipped
@921 in the table below) were plain GRPO. This very likely also affects
**exp_055** (same stack; its "all methods within ±0.13" null is exactly what 4×
identical GRPO produces) and the wider exp_049→056 shaping arc on this unsloth
version — flagged for re-audit. **The fix (item 3) is now in and verified**; a
corrected 4-method sweep with shaping actually applied is being run — those
results supersede the table below.

## Why

exp_055 was an honest **null**: the easy Big-Math integer-2000 slice saturates
Qwen3-4B (~82% of the strict-answer ceiling at step 0), so per-token shaping had
no headroom — all 4 methods landed within ±0.13 reward. The standing
cross-experiment takeaway is "shaping only helps on weaker, non-saturated
baselines (Llama); on saturated Qwen3 it's a no-op."

Omni-MATH is competition-grade (olympiad/contest problems, difficulty 1–9.5,
mean ≈4.16). The GRPO baseline should be **far from saturated** here, so this is
the missing test: *given headroom on a strong model, does tag-masked shaping
(esp. `gtpo_conf`) finally beat plain GRPO?*

## Hypothesis

- **H1:** the GRPO baseline does NOT saturate on Omni-MATH (reward well below the
  ~+7 practical ceiling), unlike exp_055.
- **H2:** if H1 holds, `gtpo_conf` (the most reliable shaped variant) opens a
  measurable gap over GRPO — the result exp_055 couldn't show. Honest null is
  still a possible (and reportable) outcome if Omni-MATH is so hard that reward
  stays near the floor (no within-group advantage signal to shape).

## Setup

Identical to exp_055 except the dataset row.

| field | value |
|---|---|
| model | Qwen/Qwen3-4B |
| max_seq_length | 6656 (512 prompt + 6144 completion) |
| gpu_memory_utilization | 0.40 |
| LoRA | r=64, α=64, 7 modules (q,k,v,o,gate,up,down) |
| **dataset** | **KbsdJames/Omni-MATH, integer-answer subset (1971 problems), shuffled seed 3407, all kept** |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| num_generations | 8 |
| max_steps | 1000 (stop early once plateau is visible) |
| lr / sched | 5e-6 cosine, warmup 0.1, wd 0.1, adamw_8bit |
| seed | 3407 (LoRA init, dataset shuffle, GRPOConfig) |

### Dataset construction (integer-answer filter)

Omni-MATH ships a single `test` split of 4428 problems with columns
`domain, difficulty, problem, solution, answer, source`. Answers are
competition-style, often LaTeX (`\boxed{60}`, `$30`, `1,700`, or non-integer
like `1 + \lceil n/2 \rceil`, `\frac{1}{2}`, `2\sqrt{3}`).

`is_integer_answer` keeps only answers that reduce to a plain signed integer
after a **minimal, safe** normalization (`train.py:_clean_integer`):
1. unwrap a single surrounding `\boxed{...}`,
2. strip surrounding inline-math `$`,
3. collapse thousands-grouped commas only (`1,700 → 1700`; **not** a European
   decimal like `3,7`, which is rejected),
4. keep iff the result matches `-?\d+`.

Result: **1971 / 4428** integer-answer problems. The `subset_size` cap (2000) is
larger than 1971, so the whole integer subset is used (shuffle just fixes order).

Difficulty of the kept subset: min 1.0, max 9.5, mean ≈4.16
(hist by rounded level: 1:89, 2:398, 3:15, 4:612, 5:596, 6:135, 7:54, 8:42, 9:26, 10:4).
Top domains: Algebra 917, Number Theory 436, Discrete Math 410, Applied 365,
Geometry 351.

### Format (Qwen3 native) — unchanged from exp_055

System prompt: *"Solve the problem step by step. Put your final integer answer
inside \boxed{}, like \boxed{42}."* `apply_chat_template` default leaves
`enable_thinking=True`, so the model emits `<think>...</think>` then the answer.

Three reward components (`train.py`):
- `reward_format_thinking`: +2.5 one matched `<think>...</think>` pair / +1.5 no thinking / -2.0 mismatched
- `reward_answer_boxed`: +3.0 correct integer in `\boxed{}` / -1.5 wrong / 0.0 no boxed (answer region is post-`</think>`; unclosed `<think>` → no answer reward)
- `reward_answer_numeric`: +1.5 correct last number after `</think>` / -0.5 wrong / 0.0 none

Practical max reward ≈ +7.

### Tag mask (per-token shaping trainers) — unchanged

`gtpo_conf` / `gtpo_ema_flipped` mask per-token shaping off on the 4 Qwen3-native
special tokens `<think>`, `</think>`, `<|im_start|>`, `<|im_end|>` (replaced by
the seq-level GRPO advantage there). No-op for `grpo` and `grpo_s_entropy`.

## Infra (this machine)

Runs **natively in a uv venv** — this box is itself an unprivileged docker
container with the GPU passed through, so the docker-wrapped launch from exp_055
(`run_055.sh`) is not usable here. Validated stack:
`unsloth 2026.3.7 · trl 0.23.1 · torch 2.9.1+cu128 · vllm 0.16.0 · transformers 4.57.6 · numpy 2.2.6`
on **NVIDIA H200 NVL 143 GB**. (Driver supports CUDA 12.8 max → torch/vllm pinned
to cu128 builds; default cu130 wheels error with "driver too old".)

## How to run

All four methods sequentially:

```bash
HF_TOKEN=<token> bash experiments/exp_057_qwen3_native_omnimath_int_4way/run_057.sh \
  > experiments/exp_057_qwen3_native_omnimath_int_4way/run_057.console.log 2>&1
```

Single method (venv active, `HF_TOKEN` exported):

```bash
source /root/aiim/venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_TOKEN=<token>
cd experiments/exp_057_qwen3_native_omnimath_int_4way
python train.py --method {grpo|grpo_s_entropy|gtpo_conf|gtpo_ema_flipped}
```

## Results (corrected — shaping actually applied)

Last-50-step averages. The shaping bug (see above) is fixed; every shaped run
below logs its `<method>/*` metrics (shaping confirmed live, grad flowing).
Methods stopped at a visible reward plateau. `grpo` is the shared baseline (its
path is unaffected by the fix, reused from its @492 run). Plot:
`figures/exp057_progress.png`.

| method | steps | reward L50 | boxed L50 | numeric L50 | format L50 | KL | clip% | Δreward vs grpo |
|---|---|---|---|---|---|---|---|---|
| **grpo** (baseline)         | 492  | **+2.56** | +1.27 | +0.63 | +0.65 | 0.010 | 46% | — |
| grpo_s_entropy              | 1000 | +1.63 | +1.09 | +0.53 | +0.02 | 0.002 | 60% | **−0.93** |
| gtpo_ema_flipped (tagmask)  | 822  | +0.36 | +0.70 | +0.34 | −0.68 | 0.005 | 74% | **−2.20** |
| gtpo_conf (tagmask)         | 1000 | −0.17 | +0.56 | +0.27 | −1.00 | 0.021 | 80% | **−2.73** |

> ⚠️ **CONFOUND found 2026-06-16 (mask was incomplete) — these shaped numbers
> are not final.** The tag mask only covered the single-token tags `<think>`,
> `</think>`, `<|im_start|>`, `<|im_end|>`. But `reward_answer_boxed` trains the
> model on `\boxed{N}`, and `\boxed{` tokenizes to a **3-token substring**
> `['\\','boxed','{']` (+ `}`) — which was **NOT masked**, so the per-token
> shaping distorted exactly those answer-format control tokens (the failure mode
> the mask exists to prevent). Fixed in `train.py` (mask now includes `\boxed{`
> and `}`; `build_tag_mask` already handles multi-token substrings; the answer
> digits stay shaped as content). **gtpo_conf / gtpo_ema_flipped need a re-run
> with the corrected mask** before the negative is final. `grpo` (no shaping)
> and `grpo_s_entropy` (seq-level, mask is a no-op) are unaffected.

**Headline (PRELIMINARY, mask-confounded): with the per-token / entropy shaping
applied, every shaped method UNDERPERFORMED the plain-GRPO baseline on
Qwen3-4B / Omni-MATH** — from a mild lag (grpo_s_entropy) to active degradation
(gtpo_conf: reward climbs to +0.9 early then collapses to −0.27; loses format,
clip 0.69→0.84). grpo alone climbs cleanly +0.64→+2.56. Step-matched over
1..492 the ranking is identical (grpo +1.54 ≫ grpo_s +0.85 > gtpo_ema +0.66 >
gtpo_conf +0.36).

**Why** — consistent with the design caveat (item 4 above): `_znorm_over_active`
makes the per-token shaped advantage zero-mean per polarity and ~uncorrelated
(corr ≈ 0.05) with the GRPO seq-advantage, so the shaping injects a
reward-misaligned gradient that drags the policy off the reward signal (and on
no-signal steps, where plain GRPO is silent, the shaping still pushes on
confidence — visible as `grad_norm ≈ 0.05` vs grpo's ≈ 5e-6).

**This recontextualizes the shaping arc.** The earlier "shaping helps" results
(exp_050 win, exp_055 null, etc.) were produced on this unsloth stack where the
shaping was silently bypassed — i.e. plain GRPO. exp_055's "all 4 within ±0.13"
is exactly 4× identical GRPO. The whole exp_049→056 arc should be re-audited the
same way (check logs for `<method>/*` metrics). See `SHAPING_BYPASS_BUGFIX.md`.

Caveat: intermittent KL spikes (178 @ grpo_s step763, 396 @ gtpo_conf ~step450;
none in gtpo_ema) — RL instability from the reward-misaligned signal, not a
systemic fix bug (degradation is gradual and systemic, not spike-driven).

**grpo baseline (492 steps):** reward +0.64 → +2.56, peak rolling-20 +3.15 @ 368;
non-saturated (ceiling ~+7) — the headroom exp_055 lacked. So the dataset premise
held; the answer is just that shaping doesn't exploit the headroom — it hurts.

## Files

```
README.md                      this file
requirements.txt               numpy<2.3 overlay
run_057.sh                     native (no-docker) launcher, 4 methods sequential
plot_metrics.py                4-metric grid
plot_reward_dynamics.py        total-reward overlay (rolling-20)
plot_answer_boxed_dynamics.py  strict-answer reward overlay
train.py                       method-switch trainer, Omni-MATH integer filter, Qwen3 native format, 3 rewards, tag-mask
src/                           shaping trainers + format_tag_mask (verbatim from exp_055)
tests/test_methods.py          6 shaping correctness tests
```
