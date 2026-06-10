# Experiments index — per-token reward shaping for GRPO

Research question: does per-token reward shaping (confidence / EMA-confidence /
sequence entropy), with structural-token masking, beat a plain GRPO baseline in
RL fine-tuning of LLMs? Each `experiments/exp_NNN_*/` has its own README with
full config + results; the tagmasked/shaping arc (exp_049→056) is summarized
below. Earlier experiments (exp_001–048) are GSM8K / MATH-500 / Big-Math method
development on Llama-3.2-3B and Qwen3-4B — see their folders.

## Shaping arc (exp_049 → exp_056)

| exp | model | data | format | methods | headline |
|---|---|---|---|---|---|
| 049 | Llama-3.2-3B | Big-Math int-2000 | custom tags | grpo + 3 shaped + numonly axis | no shaping beats grpo with full reward; format reward is load-bearing |
| 050 | Llama-3.2-3B | Big-Math int-2000 | custom tags | grpo + 3 shaped, **tag-mask** | **win**: gtpo_conf Δ+0.40, gtpo_ema_flipped Δ+0.46 vs grpo; exact_top 0.06→0.30 |
| 051 | Qwen3-4B | Big-Math int-2000 | custom tags | grpo + 3 shaped | **confounded** — custom tags fight Qwen3's native `<think>` (enable_thinking=True) |
| 052 | Qwen3-4B | Big-Math int ∩ llama8b<0.3 | custom tags | grpo + 3 shaped | harder slice; ranking inverts, 52% completion clipping |
| 053 | Qwen3-4B | Big-Math int ∩ llama8b<0.125 | custom tags | grpo + gtpo_ema_flipped, ng=16 | extra-hard; gtpo_ema_flipped lags grpo step-matched |
| 054 | Qwen3-4B | Big-Math int ∩ llama8b<0.125 | **Qwen3 native** | grpo + gtpo_ema_flipped | the format fix (`<think>`+`\boxed{}`); ema ties/loses |
| 055 | Qwen3-4B | Big-Math int-2000 (easy) | **Qwen3 native** | grpo + 3 shaped | **null**: all within ±0.13 reward — easy subset saturates Qwen3 (~82% ceiling) |
| 056 | Qwen3-4B | Search-R1 (NQ+HotpotQA + wiki-18 retrieval) | Qwen3 native + search tags | grpo + 3 shaped | retrieval-augmented multi-turn; **running** |

## Cross-experiment takeaway

- Tag-masked per-token shaping **only consistently helps on weaker, non-saturated baselines** (Llama exp_050). On strong/saturated Qwen3 baselines (exp_051/055) there's no headroom — shaping is a no-op.
- **gtpo_conf** is the most reliable shaped variant (wins exp_050 + exp_051-easy); **gtpo_ema_flipped** is model-sensitive (wins Llama, ties/loses Qwen3).
- The format reward acts as "training wheels": removing it (exp_049 numonly axis) collapses every method.
- Mechanism behind the exp_050 win: per-token shaping distorts gradients on highly-peaked structural tokens; masking shaping off there lets the format-learning signal through.

## Key reusable assets

- `experiments/exp_055_qwen3_native_int2k_4way/HANDOFF.md` — self-contained context to replicate the Qwen3 native-format 4-way on another machine.
- `experiments/exp_056_searchr1_qwen3_grpo_vs_shaped/` — Search-R1 port: multi-turn rollout (`src/searchr1_rollout.py`), retriever client (`src/retriever.py`), EM reward (`src/em_score.py`), `SearchR1GRPOTrainer`, retrieval server launcher (`retrieval/`).
- Shaping trainers (shared across exps): `src/grpo_s_trainer.py`, `src/gtpo_conf_trainer.py`, `src/gtpo_ema_flipped_trainer.py`, `src/format_tag_mask.py`.

## Infra (all experiments)

unsloth/unsloth docker on A100 80GB; `source /opt/venv/bin/activate` + `uv pip install --no-deps unsloth==2026.3.7 unsloth_zoo` overlay; HF_TOKEN from operator env (not committed). Presentations in `presentations/`, papers in `papers/`.
