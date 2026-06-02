"""
exp_053 — Qwen3-4B GRPO vs GTPO-EMA-flipped on extra-hard Big-Math subset
=========================================================================

Two methods only (grpo baseline + gtpo_ema_flipped with tag-masked
per-token shaping), on the hardest Big-Math integer-answer slice we can
extract: llama8b_solve_rate <= 0.125 (1/8). gtpo_ema_flipped was the
winner on exp_052 (Qwen3 hard subset, llama<0.3), so this is a stress
test of that result at a tougher difficulty + a bigger compute budget.

Compute config (user-spec):
  per_device_train_batch_size = 1
  gradient_accumulation_steps = 4     (global batch = 4 prompts/grad-update)
  num_generations = 16                (rollouts per prompt — group size)
  max_steps = 1000
  -> total seqs / grad-update = 1 * 4 * 16 = 64
  -> total prompt-visits     = 1 * 4 * 1000 = 4000 (each visited once on average)

Total compute ≈ 16x exp_052's per-step seq count; same per-step VRAM
(per_device unchanged) but gpu_memory_utilization dropped 0.55 -> 0.50
to give PyTorch more activation memory for ng=16 forward passes.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# CONFIG — training hyperparameters from exp_005 (shared by all four methods)
# =============================================================================

SEED = 3407

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
    "max_seq_length": 4096,          # 512 prompt + 3584 completion
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.50,  # exp_053: ng=16 -> more activation memory needed than exp_052 (ng=4)
}

LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 64,
    "use_gradient_checkpointing": "unsloth",
    "random_state": SEED,
}

# exp_005 TRAINING_CONFIG (verbatim) + seed, bf16, output set per-method below
TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,    # exp_053: global_bs = per_device * ga = 4 (user spec)
    "gradient_accumulation_steps": 4,    # exp_053
    "num_generations": 16,               # exp_053: 4x more rollouts per prompt vs exp_052
    "max_steps": 1000,                   # exp_053: 2x more updates vs exp_052
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "seed": SEED,
    "bf16": True,
    "fp16": False,
}

DATASET_CONFIG = {
    "name": "SynthLabsAI/Big-Math-RL-Verified",
    "split": "train",
    "max_prompt_tokens": 512,
    "max_completion_tokens": 3584,   # Qwen3 thinking-mode completions can be long
    "subset_size": 8000,             # exp_053: 2000 * 4 (per user spec) for larger prompt pool
    "shuffle_seed": SEED,
}

# Method-native shaping coefficients (intrinsic to each method, from its source exp)
SHAPING_CONFIG = {
    "grpo_s_entropy":   {"beta1": 1.0, "beta2": 0.1, "reward_threshold": 0.0},
    "gtpo_conf":        {"alpha1": 1.0, "alpha2": 0.1, "top_k": 20, "reward_threshold": 0.0},
    "gtpo_ema_flipped": {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20, "reward_threshold": 0.0},
}

REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"
PRINT_EVERY_STEPS = 10

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

match_format = re.compile(
    rf"^[\s]{{0,}}{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(SOLUTION_START + r".*?([-\d\.,]+)",
                           flags=re.MULTILINE | re.DOTALL)


# =============================================================================
# DATASET — Big-Math, integer-answer ∩ Llama-8B-hard subset (exp_053)
# =============================================================================
#   filter: integer answer AND llama8b_solve_rate < SOLVE_RATE_THRESHOLD
#   then: shuffle by seed, take first subset_size
#
# llama8b_solve_rate is stored as a string in the dataset; parse to float.
# Problems with missing solve rate are excluded.

SOLVE_RATE_THRESHOLD = 0.125  # exp_053: extra-hard subset, Llama-8B solves <=12.5% (1/8) of rollouts


def is_integer_answer(example: dict) -> bool:
    raw = str(example.get("answer", "")).strip().replace(",", "")
    try:
        return float(raw) == int(float(raw))
    except (ValueError, OverflowError):
        return False


def is_llama8b_hard(example: dict) -> bool:
    raw = example.get("llama8b_solve_rate")
    if raw is None:
        return False
    try:
        return float(raw) < SOLVE_RATE_THRESHOLD
    except (ValueError, TypeError):
        return False


def normalize_integer(raw: str) -> str:
    return str(int(float(raw.strip().replace(",", ""))))


def prepare_dataset():
    ds = load_dataset(DATASET_CONFIG["name"], split=DATASET_CONFIG["split"],
                      token=os.environ.get("HF_TOKEN"))
    n0 = len(ds)
    ds = ds.filter(is_integer_answer)
    n1 = len(ds)
    ds = ds.filter(is_llama8b_hard)
    n2 = len(ds)
    print(f"Filter: total={n0} -> integer-answer={n1} -> llama8b<{SOLVE_RATE_THRESHOLD}={n2}")
    ds = ds.shuffle(seed=DATASET_CONFIG["shuffle_seed"])
    ds = ds.select(range(min(DATASET_CONFIG["subset_size"], len(ds))))
    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["problem"]},
        ],
        "answer": normalize_integer(str(x["answer"])),
    })
    print(f"Dataset: {len(ds)} integer-answer examples "
          f"(Big-Math, shuffled seed={DATASET_CONFIG['shuffle_seed']})")
    return ds


# =============================================================================
# REWARDS — exp_026 family, adapted to Big-Math (exp_028 plumbing)
# =============================================================================

def reward_format_exact(completions, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) else 0.0
            for c in completions]


def reward_format_approximate(completions, **kwargs):
    scores = []
    for c in completions:
        r = c[0]["content"]
        s  = 0.5 if r.count(REASONING_START) == 1 else -1.0
        s += 0.5 if r.count(REASONING_END)   == 1 else -1.0
        s += 0.5 if r.count(SOLUTION_START)  == 1 else -1.0
        s += 0.5 if r.count(SOLUTION_END)    == 1 else -1.0
        scores.append(s)
    return scores


def reward_answer_exact(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := match_format.search(r)) else None
                 for r in responses]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0); continue
        if guess == true_answer:
            scores.append(3.0)
        elif guess.strip() == true_answer.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess.replace(",", "")) / float(true_answer)
                if   0.9 <= ratio <= 1.1: scores.append(1.0)
                elif 0.8 <= ratio <= 1.2: scores.append(0.5)
                else:                     scores.append(-1.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-1.5)
    return scores


_print_counter = 0


def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _print_counter
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := match_numbers.search(r)) else None
                 for r in responses]
    if _print_counter % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_print_counter}] GT:{answer[0]} | Pred:{extracted[0]}")
    _print_counter += 1
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0); continue
        try:
            gv = float(guess.strip().replace(",", ""))
            tv = float(true_answer.strip())
            scores.append(1.5 if gv == tv else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


REWARD_FUNCS_FULL = [reward_format_exact, reward_format_approximate,
                     reward_answer_exact, reward_answer_numeric]


# =============================================================================
# TRAINER FACTORY
# =============================================================================

def build_trainer(method, model, tokenizer, args, dataset, reward_funcs,
                  format_tag_patterns=None):
    common = dict(model=model, tokenizer=tokenizer, args=args,
                  train_dataset=dataset, reward_funcs=reward_funcs)
    if method == "grpo":
        # baseline has no per-token shaping → mask is a no-op; serves as control
        return GRPOTrainer(**common)
    if method == "grpo_s_entropy":
        from src.grpo_s_trainer import GRPOSTrainer
        # GRPO-S shaping is seq-level (mean entropy modifies the seq-level
        # advantage). there is no per-token bonus to mask, so format_tag
        # masking has no effect here; we still run it as a clean comparator.
        return GRPOSTrainer(**common, **SHAPING_CONFIG["grpo_s_entropy"])
    if method == "gtpo_conf":
        from src.gtpo_conf_trainer import GTPOConfTrainer
        return GTPOConfTrainer(**common, **SHAPING_CONFIG["gtpo_conf"],
                               format_tag_patterns=format_tag_patterns)
    if method == "gtpo_ema_flipped":
        from src.gtpo_ema_flipped_trainer import GTPOEMAFlippedTrainer
        return GTPOEMAFlippedTrainer(**common, **SHAPING_CONFIG["gtpo_ema_flipped"],
                                     format_tag_patterns=format_tag_patterns)
    raise ValueError(f"unknown method: {method}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["grpo", "gtpo_ema_flipped"])
    args_cli = ap.parse_args()
    method = args_cli.method
    reward_funcs = REWARD_FUNCS_FULL  # exp_053 uses full reward set, same as exp_050/051/052

    output_dir = f"/workspace/exp_053_qwen3_bigmath_xhard_grpo_vs_ema/outputs_{method}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== exp_051 [{method}] — Big-Math int-2000, Qwen3-4B (tagmasked shaping) ===")
    print(f"  seed={SEED}  max_seq={MODEL_CONFIG['max_seq_length']}  "
          f"steps={TRAINING_CONFIG['max_steps']}  "
          f"bs={TRAINING_CONFIG['per_device_train_batch_size']}x"
          f"ga{TRAINING_CONFIG['gradient_accumulation_steps']}x"
          f"ng{TRAINING_CONFIG['num_generations']}")

    dataset = prepare_dataset()

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        fast_inference=MODEL_CONFIG["fast_inference"],
        max_lora_rank=MODEL_CONFIG["lora_rank"],
        gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        use_gradient_checkpointing=LORA_CONFIG["use_gradient_checkpointing"],
        random_state=LORA_CONFIG["random_state"],
    )

    lengths = []
    for ex in dataset:
        toks = tokenizer.apply_chat_template(ex["prompt"],
                                             add_generation_prompt=True,
                                             tokenize=True)
        lengths.append(len(toks))
    lengths.sort()
    p99_len = lengths[int(0.99 * len(lengths))]
    max_prompt_length = min(p99_len + 1, DATASET_CONFIG["max_prompt_tokens"])
    print(f"Max prompt length (99%, capped): {max_prompt_length}")

    grpo_args = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=DATASET_CONFIG["max_completion_tokens"],
        output_dir=output_dir,
        **TRAINING_CONFIG,
    )

    # Build format-tag token-id patterns once. These are the 4 structural
    # tags from the system prompt. We feed both bare and " <tag>" variants
    # so BPE merging with a leading space is covered.
    from src.format_tag_mask import encode_tag_patterns
    format_tag_patterns = encode_tag_patterns(
        tokenizer,
        [REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END],
    )
    print(f"[tagmask] {len(format_tag_patterns)} format-tag patterns:")
    for pat in format_tag_patterns:
        print(f"           {pat}  -> {tokenizer.decode(pat)!r}")

    trainer = build_trainer(method, model, tokenizer, grpo_args, dataset,
                            reward_funcs, format_tag_patterns=format_tag_patterns)

    print(f"Starting [{method}] training...")
    print(f"  sequences/step = "
          f"{TRAINING_CONFIG['num_generations'] * TRAINING_CONFIG['per_device_train_batch_size']}")
    trainer.train()
    print(f"Done [{method}]. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
