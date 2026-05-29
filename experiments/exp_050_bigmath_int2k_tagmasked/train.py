"""
exp_049 — GRPO vs three confidence/entropy-shaped candidates on Big-Math int-2000
=================================================================================

One script, one config, one dataset, one seed. The ONLY thing that changes
between runs is the trainer class (`--method`). This guarantees the four runs
are directly comparable.

Methods
-------
  grpo              vanilla GRPOTrainer (baseline, cf. exp_014/exp_027)
  grpo_s_entropy    candidate A — seq-level entropy weighting  (exp_002)
  gtpo_conf         candidate B — per-token confidence bonus   (exp_005)
  gtpo_ema_flipped  candidate C — flipped EMA-confidence shaping (exp_026)

Shared hyperparameters
----------------------
Training hyperparameters are taken verbatim from exp_005
(Llama-3.2-3B-Instruct, lr 5e-6, bs 1 x grad_accum 4 x num_gen 4, 500 steps,
cosine, adamw_8bit, LoRA r=64). The only memory knob raised from exp_005 is
`gpu_memory_utilization` (0.9 -> 0.55) and `max_seq_length` (2048 -> 2560):
neither changes the optimization, they only fit the longer Big-Math
completions; chosen to match the proven exp_027/028 budget on this dataset.

Dataset: SynthLabsAI/Big-Math-RL-Verified, integer-answer filter, first 2000
in shuffled order (seed 3407) — identical to exp_027/028.

Reward funcs: the exp_026 family (format_exact, format_approximate,
answer_exact, answer_numeric), adapted to Big-Math's `problem`/`answer`
fields and negative integers (exp_028 plumbing). Same scoring tiers.

The O+/O- split for all three shaped methods is driven by the sign of the
standard GRPO advantage (reward_threshold=0.0), the exp_026 setting.
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
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2560,          # 512 prompt + 2048 completion (exp_027/028 budget)
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.55,  # memory-only knob (exp_005 used 0.9 @ 2048)
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
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,
    "max_steps": 500,
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
    "max_completion_tokens": 2048,
    "subset_size": 2000,
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
# DATASET — Big-Math integer-answer subset (identical to exp_027/028)
# =============================================================================

def is_integer_answer(example: dict) -> bool:
    raw = str(example.get("answer", "")).strip().replace(",", "")
    try:
        return float(raw) == int(float(raw))
    except (ValueError, OverflowError):
        return False


def normalize_integer(raw: str) -> str:
    return str(int(float(raw.strip().replace(",", ""))))


def prepare_dataset():
    ds = load_dataset(DATASET_CONFIG["name"], split=DATASET_CONFIG["split"],
                      token=os.environ.get("HF_TOKEN"))
    ds = ds.filter(is_integer_answer)
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
                    choices=["grpo", "grpo_s_entropy", "gtpo_conf", "gtpo_ema_flipped"])
    args_cli = ap.parse_args()
    method = args_cli.method
    reward_funcs = REWARD_FUNCS_FULL  # exp_050 always uses full reward set

    output_dir = f"/workspace/exp_050_bigmath_int2k_tagmasked/outputs_{method}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== exp_050 [{method}] — Big-Math int-2000, Llama-3.2-3B (tagmasked shaping) ===")
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
