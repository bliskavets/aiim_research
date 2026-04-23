"""
exp_031 — Flipped pure-proof GTPO-EMA on Big-Math integer-2000
==============================================================

Combines:
  - Dataset and hyperparameters of exp_027 (Big-Math int-2000, 500 steps,
    4 × 8 = 32 seqs/step, max_completion=2048).
  - Flipped pure-proof GTPO-EMA trainer from exp_026.
  - Binary O+/O- split driven by `reward_answer_exact >= 1.0`, analogous to
    exp_022's cache pattern but with a stricter threshold:
      O+ ∋ {+3.0 exact, +1.5 strip, +1.0 within-10%}
      O- ∋ {+0.5 within-20%, 0.0 no-format, -1.5 wrong}

Flipped shaping (from exp_026):
    O+: bonus_{i,t}   = (1/EMA(C)_{i,t} / Σ_{k∈O⁺_t} 1/EMA(C)_{k,t}) · d_t
    O-: penalty_{j,t} = (EMA(C)_{j,t}   / Σ_{k∈O⁻_t} EMA(C)_{k,t})   · h_t
"""

import re
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# CONFIG — mirrors exp_027 exactly except the trainer
# =============================================================================

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2560,
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.55,
}

LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 64,
    "use_gradient_checkpointing": "unsloth",
    "random_state": 3407,
}

TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "num_generations": 8,
    "max_steps": 500,
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "output_dir": "/workspace/outputs_exp031",
    "bf16": True,
    "fp16": False,
}

DATASET_CONFIG = {
    "name": "SynthLabsAI/Big-Math-RL-Verified",
    "split": "train",
    "max_prompt_tokens": 512,
    "max_completion_tokens": 2048,
    "subset_size": 2000,
    "shuffle_seed": 3407,
}

FLIPPED_CONFIG = {
    "alpha1": 0.3, "alpha2": 0.7, "lam": 0.9, "top_k": 20,
}
ANSWER_EXACT_THRESHOLD = 1.0     # O+ iff reward_answer_exact >= 1.0

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
from trl import GRPOConfig

from src.gtpo_ema_flipped_trainer import GTPOEMAFlippedTrainer
from src.reward_cache import _CACHE


match_format = re.compile(
    rf"^[\s]{{0,}}{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(SOLUTION_START + r".*?([-\d\.,]+)",
                           flags=re.MULTILINE | re.DOTALL)


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
# REWARDS
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
    """
    Standard multi-level correctness score; ALSO stashes the per-sequence
    boolean O+ mask into reward_cache for the trainer.
    """
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

    _CACHE.set(scores, threshold=ANSWER_EXACT_THRESHOLD)
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


REWARD_FUNCS = [reward_format_exact, reward_format_approximate,
                reward_answer_exact, reward_answer_numeric]


def main():
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    print("=== Exp 031: Flipped GTPO-EMA on Big-Math (integer, 2000 examples) ===")
    print(f"  O+/O- threshold: reward_answer_exact >= {ANSWER_EXACT_THRESHOLD}")
    print(f"    O+ = {{exact=3.0, strip=1.5, within-10%=1.0}}")
    print(f"    O- = {{within-20%=0.5, no-format=0.0, wrong=-1.5}}")

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

    args = GRPOConfig(
        learning_rate=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        optim=TRAINING_CONFIG["optim"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        num_generations=TRAINING_CONFIG["num_generations"],
        max_prompt_length=max_prompt_length,
        max_completion_length=DATASET_CONFIG["max_completion_tokens"],
        max_steps=TRAINING_CONFIG["max_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        report_to=TRAINING_CONFIG["report_to"],
        output_dir=TRAINING_CONFIG["output_dir"],
        bf16=TRAINING_CONFIG["bf16"],
        fp16=TRAINING_CONFIG["fp16"],
    )

    trainer = GTPOEMAFlippedTrainer(
        model=model, tokenizer=tokenizer, args=args,
        train_dataset=dataset, reward_funcs=REWARD_FUNCS,
        **FLIPPED_CONFIG,
    )

    print("Starting GTPO-EMA-Flipped (Big-Math) training...")
    print(f"  num_generations = {TRAINING_CONFIG['num_generations']}")
    print(f"  batch_size      = {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  sequences/step  = {TRAINING_CONFIG['num_generations'] * TRAINING_CONFIG['per_device_train_batch_size']}")
    trainer.train()
    print("Done. Saved to:", TRAINING_CONFIG["output_dir"])


if __name__ == "__main__":
    main()
