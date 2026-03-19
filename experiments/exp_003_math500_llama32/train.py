"""
GRPO + LoRA fine-tuning of Llama 3.2 3B on MATH-500
=====================================================
Based on exp_001 (GSM8K), adapted for HuggingFaceH4/MATH-500.

Changes vs exp_001:
  - Dataset: HuggingFaceH4/MATH-500 (500 examples, test split)
  - Reasoning tags: <think>...</think> <answer>...</answer>
  - Correct answer bonus: 5.0 (was 3.0)
  - Answer matching: exact string match (LaTeX-aware)
"""

# =============================================================================
# CONFIG
# =============================================================================

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2048,
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.9,
}

LORA_CONFIG = {
    "r": 64,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
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
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,
    "max_steps": 500,
    "save_steps": 250,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "output_dir": "/workspace/outputs_exp003",
}

REASONING_START = "<think>"
REASONING_END   = "</think>"
SOLUTION_START  = "<answer>"
SOLUTION_END    = "</answer>"

PRINT_EVERY_STEPS = 5

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your thoughts and the solution process out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your final answer between {SOLUTION_START}{SOLUTION_END}"
)

# =============================================================================
# IMPORTS
# =============================================================================

import re
import os

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# =============================================================================
# REGEX PATTERNS
# =============================================================================

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def normalize_answer(s: str) -> str:
    """Light normalization for LaTeX answers: strip whitespace."""
    return s.strip()

def prepare_dataset():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Raw dataset size: {len(dataset)}")

    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["problem"]},
        ],
        "answer": normalize_answer(str(x["answer"])),
    })

    return dataset

# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def reward_format_exact(completions, **kwargs):
    """
    +3.0 if response has correct <think>...</think><answer>...</answer> format.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 3.0 if match_format.search(response) is not None else 0.0
        scores.append(score)
    return scores


def reward_format_approximate(completions, **kwargs):
    """
    Partial credit for each expected tag present exactly once.
    +0.5 per correct tag, -1.0 per missing/duplicated tag. Max = +2.0
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        score += 0.5 if response.count(REASONING_START) == 1 else -1.0
        score += 0.5 if response.count(REASONING_END)   == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_START)  == 1 else -1.0
        score += 0.5 if response.count(SOLUTION_END)    == 1 else -1.0
        scores.append(score)
    return scores


_print_counter = 0

def reward_answer_exact(prompts, completions, answer, **kwargs):
    """
    Exact string match of the extracted answer vs ground truth.
      +5.0  exact match (after strip)
      +2.5  match after aggressive normalization (lowercase, spaces)
      -1.0  wrong answer
       0.0  no answer tag found
    Bonus raised to 5.0 (vs 3.0 in exp_001) to emphasize correctness.
    """
    global _print_counter

    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1).strip() if (m := match_format.search(r)) is not None else None
        for r in responses
    ]

    if _print_counter % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"][:100]
        print("=" * 60)
        print(f"[Step {_print_counter}] Q: {question}...")
        print(f"  GT: {answer[0]}")
        print(f"  Pred: {extracted[0]}")
        print("=" * 60)
    _print_counter += 1

    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue

        true_norm = normalize_answer(true_answer)

        # Exact match
        if guess == true_norm:
            scores.append(5.0)
        # Match after lowercasing and collapsing whitespace
        elif guess.lower().replace(" ", "") == true_norm.lower().replace(" ", ""):
            scores.append(2.5)
        else:
            scores.append(-1.0)
    return scores


REWARD_FUNCS = [
    reward_format_exact,
    reward_format_approximate,
    reward_answer_exact,
]

# =============================================================================
# MODEL SETUP
# =============================================================================

def load_model():
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

    return model, tokenizer


def compute_max_prompt_length(dataset, tokenizer):
    lengths = dataset.map(
        lambda x: {
            "tokens": tokenizer.apply_chat_template(
                x["prompt"],
                add_generation_prompt=True,
                tokenize=True,
            )
        },
        batched=True,
    ).map(lambda x: {"length": len(x["tokens"])})
    return max(lengths["length"])

# =============================================================================
# TRAINING
# =============================================================================

def build_trainer(model, tokenizer, dataset, max_prompt_length):
    training_args = GRPOConfig(
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
        max_completion_length=MODEL_CONFIG["max_seq_length"] - max_prompt_length,
        max_steps=TRAINING_CONFIG["max_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        report_to=TRAINING_CONFIG["report_to"],
        output_dir=TRAINING_CONFIG["output_dir"],
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=REWARD_FUNCS,
    )

    return trainer


def main():
    print("=== Exp 003: GRPO Llama-3.2-3B on MATH-500 ===")
    print(f"Tags: {REASONING_START}...{REASONING_END} {SOLUTION_START}...{SOLUTION_END}")
    print(f"Correct answer bonus: 5.0")

    print("\nLoading dataset...")
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")

    print("\nLoading model...")
    model, tokenizer = load_model()

    print("\nComputing max prompt length...")
    max_prompt_length = compute_max_prompt_length(dataset, tokenizer) + 1
    print(f"Max prompt length: {max_prompt_length}")

    print("\nBuilding trainer...")
    trainer = build_trainer(model, tokenizer, dataset, max_prompt_length)

    print("\nStarting GRPO training...")
    trainer.train()

    print("\nDone. Model saved to:", TRAINING_CONFIG["output_dir"])


if __name__ == "__main__":
    main()
