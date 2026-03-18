"""
GRPO + LoRA fine-tuning of Llama 3.2 3B on GSM8K
==================================================
Replicated from: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb

This script is structured to be easy to modify for research experiments:
  - CONFIG section at the top controls all major hyperparameters
  - Reward functions are clearly separated — modify or add new ones here
  - Training setup is at the bottom

GRPO (Group Relative Policy Optimization) overview:
  - Samples `num_generations` completions per prompt
  - Scores each with reward functions
  - Uses the group's mean/std to normalize rewards (no value model needed)
  - Updates the policy via PPO-style clipped objective
"""

# =============================================================================
# CONFIG — modify this section for experiments
# =============================================================================

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2048,
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,       # uses vLLM for generation
    "gpu_memory_utilization": 0.9,
}

LORA_CONFIG = {
    "r": 64,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # lora_alpha = r means effective scale = 1.0
    # increase alpha > r to boost LoRA contribution
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
    "num_generations": 4,         # how many completions to sample per prompt
    "max_steps": 500,
    "save_steps": 250,
    "max_grad_norm": 1.0,
    "report_to": "none",          # change to "wandb" to enable logging
    "output_dir": "outputs",
}

# Special tokens wrapping the chain-of-thought and final answer
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

# How many training steps between printing a sample completion to stdout
PRINT_EVERY_STEPS = 5

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
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

# Matches a correctly-formatted full response (reasoning block + solution tag)
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Extracts the first number from inside the <SOLUTION> tag (handles commas)
match_numbers = re.compile(
    SOLUTION_START + r".*?([\d\.,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def extract_hash_answer(text: str):
    """GSM8K stores the final answer after '####'. Extract and return it."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def prepare_dataset():
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })

    return dataset

# =============================================================================
# REWARD FUNCTIONS
# — Each function receives `completions` (list of generated message lists)
#   and optionally `prompts`, `answer` from the dataset.
# — Must return a list of floats (one score per completion).
# — Add new reward functions here and register them in `REWARD_FUNCS` below.
# =============================================================================

def reward_format_exact(completions, **kwargs):
    """
    +3.0 if the response perfectly matches the expected format.
    Zero otherwise.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 3.0 if match_format.search(response) is not None else 0.0
        scores.append(score)
    return scores


def reward_format_approximate(completions, **kwargs):
    """
    Partial credit for each expected token that appears exactly once.
    +0.5 per correct token, -1.0 per missing/duplicated token.
    Max = +2.0, useful for guiding format early in training.
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


def reward_answer_exact(prompts, completions, answer, **kwargs):
    """
    Rewards based on the text extracted from inside <SOLUTION>...</SOLUTION>.
      +3.0  exact string match
      +1.5  match after stripping whitespace
      +1.0  numeric value within 10% of ground truth
      +0.5  numeric value within 20% of ground truth
      -1.5  wrong answer or non-parseable
       0.0  format missing (no solution tag)
    """
    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1) if (m := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue

        score = 0.0
        if guess == true_answer:
            score = 3.0
        elif guess.strip() == true_answer.strip():
            score = 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if   0.9 <= ratio <= 1.1: score = 1.0
                elif 0.8 <= ratio <= 1.2: score = 0.5
                else:                     score = -1.5
            except (ValueError, ZeroDivisionError):
                score = -1.5
        scores.append(score)
    return scores


# Global counter used to throttle stdout printing inside reward function
_print_counter = 0


def reward_answer_numeric(prompts, completions, answer, **kwargs):
    """
    Rewards based on numeric extraction from the solution tag.
    Handles formatted numbers like "1,234".
      +1.5  correct number
      -0.5  wrong number
       0.0  non-parseable or format missing

    Also prints a sample completion every PRINT_EVERY_STEPS calls
    (useful for monitoring training progress without wandb).
    """
    global _print_counter

    responses   = [c[0]["content"] for c in completions]
    extracted   = [
        m.group(1) if (m := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    if _print_counter % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"]
        print("=" * 60)
        print(f"[Step {_print_counter}] Question:\n{question}")
        print(f"Ground truth: {answer[0]}")
        print(f"Response:\n{responses[0]}")
        print(f"Extracted: {extracted[0]}")
        print("=" * 60)
    _print_counter += 1

    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            guess_val  = float(guess.strip().replace(",", ""))
            true_val   = float(true_answer.strip())
            scores.append(1.5 if guess_val == true_val else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


# Register all reward functions used during training.
# To ablate or add rewards, edit this list.
REWARD_FUNCS = [
    reward_format_exact,
    reward_format_approximate,
    reward_answer_exact,
    reward_answer_numeric,
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
    """Compute the maximum tokenized prompt length in the dataset."""
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
    print("Loading dataset...")
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")

    print("Loading model...")
    model, tokenizer = load_model()

    print("Computing max prompt length...")
    max_prompt_length = compute_max_prompt_length(dataset, tokenizer)
    # +1 to avoid off-by-one truncation
    max_prompt_length = max_prompt_length + 1
    print(f"Max prompt length: {max_prompt_length}")

    print("Building trainer...")
    trainer = build_trainer(model, tokenizer, dataset, max_prompt_length)

    print("Starting GRPO training...")
    trainer.train()

    print("Done. Model saved to:", TRAINING_CONFIG["output_dir"])


if __name__ == "__main__":
    main()
