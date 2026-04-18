"""
GRPO + LoRA fine-tuning of Llama-3.2-3B-Instruct on Big-Math-RL-Verified
=========================================================================
Experiment 017: baseline GRPO on integer-filtered Big-Math with large group size.

Key differences from exp_001 (GSM8K baseline):
  - Dataset: SynthLabsAI/Big-Math-RL-Verified, filtered to integer-only answers
  - num_generations: 16 (vs 4) — larger group for better advantage estimation
  - per_device_train_batch_size: 4 (vs 1)
  - max_steps: 1000 (vs 500)
  - bf16 training, no quantization

Hypothesis: 16 generations per prompt yields a wider spread of rewards per group,
giving GRPO a cleaner O+/O- signal and more stable advantage estimates.
"""

# =============================================================================
# CONFIG
# =============================================================================

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2048,
    "lora_rank": 64,
    "load_in_4bit": False,          # no quantization — clean bf16 gradients
    "fast_inference": True,         # vLLM for generation
    "gpu_memory_utilization": 0.55, # leave headroom for training batch
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
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "num_generations": 16,
    "max_steps": 1000,
    "save_steps": 9999,             # no intermediate checkpoints (save disk)
    "max_grad_norm": 1.0,
    "report_to": "none",
    "output_dir": "/mnt/data/outputs/exp_017",
    "bf16": True,
    "fp16": False,
}

DATASET_CONFIG = {
    "name": "SynthLabsAI/Big-Math-RL-Verified",
    "split": "train",
    # cap prompt length to avoid OOM from outlier-long problems
    "max_prompt_tokens": 512,
    # cap completion length; integer solutions don't need > 768 tokens
    "max_completion_tokens": 768,
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
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    SOLUTION_START + r".*?([-\d\.,]+)",
    flags=re.MULTILINE | re.DOTALL,
)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def is_integer_answer(example: dict) -> bool:
    """Return True iff the answer field parses as a clean integer."""
    raw = str(example.get("answer", "")).strip().replace(",", "")
    try:
        return float(raw) == int(float(raw))
    except (ValueError, OverflowError):
        return False


def normalize_integer(raw: str) -> str:
    """Normalize integer answer to plain int string, e.g. '42.0' → '42'."""
    return str(int(float(raw.strip().replace(",", ""))))


def prepare_dataset():
    ds = load_dataset(
        DATASET_CONFIG["name"],
        split=DATASET_CONFIG["split"],
        token=os.environ.get("HF_TOKEN"),
    )

    # Keep only problems with clean integer answers
    ds = ds.filter(is_integer_answer)

    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["problem"]},
        ],
        "answer": normalize_integer(str(x["answer"])),
    })

    print(f"Dataset: {len(ds)} integer-answer examples after filtering")
    return ds

# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

def reward_format_exact(completions, **kwargs):
    """
    +3.0 if the response perfectly matches the expected tag format.
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        scores.append(3.0 if match_format.search(response) is not None else 0.0)
    return scores


def reward_format_approximate(completions, **kwargs):
    """
    Partial credit per expected tag (present exactly once = +0.5, else -1.0).
    Max +2.0. Guides format learning early in training.
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
    Multi-level correctness reward based on text inside <SOLUTION> tag:
      +3.0  exact string match
      +1.5  match after stripping whitespace
      +1.0  numeric within 10% of ground truth
      +0.5  numeric within 20% of ground truth
      -1.5  wrong answer or unparseable
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
    """
    Binary numeric reward: +1.5 correct integer, -0.5 wrong.
    Also prints a sample completion every PRINT_EVERY_STEPS steps.
    """
    global _print_counter

    responses = [c[0]["content"] for c in completions]
    extracted = [
        m.group(1) if (m := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    if _print_counter % PRINT_EVERY_STEPS == 0:
        question = prompts[0][-1]["content"]
        print("=" * 70)
        print(f"[Step {_print_counter}] Q: {question[:200]}")
        print(f"Ground truth: {answer[0]}")
        print(f"Response[0]:\n{responses[0][:400]}")
        print(f"Extracted: {extracted[0]}")
        print("=" * 70)
    _print_counter += 1

    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
            continue
        try:
            guess_val = float(guess.strip().replace(",", ""))
            true_val  = float(true_answer.strip())
            scores.append(1.5 if guess_val == true_val else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


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


def compute_max_prompt_length(dataset, tokenizer) -> int:
    """
    Compute prompt length at the 99th percentile (capped by DATASET_CONFIG).
    Using a percentile avoids one very long outlier blowing up memory.
    """
    lengths = []
    for example in dataset:
        toks = tokenizer.apply_chat_template(
            example["prompt"],
            add_generation_prompt=True,
            tokenize=True,
        )
        lengths.append(len(toks))
    lengths.sort()
    p99_idx = int(0.99 * len(lengths))
    p99_len = lengths[p99_idx]
    return min(p99_len + 1, DATASET_CONFIG["max_prompt_tokens"])

# =============================================================================
# TRAINING
# =============================================================================

def build_trainer(model, tokenizer, dataset, max_prompt_length: int):
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
        max_completion_length=DATASET_CONFIG["max_completion_tokens"],
        max_steps=TRAINING_CONFIG["max_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        report_to=TRAINING_CONFIG["report_to"],
        output_dir=TRAINING_CONFIG["output_dir"],
        bf16=TRAINING_CONFIG["bf16"],
        fp16=TRAINING_CONFIG["fp16"],
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
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)

    print("Loading dataset...")
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")

    print("Loading model...")
    model, tokenizer = load_model()

    print("Computing prompt length (99th percentile)...")
    max_prompt_length = compute_max_prompt_length(dataset, tokenizer)
    print(f"Max prompt length (capped): {max_prompt_length} tokens")
    print(f"Max completion length: {DATASET_CONFIG['max_completion_tokens']} tokens")
    print(f"Max total length: {max_prompt_length + DATASET_CONFIG['max_completion_tokens']} tokens")

    print("Building trainer...")
    trainer = build_trainer(model, tokenizer, dataset, max_prompt_length)

    print("Starting GRPO training...")
    print(f"  num_generations = {TRAINING_CONFIG['num_generations']}")
    print(f"  batch_size      = {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  sequences/step  = {TRAINING_CONFIG['num_generations'] * TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  max_steps       = {TRAINING_CONFIG['max_steps']}")
    print(f"  bf16            = {TRAINING_CONFIG['bf16']}")
    trainer.train()

    print("Done. Model saved to:", TRAINING_CONFIG["output_dir"])


if __name__ == "__main__":
    main()
