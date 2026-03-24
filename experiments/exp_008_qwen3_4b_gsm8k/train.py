"""
GRPO + LoRA fine-tuning of Qwen3-4B on GSM8K
=============================================
Replication of exp_001 with Qwen/Qwen3-4B instead of Llama-3.2-3B.
Same hyperparameters, same reward functions, same dataset.
"""

# =============================================================================
# CONFIG
# =============================================================================

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
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
    "output_dir": "/workspace/outputs_exp008",
}

REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"
PRINT_EVERY_STEPS = 5

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
# REGEX
# =============================================================================

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(
    SOLUTION_START + r".*?([\d\.,]{1,})",
    flags=re.MULTILINE | re.DOTALL,
)

# =============================================================================
# DATA
# =============================================================================

def extract_hash_answer(text):
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
# REWARD FUNCTIONS (identical to exp_001)
# =============================================================================

def reward_format_exact(completions, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) else 0.0 for c in completions]

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
    extracted = [m.group(1) if (m := match_format.search(r)) else None for r in responses]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None: scores.append(0.0); continue
        if guess == true_answer: scores.append(3.0)
        elif guess.strip() == true_answer.strip(): scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if   0.9 <= ratio <= 1.1: scores.append(1.0)
                elif 0.8 <= ratio <= 1.2: scores.append(0.5)
                else:                     scores.append(-1.5)
            except: scores.append(-1.5)
    return scores

_print_counter = 0

def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _print_counter
    responses  = [c[0]["content"] for c in completions]
    extracted  = [m.group(1) if (m := match_numbers.search(r)) else None for r in responses]
    if _print_counter % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_print_counter}] GT: {answer[0]}  Pred: {extracted[0]}")
    _print_counter += 1
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None: scores.append(0.0); continue
        try:
            gv = float(guess.strip().replace(",", ""))
            tv = float(true_answer.strip())
            scores.append(1.5 if gv == tv else -0.5)
        except: scores.append(0.0)
    return scores

REWARD_FUNCS = [
    reward_format_exact,
    reward_format_approximate,
    reward_answer_exact,
    reward_answer_numeric,
]

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== Exp 008: GRPO Qwen3-4B on GSM8K ===")
    print(f"Model: {MODEL_CONFIG['model_name']}")

    print("Loading dataset...")
    dataset = prepare_dataset()
    print(f"Dataset size: {len(dataset)}")

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

    print("Computing max prompt length...")
    lengths = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    ).map(lambda x: {"length": len(x["tokens"])})
    max_prompt_length = max(lengths["length"]) + 1
    print(f"Max prompt length: {max_prompt_length}")

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

    print("Starting GRPO training...")
    trainer.train()
    print("Done. Model saved to:", TRAINING_CONFIG["output_dir"])

if __name__ == "__main__":
    main()
