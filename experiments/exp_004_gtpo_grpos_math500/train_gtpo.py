"""
GTPO training on MATH-500 (exp_004).
Combines: exp_002 GTPO trainer + exp_003 dataset/tags/rewards.
"""

import re, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../exp_002_gtpo_and_grpo_s"))

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
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_generations": 4,
    "max_steps": 500,
    "save_steps": 250,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "output_dir": "/workspace/outputs_exp004_gtpo",
}

GTPO_CONFIG = {
    "alpha1": 1.0,
    "alpha2": 0.1,
    "eps_entropy_low": 0.2,
    "eps_entropy_high": 0.28,
    "reward_threshold": 0.0,
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

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig
from src.gtpo_trainer import GTPOTrainer

# =============================================================================
# REGEX
# =============================================================================

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# =============================================================================
# DATA
# =============================================================================

def normalize_answer(s):
    return str(s).strip()

def prepare_dataset():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["problem"]},
        ],
        "answer": normalize_answer(x["answer"]),
    })
    return dataset

# =============================================================================
# REWARD FUNCTIONS (exp_003 style)
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

_print_counter = 0

def reward_answer_exact(prompts, completions, answer, **kwargs):
    global _print_counter
    responses  = [c[0]["content"] for c in completions]
    extracted  = [m.group(1).strip() if (m := match_format.search(r)) else None for r in responses]
    if _print_counter % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_print_counter}] Q: {prompts[0][-1]['content'][:80]}...")
        print(f"  GT: {answer[0]}  |  Pred: {extracted[0]}")
    _print_counter += 1
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0); continue
        true_norm = normalize_answer(true_answer)
        if guess == true_norm:
            scores.append(5.0)
        elif guess.lower().replace(" ", "") == true_norm.lower().replace(" ", ""):
            scores.append(2.5)
        else:
            scores.append(-1.0)
    return scores

REWARD_FUNCS = [reward_format_exact, reward_format_approximate, reward_answer_exact]

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== Exp 004: GTPO on MATH-500 ===")
    print(f"GTPO config: {GTPO_CONFIG}")

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
        model, r=LORA_CONFIG["r"], target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        use_gradient_checkpointing=LORA_CONFIG["use_gradient_checkpointing"],
        random_state=LORA_CONFIG["random_state"],
    )

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

    trainer = GTPOTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset, reward_funcs=REWARD_FUNCS, **GTPO_CONFIG,
    )

    print("Starting GTPO training...")
    trainer.train()
    print("Done. Model saved to:", TRAINING_CONFIG["output_dir"])

if __name__ == "__main__":
    main()
