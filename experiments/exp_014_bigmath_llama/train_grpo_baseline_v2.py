"""
GRPO baseline on Big-Math-RL-Verified with Llama-3.2-3B (exp_014)
= exp_001 hyperparams + Big-Math dataset
300 steps
"""
import re

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2048,
    "lora_rank": 64,
    "load_in_4bit": True,
    "fast_inference": True,
    "gpu_memory_utilization": 0.45,
}
LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
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
    "num_generations": 8,
    "max_steps": 300,
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "output_dir": "/tmp/outputs_exp014b_grpo",
}

REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"
PRINT_EVERY_STEPS = 5
SYSTEM_PROMPT = (
    f"You are given a math problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your final answer between {SOLUTION_START}{SOLUTION_END}"
)

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

match_format = re.compile(
    rf"^[\s]{{0,}}{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

def prepare_dataset():
    ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    print(f"Dataset loaded: {len(ds)} examples")
    # Filter: only problems with verifiable numeric/symbolic answers
    ds = ds.filter(lambda x: x.get("answer") is not None and str(x.get("answer","")).strip() != "")
    print(f"After filter: {len(ds)} examples")
    return ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["problem"]},
        ],
        "answer": str(x["answer"]).strip(),
    })

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

_cnt = 0
def reward_answer_exact(prompts, completions, answer, **kwargs):
    global _cnt
    responses  = [c[0]["content"] for c in completions]
    extracted  = [m.group(1).strip() if (m := match_format.search(r)) else None for r in responses]
    if _cnt % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_cnt}] GT: {answer[0]} | Pred: {extracted[0]}")
    _cnt += 1
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None: scores.append(0.0); continue
        if guess == true_answer: scores.append(3.0)
        elif guess.strip().lower() == true_answer.strip().lower(): scores.append(1.5)
        else:
            try:
                ratio = float(guess.replace(",","")) / float(true_answer.replace(",",""))
                if   0.99 <= ratio <= 1.01: scores.append(3.0)
                elif 0.9  <= ratio <= 1.1:  scores.append(1.0)
                else:                       scores.append(-1.5)
            except: scores.append(-1.5)
    return scores

REWARD_FUNCS = [reward_format_exact, reward_format_approximate, reward_answer_exact]

def main():
    print("=== Exp 014: GRPO Llama-3.2-3B on Big-Math-RL-Verified (baseline) ===")
    print(f"Config: max_seq={MODEL_CONFIG['max_seq_length']}, steps={TRAINING_CONFIG['max_steps']}, "
          f"lr={TRAINING_CONFIG['learning_rate']}, batch={TRAINING_CONFIG['per_device_train_batch_size']}, "
          f"grad_accum={TRAINING_CONFIG['gradient_accumulation_steps']}, num_gen={TRAINING_CONFIG['num_generations']}")

    dataset = prepare_dataset()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        fast_inference=MODEL_CONFIG["fast_inference"],
        max_lora_rank=MODEL_CONFIG["lora_rank"],
        gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    lengths = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(
            x["prompt"], add_generation_prompt=True, tokenize=True)},
        batched=True,
    ).map(lambda x: {"length": len(x["tokens"])})
    max_prompt_length = max(lengths["length"]) + 1
    print(f"Max prompt length: {max_prompt_length}")

    args = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=MODEL_CONFIG["max_seq_length"] - max_prompt_length,
        **TRAINING_CONFIG,
    )
    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, args=args,
        train_dataset=dataset, reward_funcs=REWARD_FUNCS,
    )
    print("Starting training...")
    trainer.train()
    print("Done. Saved to:", TRAINING_CONFIG["output_dir"])

if __name__ == "__main__":
    main()
