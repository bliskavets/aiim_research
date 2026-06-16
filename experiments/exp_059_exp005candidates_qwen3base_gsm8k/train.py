"""
exp_059 — exp_005 candidates (GTPO-Conf, GRPO-S-Conf) + GRPO baseline on
Qwen3-4B-BASE, GSM8K.
======================================================================================

Replicates exp_005 (confidence-based GTPO / GRPO-S on GSM8K with the custom
<start_working_out>/<SOLUTION> format) with the SAME hyperparameters, but:
  * model swapped from Llama-3.2-3B-Instruct -> **Qwen/Qwen3-4B-Base** (a BASE
    model: it must learn the format + answering from scratch via RL);
  * the shaping runs through the FIXED injection framework (src/shaped_loss.py) so
    it is actually applied — exp_005's original trainers override _compute_loss,
    which unsloth's compiled loss silently bypasses on this stack
    (see ../exp_057.../SHAPING_BYPASS_BUGFIX.md);
  * a plain `grpo` baseline is run for comparison.

Confidence shaping math is exp_005's (src/confidence_utils.py, verbatim):
  C = -mean_top-k(log p); compress log(1+C); O+ bonus ∝ log(1+C),
  O- penalty ∝ log(1+1/C); separate z-norm over O+/O- tokens.

Hyperparameters (exp_005): lr 5e-6 cosine, wd 0.1, warmup 0.1, adamw_8bit,
bs=1, ga=4, num_generations=4, max_steps=500, max_seq=2048, LoRA r=64,
top_k=20, alpha1=beta1=1.0, alpha2=beta2=0.1, reward_threshold=0.0, seed 3407.

Methods: grpo, gtpo_conf, grpo_s_conf.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

SEED = 3407

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B-Base",
    "max_seq_length": 2048,
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    # exp_005 used 0.9; we lower it because the shaping does an extra no-grad
    # full-logits forward for confidence. Pure infra knob (KV-cache size).
    "gpu_memory_utilization": 0.60,
}

LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 64,
    "use_gradient_checkpointing": "unsloth",
    "random_state": SEED,
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
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "seed": SEED,
    "bf16": True,
    "fp16": False,
}

SHAPING_CONFIG = {
    "gtpo_conf":   {"alpha1": 1.0, "alpha2": 0.1, "top_k": 20, "reward_threshold": 0.0},
    "grpo_s_conf": {"beta1": 1.0, "beta2": 0.1, "top_k": 20, "reward_threshold": 0.0},
}

# exp_005 custom format
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

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

match_format = re.compile(
    rf"^[\s]{{0,}}{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(SOLUTION_START + r".*?([\d\.,]{1,})", flags=re.MULTILINE | re.DOTALL)


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def prepare_dataset():
    dataset = load_dataset("openai/gsm8k", "main", split="train",
                           token=os.environ.get("HF_TOKEN"))
    return dataset.map(lambda x: {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": x["question"]}],
        "answer": extract_hash_answer(x["answer"]),
    })


# ── exp_005 reward functions (verbatim) ──────────────────────────────────────
def reward_format_exact(completions, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) else 0.0 for c in completions]


def reward_format_approximate(completions, **kwargs):
    scores = []
    for c in completions:
        r = c[0]["content"]
        s = 0.5 if r.count(REASONING_START) == 1 else -1.0
        s += 0.5 if r.count(REASONING_END) == 1 else -1.0
        s += 0.5 if r.count(SOLUTION_START) == 1 else -1.0
        s += 0.5 if r.count(SOLUTION_END) == 1 else -1.0
        scores.append(s)
    return scores


def reward_answer_exact(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := match_format.search(r)) else None for r in responses]
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
                ratio = float(guess) / float(true_answer)
                scores.append(1.0 if 0.9 <= ratio <= 1.1 else 0.5 if 0.8 <= ratio <= 1.2 else -1.5)
            except Exception:
                scores.append(-1.5)
    return scores


_cnt = 0
def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _cnt
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m := match_numbers.search(r)) else None for r in responses]
    if _cnt % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_cnt}] GT:{answer[0]} | Pred:{extracted[0]}")
    _cnt += 1
    scores = []
    for g, t in zip(extracted, answer):
        if g is None:
            scores.append(0.0); continue
        try:
            scores.append(1.5 if float(g.replace(",", "")) == float(t) else -0.5)
        except Exception:
            scores.append(0.0)
    return scores


REWARD_FUNCS = [reward_format_exact, reward_format_approximate,
                reward_answer_exact, reward_answer_numeric]


def build_trainer(method, model, tokenizer, args, dataset, format_tag_patterns):
    common = dict(model=model, tokenizer=tokenizer, args=args,
                  train_dataset=dataset, reward_funcs=REWARD_FUNCS)
    if method == "grpo":
        return GRPOTrainer(**common)
    if method == "gtpo_conf":
        from src.gtpo_conf_trainer import GTPOConfTrainer
        return GTPOConfTrainer(**common, **SHAPING_CONFIG["gtpo_conf"],
                               format_tag_patterns=format_tag_patterns)
    if method == "grpo_s_conf":
        from src.grpo_s_conf_trainer import GRPOSConfTrainer
        return GRPOSConfTrainer(**common, **SHAPING_CONFIG["grpo_s_conf"])
    raise ValueError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["grpo", "gtpo_conf", "grpo_s_conf"])
    method = ap.parse_args().method

    print(f"=== exp_059 [{method}] — Qwen3-4B-Base, GSM8K (exp_005 candidates) ===")
    dataset = prepare_dataset()
    print(f"Dataset: {len(dataset)} examples")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"],
        max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"],
        fast_inference=MODEL_CONFIG["fast_inference"],
        max_lora_rank=MODEL_CONFIG["lora_rank"],
        gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    lengths = [len(tokenizer.apply_chat_template(ex["prompt"], add_generation_prompt=True,
                                                 tokenize=True)) for ex in dataset]
    max_prompt_length = max(lengths) + 1
    print(f"Max prompt length: {max_prompt_length}")

    output_dir = os.path.join(os.path.dirname(__file__), f"outputs_{method}")
    os.makedirs(output_dir, exist_ok=True)
    args = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=MODEL_CONFIG["max_seq_length"] - max_prompt_length,
        output_dir=output_dir,
        **TRAINING_CONFIG,
    )

    # tag mask for gtpo_conf — the custom format tags (multi-token on Qwen3;
    # build_tag_mask masks the whole id-subsequence). No-op for grpo/grpo_s_conf.
    from src.format_tag_mask import encode_tag_patterns
    format_tag_patterns = encode_tag_patterns(
        tokenizer, [REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END])
    print(f"[tagmask] {len(format_tag_patterns)} patterns:")
    for pat in format_tag_patterns:
        print(f"           {pat} -> {tokenizer.decode(pat)!r}")

    trainer = build_trainer(method, model, tokenizer, args, dataset, format_tag_patterns)
    print(f"Starting [{method}] training...  sequences/step = "
          f"{TRAINING_CONFIG['num_generations'] * TRAINING_CONFIG['per_device_train_batch_size']}")
    trainer.train()
    print(f"Done [{method}]. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
