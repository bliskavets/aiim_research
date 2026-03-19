"""GRPO-S-Conf training on GSM8K (exp_005)."""
import re, os, sys
sys.path.insert(0, os.path.dirname(__file__))

MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "max_seq_length": 2048, "lora_rank": 64, "load_in_4bit": False,
    "fast_inference": True, "gpu_memory_utilization": 0.9,
}
LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "lora_alpha": 64, "use_gradient_checkpointing": "unsloth", "random_state": 3407,
}
TRAINING_CONFIG = {
    "learning_rate": 5e-6, "weight_decay": 0.1, "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine", "optim": "adamw_8bit", "logging_steps": 1,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4,
    "num_generations": 4, "max_steps": 500, "save_steps": 250,
    "max_grad_norm": 1.0, "report_to": "none",
    "output_dir": "/workspace/outputs_exp005_grpos_conf",
}
GRPO_S_CONF_CONFIG = {"beta1": 1.0, "beta2": 0.1, "top_k": 20, "reward_threshold": 0.0}

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
from trl import GRPOConfig
from src.grpo_s_conf_trainer import GRPOSConfTrainer

match_format = re.compile(
    rf"^[\s]{{0,}}{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(SOLUTION_START + r".*?([\d\.,]{1,})", flags=re.MULTILINE | re.DOTALL)

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

def prepare_dataset():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    return dataset.map(lambda x: {
        "prompt": [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":x["question"]}],
        "answer": extract_hash_answer(x["answer"]),
    })

def reward_format_exact(completions, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) else 0.0 for c in completions]

def reward_format_approximate(completions, **kwargs):
    scores = []
    for c in completions:
        r = c[0]["content"]
        s  = 0.5 if r.count(REASONING_START)==1 else -1.0
        s += 0.5 if r.count(REASONING_END)==1   else -1.0
        s += 0.5 if r.count(SOLUTION_START)==1   else -1.0
        s += 0.5 if r.count(SOLUTION_END)==1     else -1.0
        scores.append(s)
    return scores

def reward_answer_exact(prompts, completions, answer, **kwargs):
    responses  = [c[0]["content"] for c in completions]
    extracted  = [m.group(1) if (m:=match_format.search(r)) else None for r in responses]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None: scores.append(0.0); continue
        if guess == true_answer: scores.append(3.0)
        elif guess.strip() == true_answer.strip(): scores.append(1.5)
        else:
            try:
                ratio = float(guess)/float(true_answer)
                scores.append(1.0 if 0.9<=ratio<=1.1 else 0.5 if 0.8<=ratio<=1.2 else -1.5)
            except: scores.append(-1.5)
    return scores

_cnt = 0
def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _cnt
    responses  = [c[0]["content"] for c in completions]
    extracted  = [m.group(1) if (m:=match_numbers.search(r)) else None for r in responses]
    if _cnt % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_cnt}] GT:{answer[0]} | Pred:{extracted[0]}")
    _cnt += 1
    scores = []
    for g, t in zip(extracted, answer):
        if g is None: scores.append(0.0); continue
        try: scores.append(1.5 if float(g.replace(",",""))==float(t) else -0.5)
        except: scores.append(0.0)
    return scores

REWARD_FUNCS = [reward_format_exact, reward_format_approximate, reward_answer_exact, reward_answer_numeric]

def main():
    print("=== Exp 005: GRPO-S-Conf on GSM8K ===")
    dataset = prepare_dataset()
    model, tokenizer = FastLanguageModel.from_pretrained(**{k:v for k,v in MODEL_CONFIG.items() if k!='lora_rank'}, max_lora_rank=MODEL_CONFIG['lora_rank'])
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    lengths = dataset.map(lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"],add_generation_prompt=True,tokenize=True)},batched=True).map(lambda x:{"length":len(x["tokens"])})
    max_prompt_length = max(lengths["length"]) + 1
    args = GRPOConfig(max_prompt_length=max_prompt_length, max_completion_length=MODEL_CONFIG["max_seq_length"]-max_prompt_length, **{k:v for k,v in TRAINING_CONFIG.items()})
    trainer = GRPOSConfTrainer(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset, reward_funcs=REWARD_FUNCS, **GRPO_S_CONF_CONFIG)
    print("Starting GRPO-S-Conf training...")
    trainer.train()
    print("Done. Saved to:", TRAINING_CONFIG["output_dir"])

if __name__ == "__main__":
    main()
