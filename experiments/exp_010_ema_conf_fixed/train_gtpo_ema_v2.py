"""
GTPO-EMA v2 on GSM8K with Qwen3-4B (exp_010)
Fixed: EMA signal preserved — no z-score normalization destroying it.
"""
import re, sys
sys.path.insert(0, "/workspace/exp_010")

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
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
    "num_generations": 4, "max_steps": 500, "save_steps": 9999,
    "max_grad_norm": 1.0, "report_to": "none",
    "output_dir": "/workspace/outputs_exp010_gtpo_ema_v2",
}
GTPO_CONFIG = {"alpha1": 1.0, "alpha2": 0.5, "top_k": 20, "lam": 0.9, "reward_threshold": 0.0}

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

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig
from src.gtpo_ema_v2_trainer import GTPOEMAv2Trainer

match_format = re.compile(
    rf"^[\s]{{0,}}{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(re.escape(SOLUTION_START) + r".*?([\d\.,]{1,})", flags=re.MULTILINE | re.DOTALL)

def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

def prepare_dataset():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return ds.map(lambda x: {
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
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m:=match_format.search(r)) else None for r in responses]
    scores = []
    for g, t in zip(extracted, answer):
        if g is None: scores.append(0.0); continue
        if g == t: scores.append(3.0)
        elif g.strip() == t.strip(): scores.append(1.5)
        else:
            try:
                ratio = float(g) / float(t)
                scores.append(1.0 if 0.9<=ratio<=1.1 else 0.5 if 0.8<=ratio<=1.2 else -1.5)
            except: scores.append(-1.5)
    return scores

_cnt = 0
def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _cnt
    responses = [c[0]["content"] for c in completions]
    extracted = [m.group(1) if (m:=match_numbers.search(r)) else None for r in responses]
    if _cnt % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_cnt}] GT:{answer[0]} | Pred:{extracted[0]}")
    _cnt += 1
    scores = []
    for g, t in zip(extracted, answer):
        if g is None: scores.append(0.0); continue
        try: scores.append(1.5 if float(g.strip().replace(",",""))==float(t.strip()) else -0.5)
        except: scores.append(0.0)
    return scores

REWARD_FUNCS = [reward_format_exact, reward_format_approximate, reward_answer_exact, reward_answer_numeric]

def main():
    print(f"=== Exp 010: GTPO-EMA v2 Qwen3-4B GSM8K (fixed EMA signal, alpha2={GTPO_CONFIG['alpha2']}) ===")
    dataset = prepare_dataset()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"], max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"], fast_inference=MODEL_CONFIG["fast_inference"],
        max_lora_rank=MODEL_CONFIG["lora_rank"], gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"],
    )
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    lengths = dataset.map(
        lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"],add_generation_prompt=True,tokenize=True)},
        batched=True).map(lambda x: {"length": len(x["tokens"])})
    max_prompt_length = max(lengths["length"]) + 1
    print(f"Max prompt length: {max_prompt_length}")
    args = GRPOConfig(max_prompt_length=max_prompt_length,
                      max_completion_length=MODEL_CONFIG["max_seq_length"]-max_prompt_length,
                      **TRAINING_CONFIG)
    trainer = GTPOEMAv2Trainer(model=model, tokenizer=tokenizer, args=args,
                                train_dataset=dataset, reward_funcs=REWARD_FUNCS, **GTPO_CONFIG)
    trainer.train()
    print("Done. Saved to:", TRAINING_CONFIG["output_dir"])

if __name__ == "__main__":
    main()
