"""
exp_042: GTPO-EMA-flipped — Qwen3-4B on MATH benchmark (levels 3-5, integer answers).

Same dataset and reward functions as exp_041 (GRPO baseline on same data),
but replaces sequence-level GRPO advantages with GTPO-EMA-flipped per-token
advantages so that training focuses on:
  - O+ rollouts: tokens where the model was uncertain (exploration on correct paths)
  - O- rollouts: tokens where the model was overconfident (penalise confident mistakes)

compute_loss is overridden directly (not _compute_loss) to ensure unsloth's
patched GRPOTrainer does not bypass the GTPO logic.

Rollout data saved to: <EXP_DIR>/rollout_logs/step_NNNNN.npz
"""
import re, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig
from src import GTPORolloutTrainer

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
    "max_seq_length": 4096,
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.7,
}
LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "lora_alpha": 64, "use_gradient_checkpointing": "unsloth", "random_state": 3407,
}
TRAINING_CONFIG = {
    "learning_rate": 5e-6, "weight_decay": 0.1, "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine", "optim": "adamw_8bit", "logging_steps": 1,
    "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1,
    "num_generations": 8, "max_steps": 1000, "save_steps": 500,
    "max_grad_norm": 1.0, "report_to": "none",
    "output_dir": "/workspace/outputs_exp042",
    "bf16": True, "fp16": False,
}
DATASET_CONFIG = {
    "name": "DigitalLearningGmbH/MATH-lighteval",
    "split": "train",
    "levels": ["Level 3", "Level 4", "Level 5"],
    "max_prompt_tokens": 512,
    "max_completion_tokens": 3584,
    "shuffle_seed": 3407,
}
GTPO_CONFIG = {
    "alpha1": 0.9,
    "alpha2": 0.1,
    "lam": 0.9,
    "gtpo_top_k": 20,
    "reward_threshold": 0.0,
}

REASONING_START = "<working_out>"
REASONING_END   = "</working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"
PRINT_EVERY_STEPS = 10
SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

match_format = re.compile(
    rf"^[\s]{{0,}}{re.escape(REASONING_START)}.+?{re.escape(REASONING_END)}.*?"
    rf"{re.escape(SOLUTION_START)}(.+?){re.escape(SOLUTION_END)}",
    flags=re.MULTILINE | re.DOTALL,
)
match_numbers = re.compile(SOLUTION_START + r".*?([-\d\.,]+)", flags=re.MULTILINE | re.DOTALL)


def _extract_boxed(text):
    idx = text.rfind(r'\boxed{')
    if idx == -1:
        return None
    depth, start = 0, idx + len(r'\boxed{')
    for i, c in enumerate(text[start:]):
        if c == '{':   depth += 1
        elif c == '}':
            if depth == 0: return text[start:start + i]
            depth -= 1
    return None


def extract_solution_answer(raw):
    if raw is None:
        return None
    s = raw.strip().replace(',', '').replace('+', '').replace(' ', '')
    try:
        float(s)
        return s
    except ValueError:
        pass
    candidate = _extract_boxed(raw)
    if candidate is not None:
        c = candidate.strip().replace(',', '').replace('+', '').replace(' ', '')
        try:
            float(c)
            return c
        except ValueError:
            pass
    return None


_correctness_store = {}


# ── dataset helpers ────────────────────────────────────────────────────────────

def extract_boxed(solution):
    idx = solution.rfind(r'\boxed{')
    if idx == -1:
        return None
    depth, start = 0, idx + len(r'\boxed{')
    for i, c in enumerate(solution[start:]):
        if c == '{':
            depth += 1
        elif c == '}':
            if depth == 0:
                return solution[start:start + i].strip()
            depth -= 1
    return None


def normalize_integer(raw):
    return str(int(float(raw.strip().replace(",", "").replace("+", "").replace(" ", ""))))


def is_integer_answer(example):
    ans = extract_boxed(example.get("solution", ""))
    if not ans:
        return False
    try:
        s = ans.replace(",", "").replace("+", "").replace(" ", "")
        return float(s) == int(float(s))
    except (ValueError, OverflowError):
        return False


def prepare_dataset():
    ds = load_dataset(DATASET_CONFIG["name"], split=DATASET_CONFIG["split"],
                      token=os.environ.get("HF_TOKEN"))
    ds = ds.filter(lambda x: x["level"] in DATASET_CONFIG["levels"])
    ds = ds.filter(is_integer_answer)
    ds = ds.shuffle(seed=DATASET_CONFIG["shuffle_seed"])
    ds = ds.map(lambda x: {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user",   "content": x["problem"]}],
        "answer": normalize_integer(extract_boxed(x["solution"])),
        "level":  x["level"],
        "type":   x["type"],
    })
    level_counts = {}
    for ex in ds:
        level_counts[ex["level"]] = level_counts.get(ex["level"], 0) + 1
    print(f"Dataset: {len(ds)} integer-answer examples from MATH levels 3-5")
    for lvl in sorted(level_counts):
        print(f"  {lvl}: {level_counts[lvl]}")
    return ds


# ── reward functions ───────────────────────────────────────────────────────────

def reward_format_exact(completions, **kwargs):
    return [3.0 if match_format.search(c[0]["content"]) else 0.0 for c in completions]

def reward_format_approximate(completions, **kwargs):
    scores = []
    for c in completions:
        r = c[0]["content"]
        s  = 0.5 if r.count(REASONING_START) == 1 else -1.0
        s += 0.5 if r.count(REASONING_END) == 1   else -1.0
        s += 0.5 if r.count(SOLUTION_START) == 1  else -1.0
        s += 0.5 if r.count(SOLUTION_END) == 1    else -1.0
        scores.append(s)
    return scores

def reward_answer_exact(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    raw_inside = [m.group(1) if (m := match_format.search(r)) else None for r in responses]
    extracted  = [extract_solution_answer(raw) for raw in raw_inside]
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0)
        elif guess == true_answer:
            scores.append(3.0)
        else:
            try:
                ratio = float(guess) / float(true_answer)
                scores.append(1.0 if 0.9 <= ratio <= 1.1 else 0.5 if 0.8 <= ratio <= 1.2 else -1.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-1.5)
    for r, s, guess, true_answer in zip(responses, scores, extracted, answer):
        _correctness_store[r] = (s == 3.0, true_answer, guess or "")
    return scores

_cnt = 0
def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _cnt
    responses  = [c[0]["content"] for c in completions]
    extracted  = [m.group(1) if (m := match_numbers.search(r)) else None for r in responses]
    if _cnt % PRINT_EVERY_STEPS == 0:
        print(f"[Step {_cnt}] GT:{answer[0]} | Pred:{extracted[0]}")
    _cnt += 1
    scores = []
    for g, t in zip(extracted, answer):
        if g is None:
            scores.append(0.0)
            continue
        try:
            scores.append(1.5 if float(g.strip().replace(",", "")) == float(t) else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores

REWARD_FUNCS = [reward_format_exact, reward_format_approximate,
                reward_answer_exact, reward_answer_numeric]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    rollout_log_dir = os.path.join(exp_dir, "rollout_logs")

    print("=== Exp 042: GTPO-EMA-flipped, Qwen3-4B, MATH levels 3-5 integer answers ===")
    print(f"GTPO config: {GTPO_CONFIG}")
    print(f"Rollout logs → {rollout_log_dir}")
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

    lengths = []
    for ex in dataset:
        toks = tokenizer.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=True)
        lengths.append(len(toks))
    lengths.sort()
    p99_len = lengths[int(0.99 * len(lengths))]
    max_prompt_length = min(p99_len + 1, DATASET_CONFIG["max_prompt_tokens"])
    print(f"Max prompt length (p99, capped): {max_prompt_length}")

    args = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=DATASET_CONFIG["max_completion_tokens"],
        **{k: v for k, v in TRAINING_CONFIG.items()},
    )
    trainer = GTPORolloutTrainer(
        model=model, tokenizer=tokenizer, args=args,
        train_dataset=dataset, reward_funcs=REWARD_FUNCS,
        rollout_log_dir=rollout_log_dir,
        correctness_store=_correctness_store,
        conf_top_k=20,
        save_every_steps=1,
        **GTPO_CONFIG,
    )

    print(f"Starting: {TRAINING_CONFIG['num_generations']} gens × "
          f"bs={TRAINING_CONFIG['per_device_train_batch_size']} = "
          f"{TRAINING_CONFIG['num_generations'] * TRAINING_CONFIG['per_device_train_batch_size']} seqs/step")
    trainer.train()
    print("Done. Model saved to:", TRAINING_CONFIG["output_dir"])
    print(f"Rollout logs saved to: {rollout_log_dir}")

if __name__ == "__main__":
    main()
