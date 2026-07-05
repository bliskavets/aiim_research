"""
exp_061 — GROP @ GRPO vs FIXED gtpo_ema_flipped, across THREE datasets.
=======================================================================
Follow-up to exp_058. Two setups per dataset, identical to the LAST exp_058
setups (Qwen3-4B-Base, ng=4, bs=1, ga=4, lr 5e-6 cosine, 300 steps, seed 3407,
max_seq 4096 = 512 prompt + 3584 completion, integer-answer reward + tag-mask):

  - grpo_grop              : plain GRPO + GROP (arXiv:2508.04349 App.D) as a
                            REWARD term (paper-faithful injection point).
  - gtpo_ema_flipped_fixed : gtpo_ema_flipped with the shaped advantage computed
                            on the FULL group in _generate_and_score (the B=1 fix
                            from exp_058 DIAG_LENGTH_EXPLOSION.md).

Datasets (all integer-answer, exact-match verifiable; --dataset selects):
  gsm8k    : openai/gsm8k (main/train)          — gold = the number after "####"
  math500  : HuggingFaceH4/MATH-500 (test)       — integer-answer subset (312)
  omnimath : KbsdJames/Omni-MATH (test)          — integer-answer subset (1971)
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

SEED = 3407

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B-Base",
    "max_seq_length": 4096,
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.55,
}
LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    "max_steps": int(os.environ.get("SMOKE_MAX_STEPS", 300)),
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "seed": SEED,
    "bf16": True,
    "fp16": False,
}
DATA_COMMON = {"max_prompt_tokens": 512, "max_completion_tokens": 3584,
               "subset_size": 2000, "shuffle_seed": SEED}
_CONF_MICRO_BS = int(os.environ.get("CONF_MICRO_BS", 1))

_SHAPE_BASE = {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20,
               "reward_threshold": 0.0, "conf_micro_bs": _CONF_MICRO_BS}
SHAPING_CONFIG = {
    "gtpo_ema_flipped_fixed": dict(_SHAPE_BASE),
    # exp_062 non-entropy candidates (all on the group-visible FIXED pattern)
    "sign_gate":    dict(_SHAPE_BASE),                         # 6A sign-consistency gate
    "pos_discount": dict(_SHAPE_BASE, pos_tau=1024.0),        # gentle g(t)=tau/(tau+t) on bonus
    "raw_c":        dict(_SHAPE_BASE),                         # raw C instead of EMA(C)
    "ref_delta":    dict(_SHAPE_BASE),                         # 3A reference-relative log-delta
    # exp_068: dynamic nucleus (top-p) k for C, base FIXED lam=0.7 + pos_discount
    "nucleus_c": dict(_SHAPE_BASE, lam=0.7, pos_tau=1024.0, nucleus_top_p=0.9, min_k=1, nucleus_cap=256),
    # exp_079: our shaping (pos_discount FIXED, λ=0.7, k=5) applied ON TOP of Dr.GRPO.
    "drgrpo_shaped": dict(_SHAPE_BASE, lam=0.7, top_k=5, pos_tau=1024.0),
}
# exp_079: Dr.GRPO (arXiv:2503.20783) — unbiased GRPO: constant-normalized token loss
# (loss_type='dr_grpo') + NO std scaling of rewards (scale_rewards='none') to remove the
# length and question-difficulty biases.
DRGRPO_METHODS = ("drgrpo", "drgrpo_shaped")
DRGRPO_CONFIG = {"loss_type": "dr_grpo", "scale_rewards": "none"}
GROP_GAMMA1 = 0.75
PRINT_EVERY_STEPS = 10
SYSTEM_PROMPT = ("Solve the problem step by step. "
                 "Put your final integer answer inside \\boxed{}, like \\boxed{42}.")

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ── answer extraction (Qwen3 native: \boxed{} after </think>) ──
_boxed_re = re.compile(r"\\boxed\{\s*(-?\d[\d.,]*)\s*\}")
_last_number_re = re.compile(r"-?\d[\d.,]*")
_think_open, _think_close = "<think>", "</think>"


def _answer_region(text):
    ho, hc = _think_open in text, _think_close in text
    if not ho and not hc:
        return text
    if ho and hc:
        return text.rsplit(_think_close, 1)[1]
    return None


def _extract_boxed_answer(text):
    region = _answer_region(text)
    if region is None:
        return None
    matches = _boxed_re.findall(region)
    return matches[-1] if matches else None


def _extract_last_number_after_thinking(text):
    region = _answer_region(text)
    if region is None:
        return None
    nums = _last_number_re.findall(region)
    return nums[-1] if nums else None


# ── integer normalization shared by MATH-500 / Omni-MATH ──
_INT_RE = re.compile(r"-?\d+")
_THOUSANDS_RE = re.compile(r"-?\d{1,3}(,\d{3})+")
_BOXED_WRAP_RE = re.compile(r"^\\boxed\{(.+)\}$")


def _clean_integer(raw) -> str:
    s = str(raw).strip()
    m = _BOXED_WRAP_RE.match(s)
    if m:
        s = m.group(1).strip()
    s = s.strip("$").strip()
    if _THOUSANDS_RE.fullmatch(s):
        s = s.replace(",", "")
    return s if _INT_RE.fullmatch(s) else ""


_GSM_GOLD_RE = re.compile(r"####\s*(-?[\d,]+)")


def _gsm_gold(answer_text):
    m = _GSM_GOLD_RE.search(answer_text)
    return m.group(1).replace(",", "") if m else ""


def _to_prompt(problem, gold_int):
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": problem}],
            "answer": str(int(gold_int))}


def prepare_dataset(name):
    tok = os.environ.get("HF_TOKEN")
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train", token=tok)
        ds = ds.map(lambda x: _to_prompt(x["question"], _gsm_gold(x["answer"])))
        n0, n1 = len(ds), len(ds)
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", token=tok)
        n0 = len(ds)
        ds = ds.filter(lambda x: _clean_integer(x["answer"]) != "")
        n1 = len(ds)
        ds = ds.map(lambda x: _to_prompt(x["problem"], _clean_integer(x["answer"])))
    elif name == "omnimath":
        ds = load_dataset("KbsdJames/Omni-MATH", split="test", token=tok)
        n0 = len(ds)
        ds = ds.filter(lambda x: _clean_integer(x["answer"]) != "")
        n1 = len(ds)
        ds = ds.map(lambda x: _to_prompt(x["problem"], _clean_integer(x["answer"])))
    elif name == "bigmath":
        # exact exp_058 setup (the exp058_fix_grop.png figure): integer subset, first 2000
        ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train", token=tok)
        n0 = len(ds)
        ds = ds.filter(lambda x: _clean_integer(x["answer"]) != "")
        n1 = len(ds)
        ds = ds.map(lambda x: _to_prompt(x["problem"], _clean_integer(x["answer"])))
    else:
        raise ValueError(f"unknown dataset: {name}")
    ds = ds.shuffle(seed=DATA_COMMON["shuffle_seed"])
    ds = ds.select(range(min(DATA_COMMON["subset_size"], len(ds))))
    print(f"Dataset[{name}]: total={n0} -> integer={n1} -> using {len(ds)} (shuffled seed={SEED})")
    return ds


# ── rewards (identical to exp_058 integer setup) ──
def reward_format_thinking(completions, **kwargs):
    scores = []
    for c in completions:
        r = c[0]["content"]
        no, nc = r.count(_think_open), r.count(_think_close)
        scores.append(2.5 if (no == 1 and nc == 1) else (1.5 if (no == 0 and nc == 0) else -2.0))
    return scores


def reward_answer_boxed(prompts, completions, answer, **kwargs):
    scores = []
    for c, true_answer in zip(completions, answer):
        guess = _extract_boxed_answer(c[0]["content"])
        if guess is None:
            scores.append(0.0); continue
        try:
            gv = float(guess.strip().replace(",", "")); tv = float(true_answer.strip())
            scores.append(3.0 if gv == tv else -1.5)
        except (ValueError, ZeroDivisionError):
            scores.append(-1.5)
    return scores


_print_counter = 0


def reward_answer_numeric(prompts, completions, answer, **kwargs):
    global _print_counter
    responses = [c[0]["content"] for c in completions]
    extracted = [_extract_last_number_after_thinking(r) for r in responses]
    if _print_counter % PRINT_EVERY_STEPS == 0:
        first = responses[0]
        boxed = _extract_boxed_answer(first)
        print(f"[Step {_print_counter}] GT:{answer[0]} | last-num:{extracted[0]} | "
              f"think:({first.count(_think_open)},{first.count(_think_close)}) | boxed:{boxed}")
    _print_counter += 1
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0); continue
        try:
            gv = float(guess.strip().replace(",", "")); tv = float(true_answer.strip())
            scores.append(1.5 if gv == tv else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


REWARD_FUNCS_FULL = [reward_format_thinking, reward_answer_boxed, reward_answer_numeric]


def make_grop_reward(tokenizer, num_generations, gamma1=GROP_GAMMA1):
    """GROP (arXiv:2508.04349 App.D) as a REWARD term — paper-faithful injection.
    Runs on the full generation batch (groups of num_generations), no B=1."""
    from src.adaptive_lenpen_utils import group_relative_overlong_punishment
    G = num_generations

    def reward_grop_overlong(prompts, completions, answer, **kwargs):
        texts = [c[0]["content"] for c in completions]
        lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        correct = []
        for t, gold in zip(texts, answer):
            guess = _extract_boxed_answer(t)
            ok = False
            if guess is not None:
                try:
                    ok = float(str(guess).strip().replace(",", "")) == float(str(gold).strip())
                except (ValueError, TypeError):
                    ok = False
            correct.append(1.0 if ok else 0.0)
        n = len(texts)
        if n % G != 0:
            return [0.0] * n
        pen, _ = group_relative_overlong_punishment(
            torch.tensor(lengths, dtype=torch.float32), torch.tensor(correct), G, gamma1=gamma1)
        return [-float(p) for p in pen.tolist()]

    reward_grop_overlong.__name__ = "reward_grop_overlong"
    return reward_grop_overlong


def build_trainer(method, model, tokenizer, args, dataset, reward_funcs, format_tag_patterns=None):
    common = dict(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset,
                  reward_funcs=reward_funcs)
    if method == "grpo":
        return GRPOTrainer(**common)                          # plain baseline (missing setup)
    if method == "grpo_grop":
        grop = make_grop_reward(tokenizer, TRAINING_CONFIG["num_generations"])
        c = dict(common); c["reward_funcs"] = list(reward_funcs) + [grop]
        return GRPOTrainer(**c)
    if method == "gtpo_ema_flipped_fixed":
        from src.gtpo_ema_flipped_fixed_trainer import GTPOEMAFlippedFixedTrainer
        return GTPOEMAFlippedFixedTrainer(**common, **SHAPING_CONFIG["gtpo_ema_flipped_fixed"],
                                          format_tag_patterns=format_tag_patterns)
    # ── exp_062 candidates (group-visible FIXED pattern) ──
    if method in ("sign_gate", "pos_discount", "raw_c", "ref_delta"):
        from src.novel_trainers import (SignGateTrainer, PosDiscountTrainer,
                                         RawCTrainer, RefDeltaTrainer)
        cls = {"sign_gate": SignGateTrainer, "pos_discount": PosDiscountTrainer,
               "raw_c": RawCTrainer, "ref_delta": RefDeltaTrainer}[method]
        return cls(**common, **SHAPING_CONFIG[method], format_tag_patterns=format_tag_patterns)
    if method == "drgrpo":
        return GRPOTrainer(**common)                          # plain GRPOTrainer + Dr.GRPO config knobs
    if method == "drgrpo_shaped":
        from src.novel_trainers import PosDiscountTrainer     # our shaping ON TOP of Dr.GRPO
        return PosDiscountTrainer(**common, **SHAPING_CONFIG["drgrpo_shaped"],
                                  format_tag_patterns=format_tag_patterns)
    raise ValueError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500", "omnimath", "bigmath"])
    ap.add_argument("--method", required=True,
                    choices=["grpo", "grpo_grop", "gtpo_ema_flipped_fixed",
                             "sign_gate", "pos_discount", "raw_c", "ref_delta", "nucleus_c",
                             "drgrpo", "drgrpo_shaped"])
    ap.add_argument("--lam", type=float, default=None,
                    help="override EMA lambda for gtpo_ema_flipped_fixed (default 0.9)")
    ap.add_argument("--top_k", type=int, default=None, help="override fixed top_k for C")
    ap.add_argument("--top_p", type=float, default=None, help="nucleus top_p for nucleus_c")
    ap.add_argument("--min_k", type=int, default=None, help="min k for nucleus_c/rank_c")
    ap.add_argument("--cap", type=int, default=None, help="rank_c cap on adaptive k (max k)")
    a = ap.parse_args()

    tag = a.method
    if a.lam is not None and a.method in SHAPING_CONFIG and "lam" in SHAPING_CONFIG[a.method]:
        SHAPING_CONFIG[a.method]["lam"] = a.lam
        tag = f"{tag}_lam{a.lam}"
        print(f"[lam override] {a.method} lam={a.lam}")
    if a.top_k is not None and a.method in SHAPING_CONFIG and "top_k" in SHAPING_CONFIG[a.method]:
        SHAPING_CONFIG[a.method]["top_k"] = a.top_k; tag = f"{tag}_k{a.top_k}"
        print(f"[top_k override] {a.method} top_k={a.top_k}")
    if a.top_p is not None and a.method in SHAPING_CONFIG and "nucleus_top_p" in SHAPING_CONFIG[a.method]:
        SHAPING_CONFIG[a.method]["nucleus_top_p"] = a.top_p; tag = f"{tag}_p{a.top_p}"
        print(f"[top_p override] {a.method} nucleus_top_p={a.top_p}")
    if a.min_k is not None and a.method in SHAPING_CONFIG and "min_k" in SHAPING_CONFIG[a.method]:
        SHAPING_CONFIG[a.method]["min_k"] = a.min_k
        print(f"[min_k override] {a.method} min_k={a.min_k}")
    if a.cap is not None and a.method in SHAPING_CONFIG and "rank_cap" in SHAPING_CONFIG[a.method]:
        SHAPING_CONFIG[a.method]["rank_cap"] = a.cap; tag = f"{tag}_cap{a.cap}"
        print(f"[cap override] {a.method} rank_cap={a.cap}")
    output_dir = os.path.join(os.path.dirname(__file__), f"outputs_{a.dataset}_{tag}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== exp_061 [{a.dataset} | {a.method}] Qwen3-4B-BASE — steps={TRAINING_CONFIG['max_steps']} "
          f"bs1xga4xng4 seed={SEED} ===")

    dataset = prepare_dataset(a.dataset)

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_CONFIG["model_name"], max_seq_length=MODEL_CONFIG["max_seq_length"],
        load_in_4bit=MODEL_CONFIG["load_in_4bit"], fast_inference=MODEL_CONFIG["fast_inference"],
        max_lora_rank=MODEL_CONFIG["lora_rank"], gpu_memory_utilization=MODEL_CONFIG["gpu_memory_utilization"])
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_CONFIG["r"], target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        use_gradient_checkpointing=LORA_CONFIG["use_gradient_checkpointing"],
        random_state=LORA_CONFIG["random_state"])

    if getattr(tokenizer, "chat_template", None) is None:
        from transformers import AutoTokenizer
        tmpl = None
        for src in (MODEL_CONFIG["model_name"], "Qwen/Qwen3-4B"):
            try:
                tmpl = AutoTokenizer.from_pretrained(src, token=os.environ.get("HF_TOKEN")).chat_template
                if tmpl:
                    print(f"[chat_template] set from {src}"); break
            except Exception as e:
                print(f"[chat_template] {src} failed: {repr(e)[:80]}")
        if not tmpl:
            raise RuntimeError("no chat_template available for the base model")
        tokenizer.chat_template = tmpl

    lengths = sorted(len(tokenizer.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=True))
                     for ex in dataset)
    max_prompt_length = min(lengths[int(0.99 * len(lengths))] + 1, DATA_COMMON["max_prompt_tokens"])
    print(f"Max prompt length (99%, capped): {max_prompt_length}")

    grpo_args = GRPOConfig(max_prompt_length=max_prompt_length,
                           max_completion_length=DATA_COMMON["max_completion_tokens"],
                           output_dir=output_dir, **TRAINING_CONFIG)

    if a.method in DRGRPO_METHODS:
        for k, v in DRGRPO_CONFIG.items():
            setattr(grpo_args, k, v)
        print(f"[Dr.GRPO] {DRGRPO_CONFIG} (const-normalized loss, no reward scaling)")

    from src.format_tag_mask import encode_tag_patterns
    format_tag_patterns = encode_tag_patterns(tokenizer, ["<think>", "</think>", "<|im_start|>", "<|im_end|>"])

    trainer = build_trainer(a.method, model, tokenizer, grpo_args, dataset,
                            REWARD_FUNCS_FULL, format_tag_patterns=format_tag_patterns)
    print(f"Starting [{a.dataset} | {a.method}] ...")
    trainer.train()
    print(f"Done [{a.dataset} | {a.method}]. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
