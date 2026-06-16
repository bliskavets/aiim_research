"""
exp_057 — Qwen3-4B native format, all 4 methods on Omni-MATH integer subset
======================================================================================

Exact re-run of exp_055 (same model, same 4 methods, same hyperparameters,
same Qwen3 NATIVE format) but on a DIFFERENT dataset: KbsdJames/Omni-MATH,
restricted to its integer-answer subset.

Motivation: exp_055 was a null result — the easy Big-Math integer-2000 slice
saturates Qwen3-4B (~82% of the strict-answer ceiling at step 0), so shaping
had no headroom. Omni-MATH is competition-grade (olympiad/contest problems,
difficulty 1-9.5, mean ~4.16), so the GRPO baseline should be far from
saturated and per-token shaping may finally have room to act. This is the
non-saturated Qwen3 test the cross-experiment takeaway has been missing.

Everything except the dataset is held identical to exp_055:
  SYSTEM_PROMPT  "Solve step by step. Final integer answer in \\boxed{}."
  reward funcs   reward_format_thinking + reward_answer_boxed + reward_answer_numeric
  tag-mask       <think>, </think>, <|im_start|>, <|im_end|> (4 native tags)
  parsing        \\boxed{N} primary, last-number-after-</think> fallback;
                 unclosed <think> -> no answer reward
  max_seq=6656 (512 + 6144), gpu_memory_utilization=0.40, ng=8
  bs=1, ga=4, lr=5e-6 cosine, max_steps=1000, seed=3407

Dataset:
  KbsdJames/Omni-MATH (single split 'test', 4428 problems).
  Filter to integer answers: unwrap a single \\boxed{...} / surrounding $,
  collapse thousands-commas, keep only answers matching -?\\d+.
  -> 1971 integer-answer problems. Shuffled (seed 3407), all kept
  (subset_size cap 2000 > 1971, so the whole integer subset is used).

Methods: all 4 — grpo, grpo_s_entropy, gtpo_conf, gtpo_ema_flipped.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# CONFIG — training hyperparameters from exp_005 (shared by all four methods)
# =============================================================================

SEED = 3407

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
    "max_seq_length": 6656,          # 512 prompt + 6144 completion (v3: bumped from 4096 — 85% clip in v2)
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.40,  # v3: dropped 0.50 -> 0.40 to fit longer context + ng=16 activations
}

LORA_CONFIG = {
    "r": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 64,
    "use_gradient_checkpointing": "unsloth",
    "random_state": SEED,
}

# exp_005 TRAINING_CONFIG (verbatim) + seed, bf16, output set per-method below
TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "per_device_train_batch_size": 1,    # exp_055: global_bs = per_device * ga = 4 (user spec)
    "gradient_accumulation_steps": 4,    # exp_055
    "num_generations": 8,                # exp_055: ng=8 (kept conservative; easy subset has shorter completions so ng=16 may fit too, but stay safe)
    "max_steps": 1000,                   # exp_055: 2x more updates vs exp_052
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "seed": SEED,
    "bf16": True,
    "fp16": False,
}

DATASET_CONFIG = {
    "name": "KbsdJames/Omni-MATH",
    "split": "test",                 # Omni-MATH ships a single 'test' split (4428)
    "max_prompt_tokens": 512,
    "max_completion_tokens": 6144,   # exp_055 v3: 3584 caused 85% clipping on Qwen3 thinking-mode
    "subset_size": 2000,             # cap > integer-subset size (1971) -> use the whole integer subset
    "shuffle_seed": SEED,
}

# Method-native shaping coefficients (intrinsic to each method, from its source exp)
SHAPING_CONFIG = {
    "grpo_s_entropy":   {"beta1": 1.0, "beta2": 0.1, "reward_threshold": 0.0},
    "gtpo_conf":        {"alpha1": 1.0, "alpha2": 0.1, "top_k": 20, "reward_threshold": 0.0},
    "gtpo_ema_flipped": {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20, "reward_threshold": 0.0},
}

# exp_055: use Qwen3's NATIVE format — <think>...</think> for reasoning, then
# \boxed{N} for the final answer. No custom format tags. enable_thinking is
# left at the chat-template default (True), so the assistant span begins with
# the model's choice (Qwen3 typically opens a <think> block on math prompts).
PRINT_EVERY_STEPS = 10

SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Put your final integer answer inside \\boxed{}, like \\boxed{42}."
)

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Parse the final answer from the assistant span. Qwen3 native math format
# emits a \boxed{...} block after </think> (or anywhere, if thinking is off).
# We extract the LAST \boxed{...} match in the completion — agrees with the
# convention that the final answer is the last \boxed{} block.
_boxed_re = re.compile(r"\\boxed\{\s*(-?\d[\d.,]*)\s*\}")
# require at least one digit so the regex never extracts a lone "." or ","
_last_number_re = re.compile(r"-?\d[\d.,]*")
_think_open = "<think>"
_think_close = "</think>"


def _answer_region(text: str):
    """Return the substring where the final answer is expected, or None.

      no <think> at all      -> the whole text       (model skipped thinking, OK)
      <think>...</think>    -> the part after </think>
      <think> opened, never closed  -> None          (rollout clipped mid-think;
                                                       refuses to score boxed
                                                       found inside thinking)

    This blocks the exploit observed in first 107 steps of exp_055 grpo:
    model emitting \\boxed{} inside an unclosed thinking block, farming
    answer-correctness reward without ever committing to a final answer.
    """
    has_open = _think_open in text
    has_close = _think_close in text
    if not has_open and not has_close:
        return text                                  # answer-direct mode
    if has_open and has_close:
        return text.rsplit(_think_close, 1)[1]       # post-thinking tail
    return None                                      # asymmetric — open w/o close


def _extract_boxed_answer(text: str):
    """Last \\boxed{...} numeric content in the answer region, or None."""
    region = _answer_region(text)
    if region is None:
        return None
    matches = _boxed_re.findall(region)
    return matches[-1] if matches else None


def _extract_last_number_after_thinking(text: str):
    """Fallback: last number in the answer region, or None."""
    region = _answer_region(text)
    if region is None:
        return None
    nums = _last_number_re.findall(region)
    return nums[-1] if nums else None


# =============================================================================
# DATASET — Omni-MATH, integer-answer subset
# =============================================================================
#   filter: integer answer ONLY
#   then: shuffle by seed, take first subset_size (cap 2000 > 1971 -> all)
#
# Omni-MATH answers are competition-style and often LaTeX-wrapped, e.g.
# "\\boxed{60}", "$30", "1,700", or non-integer like "1 + \\lceil n/2 \\rceil",
# "\\frac{1}{2}", "2\\sqrt{3}". We keep ONLY answers that reduce to a plain
# signed integer after a minimal, safe normalization (no aggressive comma
# stripping that would merge a European decimal "3,7" into "37").


_INT_RE = re.compile(r"-?\d+")
_THOUSANDS_RE = re.compile(r"-?\d{1,3}(,\d{3})+")
_BOXED_WRAP_RE = re.compile(r"^\\boxed\{(.+)\}$")


def _clean_integer(raw) -> str:
    """Reduce an Omni-MATH answer to a plain signed-integer string, or '' if
    it is not an integer answer. Mirrors the filter used by is_integer_answer."""
    s = str(raw).strip()
    m = _BOXED_WRAP_RE.match(s)          # unwrap a single surrounding \boxed{...}
    if m:
        s = m.group(1).strip()
    s = s.strip("$").strip()             # drop surrounding inline-math $...$
    if _THOUSANDS_RE.fullmatch(s):       # 1,700 -> 1700 (thousands grouping only)
        s = s.replace(",", "")
    return s if _INT_RE.fullmatch(s) else ""


def is_integer_answer(example: dict) -> bool:
    return _clean_integer(example.get("answer", "")) != ""


def normalize_integer(raw: str) -> str:
    cleaned = _clean_integer(raw)
    return str(int(cleaned))


def prepare_dataset():
    ds = load_dataset(DATASET_CONFIG["name"], split=DATASET_CONFIG["split"],
                      token=os.environ.get("HF_TOKEN"))
    n0 = len(ds)
    ds = ds.filter(is_integer_answer)
    n1 = len(ds)
    print(f"Filter: total={n0} -> integer-answer={n1}")
    ds = ds.shuffle(seed=DATASET_CONFIG["shuffle_seed"])
    ds = ds.select(range(min(DATASET_CONFIG["subset_size"], len(ds))))
    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["problem"]},
        ],
        "answer": normalize_integer(str(x["answer"])),
    })
    print(f"Dataset: {len(ds)} integer-answer examples "
          f"(Omni-MATH, shuffled seed={DATASET_CONFIG['shuffle_seed']})")
    return ds


# =============================================================================
# REWARDS — Qwen3 native format (exp_055)
# =============================================================================
#
# Three reward components, all returning a float per generation:
#   reward_format_thinking  — soft preference for exactly one matched
#                             <think>...</think> block (Qwen3's natural shape)
#   reward_answer_boxed     — strong preference for the answer being inside
#                             a \boxed{...} block matching the GT integer
#   reward_answer_numeric   — fallback numeric correctness on the last number
#                             after </think> (catches the "right answer but
#                             not boxed" case so the model still gets credit
#                             for solving the problem while we shape it
#                             toward boxed format)

def reward_format_thinking(completions, **kwargs):
    """Bigger format signal so model gets gradient toward closing <think>
    even before it solves the math:
      +2.5  exactly one matched <think>...</think> pair
      +1.5  no <think>/</think> blocks at all (direct-answer mode, also OK)
      -2.0  open without close (or any mismatch — strong push to close)
    v3: boosted from +1.0/+0.5/-0.5 because v2 had model stuck at 85%
    clipping and KL=0.0008 — too weak a signal to learn the format alone.
    """
    scores = []
    for c in completions:
        r = c[0]["content"]
        n_open  = r.count(_think_open)
        n_close = r.count(_think_close)
        if n_open == 1 and n_close == 1:
            scores.append(2.5)
        elif n_open == 0 and n_close == 0:
            scores.append(1.5)
        else:
            scores.append(-2.0)
    return scores


def reward_answer_boxed(prompts, completions, answer, **kwargs):
    """Integer-only: +3.0 exact match in \\boxed{N}, -1.5 wrong number, 0.0 no boxed."""
    scores = []
    for c, true_answer in zip(completions, answer):
        guess = _extract_boxed_answer(c[0]["content"])
        if guess is None:
            scores.append(0.0); continue
        try:
            gv = float(guess.strip().replace(",", ""))
            tv = float(true_answer.strip())
            scores.append(3.0 if gv == tv else -1.5)
        except (ValueError, ZeroDivisionError):
            scores.append(-1.5)
    return scores


_print_counter = 0


def reward_answer_numeric(prompts, completions, answer, **kwargs):
    """+1.5 if the LAST number after </think> matches GT (numeric equality),
    else -0.5 if number found but wrong, else 0.0 if no number."""
    global _print_counter
    responses = [c[0]["content"] for c in completions]
    extracted = [_extract_last_number_after_thinking(r) for r in responses]
    if _print_counter % PRINT_EVERY_STEPS == 0:
        # also print whether <think>...</think> closed correctly + any \boxed{}
        first = responses[0]
        nopen = first.count(_think_open); nclose = first.count(_think_close)
        boxed = _extract_boxed_answer(first)
        print(f"[Step {_print_counter}] GT:{answer[0]} | last-num:{extracted[0]} | "
              f"think:({nopen},{nclose}) | boxed:{boxed}")
    _print_counter += 1
    scores = []
    for guess, true_answer in zip(extracted, answer):
        if guess is None:
            scores.append(0.0); continue
        try:
            gv = float(guess.strip().replace(",", ""))
            tv = float(true_answer.strip())
            scores.append(1.5 if gv == tv else -0.5)
        except (ValueError, AttributeError):
            scores.append(0.0)
    return scores


REWARD_FUNCS_FULL = [reward_format_thinking, reward_answer_boxed, reward_answer_numeric]


# =============================================================================
# TRAINER FACTORY
# =============================================================================

def build_trainer(method, model, tokenizer, args, dataset, reward_funcs,
                  format_tag_patterns=None):
    common = dict(model=model, tokenizer=tokenizer, args=args,
                  train_dataset=dataset, reward_funcs=reward_funcs)
    if method == "grpo":
        # baseline has no per-token shaping → mask is a no-op; serves as control
        return GRPOTrainer(**common)
    if method == "grpo_s_entropy":
        from src.grpo_s_trainer import GRPOSTrainer
        # GRPO-S shaping is seq-level (mean entropy modifies the seq-level
        # advantage). there is no per-token bonus to mask, so format_tag
        # masking has no effect here; we still run it as a clean comparator.
        return GRPOSTrainer(**common, **SHAPING_CONFIG["grpo_s_entropy"])
    if method == "gtpo_conf":
        from src.gtpo_conf_trainer import GTPOConfTrainer
        return GTPOConfTrainer(**common, **SHAPING_CONFIG["gtpo_conf"],
                               format_tag_patterns=format_tag_patterns)
    if method == "gtpo_ema_flipped":
        from src.gtpo_ema_flipped_trainer import GTPOEMAFlippedTrainer
        return GTPOEMAFlippedTrainer(**common, **SHAPING_CONFIG["gtpo_ema_flipped"],
                                     format_tag_patterns=format_tag_patterns)
    raise ValueError(f"unknown method: {method}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["grpo", "grpo_s_entropy", "gtpo_conf", "gtpo_ema_flipped"])
    args_cli = ap.parse_args()
    method = args_cli.method
    reward_funcs = REWARD_FUNCS_FULL  # exp_055 uses full reward set, same as exp_050/051/052

    output_dir = os.path.join(os.path.dirname(__file__), f"outputs_{method}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== exp_057 [{method}] — Omni-MATH integer subset, Qwen3-4B (tagmasked shaping) ===")
    print(f"  seed={SEED}  max_seq={MODEL_CONFIG['max_seq_length']}  "
          f"steps={TRAINING_CONFIG['max_steps']}  "
          f"bs={TRAINING_CONFIG['per_device_train_batch_size']}x"
          f"ga{TRAINING_CONFIG['gradient_accumulation_steps']}x"
          f"ng{TRAINING_CONFIG['num_generations']}")

    dataset = prepare_dataset()

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

    lengths = []
    for ex in dataset:
        toks = tokenizer.apply_chat_template(ex["prompt"],
                                             add_generation_prompt=True,
                                             tokenize=True)
        lengths.append(len(toks))
    lengths.sort()
    p99_len = lengths[int(0.99 * len(lengths))]
    max_prompt_length = min(p99_len + 1, DATASET_CONFIG["max_prompt_tokens"])
    print(f"Max prompt length (99%, capped): {max_prompt_length}")

    grpo_args = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=DATASET_CONFIG["max_completion_tokens"],
        output_dir=output_dir,
        **TRAINING_CONFIG,
    )

    # Build format-tag token-id patterns once.
    #
    # These are the structural/format substrings the model becomes extremely
    # peaked on AND that the reward functions key on; per-token shaping must NOT
    # rewrite the gradient there (it gets reverted to the seq-level GRPO adv).
    #   - <think>, </think>        : thinking protocol — single token ids (151667/151668)
    #   - <|im_start|>, <|im_end|> : ChatML role boundaries — single token ids
    #   - \boxed{ , }              : answer-format delimiters that reward_answer_boxed
    #                                rewards. NOTE these are MULTI-token substrings
    #                                (\boxed{ -> ['\\','boxed','{'] = [59,79075,90]),
    #                                so build_tag_mask masks the whole id-subsequence
    #                                window. exp_055/057-v1 OMITTED these, so the
    #                                shaping was distorting the \boxed control tokens
    #                                (the exact failure mode the mask is meant to
    #                                prevent). The digits inside \boxed{N} stay shaped
    #                                (they are the model's actual answer = content).
    from src.format_tag_mask import encode_tag_patterns
    format_tag_patterns = encode_tag_patterns(
        tokenizer,
        ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "\\boxed{", "}"],
    )
    print(f"[tagmask] {len(format_tag_patterns)} format-tag patterns:")
    for pat in format_tag_patterns:
        print(f"           {pat}  -> {tokenizer.decode(pat)!r}")

    trainer = build_trainer(method, model, tokenizer, grpo_args, dataset,
                            reward_funcs, format_tag_patterns=format_tag_patterns)

    print(f"Starting [{method}] training...")
    print(f"  sequences/step = "
          f"{TRAINING_CONFIG['num_generations'] * TRAINING_CONFIG['per_device_train_batch_size']}")
    trainer.train()
    print(f"Done [{method}]. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
