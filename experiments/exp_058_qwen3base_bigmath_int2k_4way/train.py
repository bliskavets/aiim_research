"""
exp_058 — Qwen3-4B-BASE (pretrained, NOT instruction-tuned), 4 methods on Big-Math int-2000
======================================================================================

The exp_051 recipe (grpo + 3 shaped candidates on SynthLabsAI/Big-Math-RL-Verified
integer-answer first-2000) re-run on the **base** model `Qwen/Qwen3-4B-Base`
instead of the instruction-tuned `Qwen/Qwen3-4B`.

Goal: see how the shaping methods behave on a NON-converged / non-instruction-tuned
model — one that has not been aligned to follow the "reason then \\boxed{}" format,
so GRPO has to bootstrap both the format AND the answer. (Instruct Qwen3 already
near-saturates the easy Big-Math slice; the base model should be far from it.)

IMPORTANT: this uses the FIXED (non-bypassed) shaped trainers — they override
`compute_loss` and inject the per-token shaped advantage into unsloth's compiled
loss (see src/shaped_loss.py and exp_057's SHAPING_BYPASS_BUGFIX.md). The chunked
no-grad confidence/entropy forward (conf_micro_bs, seq-chunked log_softmax) keeps
it within A100-80GB. The base Qwen3 tokenizer is the SAME as instruct (chat
template present; <think>/<|im_start|>/<|im_end|> are single special tokens), so
the native-format + tag-mask setup transfers unchanged.

Held identical to exp_055/057 otherwise:
  SYSTEM_PROMPT  "Solve step by step. Final integer answer in \\boxed{}."
  reward funcs   reward_format_thinking + reward_answer_boxed + reward_answer_numeric
  tag-mask       <think>, </think>, <|im_start|>, <|im_end|>
  max_seq=6656 (512 + 6144), gpu_memory_utilization=0.40, ng=8
  bs=1, ga=4, lr=5e-6 cosine, max_steps=1000, seed=3407

Dataset: SynthLabsAI/Big-Math-RL-Verified, integer-answer filter, shuffle seed
3407, first 2000 (identical selection to exp_051).

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
    "model_name": "Qwen/Qwen3-4B-Base",   # exp_058: BASE (pretrained, NOT instruction-tuned)
    "max_seq_length": 4096,          # exp_051 config (512 prompt + 3584 completion)
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    "gpu_memory_utilization": 0.55,  # exp_051 config (moot under unsloth standby, kept for record)
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
    "num_generations": 4,                # exp_051 config (was ng=8 from exp_057 -> 127 s/it on base; ng=4 halves gen)
    "max_steps": int(os.environ.get("SMOKE_MAX_STEPS", 1000)),  # env-overridable (cap < disk/time budget)
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "seed": SEED,
    "bf16": True,
    "fp16": False,
}

DATASET_CONFIG = {
    "name": "SynthLabsAI/Big-Math-RL-Verified",   # exp_051 dataset
    "split": "train",
    "max_prompt_tokens": 512,
    "max_completion_tokens": 3584,   # exp_051 config (base answers directly ~640 tok; caps the rare ramble tail)
    "subset_size": 2000,             # integer-answer filter then first 2000 (shuffled seed 3407)
    "shuffle_seed": SEED,
}

# conf_micro_bs: batch chunk for the no-grad confidence/entropy forward in the
# shaped trainers. =1 is safest on A100 (base model rambles -> long completions).
_CONF_MICRO_BS = int(os.environ.get("CONF_MICRO_BS", 1))

# length_L: the length-penalty knee for the two lenpen methods. Env-overridable
# so an L-sweep (e.g. 3096 / 2048 / 1536) can re-run the same code without edits.
_LENGTH_L = int(os.environ.get("LENGTH_L", 1024))

# Method-native shaping coefficients (intrinsic to each method, from its source exp)
SHAPING_CONFIG = {
    "grpo_s_entropy":   {"beta1": 1.0, "beta2": 0.1, "reward_threshold": 0.0, "conf_micro_bs": _CONF_MICRO_BS},
    "gtpo_conf":        {"alpha1": 1.0, "alpha2": 0.1, "top_k": 20, "reward_threshold": 0.0, "conf_micro_bs": _CONF_MICRO_BS},
    "gtpo_ema_flipped": {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20, "reward_threshold": 0.0, "conf_micro_bs": _CONF_MICRO_BS},
    # NEW #1: gtpo_ema_flipped + Lagrange-like length penalty alpha_len*max(0,|o|-L).
    # L=1024 (0 < L < max_completion 3584; base answers ~640 tok, leaves headroom).
    # alpha_len=0.0015 (~2000-tok overshoot ≈ -3 advantage, cancels a boxed hit).
    "gtpo_ema_lenpen": {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20, "reward_threshold": 0.0,
                        "conf_micro_bs": _CONF_MICRO_BS, "alpha_len": 0.005, "length_L": _LENGTH_L},
    # NEW #2: same penalty, GATED by low-temperature success. Per prompt we also
    # sample 2 extra (unused-for-update) completions at t=0 and t2=0.5; if either
    # gives the exact answer, the problem is concisely solvable -> apply the length
    # penalty to that prompt's training completions; else skip it.
    "gtpo_ema_lenpen_gated": {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20, "reward_threshold": 0.0,
                              "conf_micro_bs": _CONF_MICRO_BS, "alpha_len": 0.005, "length_L": _LENGTH_L,
                              "gate_temps": (0.0, 0.5), "gate_max_tokens": 3584},
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
          f"(Big-Math-RL-Verified, shuffled seed={DATASET_CONFIG['shuffle_seed']})")
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
    if method == "gtpo_ema_lenpen":
        from src.gtpo_ema_lenpen_trainer import GTPOEMAFlippedLenPenTrainer
        return GTPOEMAFlippedLenPenTrainer(**common, **SHAPING_CONFIG["gtpo_ema_lenpen"],
                                           format_tag_patterns=format_tag_patterns)
    if method == "gtpo_ema_lenpen_gated":
        from src.gtpo_ema_lenpen_gated_trainer import GTPOEMAFlippedLenPenGatedTrainer
        return GTPOEMAFlippedLenPenGatedTrainer(
            **common, **SHAPING_CONFIG["gtpo_ema_lenpen_gated"],
            format_tag_patterns=format_tag_patterns,
            answer_extractor=_extract_boxed_answer)
    raise ValueError(f"unknown method: {method}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["grpo", "grpo_s_entropy", "gtpo_conf", "gtpo_ema_flipped", "gtpo_ema_lenpen", "gtpo_ema_lenpen_gated"])
    args_cli = ap.parse_args()
    method = args_cli.method
    reward_funcs = REWARD_FUNCS_FULL  # exp_055 uses full reward set, same as exp_050/051/052

    output_dir = os.path.join(os.path.dirname(__file__), f"outputs_{method}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== exp_058 [{method}] — Big-Math int-2000, Qwen3-4B-BASE (tagmasked shaping, FIXED non-bypassed) ===")
    if method in ("gtpo_ema_lenpen", "gtpo_ema_lenpen_gated"):
        print(f"  length penalty: alpha_len={SHAPING_CONFIG[method]['alpha_len']}  L={SHAPING_CONFIG[method]['length_L']}")
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

    # Qwen3-4B-Base: unsloth's loader returns the tokenizer WITHOUT a
    # chat_template (the base repo ships one but it isn't wired in here), so
    # apply_chat_template would crash. Set the standard Qwen3 ChatML+thinking
    # template explicitly — the base model has to LEARN to follow it via RL,
    # which is exactly the non-instruction-tuned test we want. Prefer the base
    # repo's own template; fall back to the instruct Qwen3-4B template.
    if getattr(tokenizer, "chat_template", None) is None:
        from transformers import AutoTokenizer
        tmpl = None
        for src in (MODEL_CONFIG["model_name"], "Qwen/Qwen3-4B"):
            try:
                tmpl = AutoTokenizer.from_pretrained(
                    src, token=os.environ.get("HF_TOKEN")).chat_template
                if tmpl:
                    print(f"[chat_template] set from {src} (unsloth loader had none)")
                    break
            except Exception as e:
                print(f"[chat_template] {src} failed: {repr(e)[:80]}")
        if not tmpl:
            raise RuntimeError("no chat_template available for the base model")
        tokenizer.chat_template = tmpl

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
    # exp_055 uses ONLY Qwen3 native structural tokens. No custom format tags.
    # Mask covers 4 substrings — all of which Qwen3 tokenises as single
    # special token ids (so the mask is one-token-precise):
    #   - <think>, </think>    : thinking-mode protocol (151667 / 151668)
    #   - <|im_start|>, <|im_end|>: ChatML role boundaries
    # These are the tokens where the model is extremely peaked; without
    # masking, per-token shaping concentrates bonus/penalty on them and
    # distorts the gradient that should be teaching content reasoning.
    from src.format_tag_mask import encode_tag_patterns
    format_tag_patterns = encode_tag_patterns(
        tokenizer,
        ["<think>", "</think>", "<|im_start|>", "<|im_end|>"],
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
