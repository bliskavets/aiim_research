"""
exp_056 — Search-R1 with our 4 shaping candidates (Qwen3-4B, unsloth+TRL)
========================================================================

Model interleaves <think>, <search>, retrieved <information>, <answer>
to answer open-domain QA (PeterJinGo/nq_hotpotqa_train). The custom
multi-turn rollout lives in `src/searchr1_rollout.py` and is wired into
TRL via `src/searchr1_trainer.py::SearchR1GRPOTrainer`. The 3 shaping
trainers inherit from that base, so their `_compute_loss` overrides
are unchanged from exp_054/055.

Retriever
---------
Default is `StubRetriever` — returns mock docs. Validates the pipeline
end-to-end without the heavy E5+wiki-18 infrastructure. Once the
retrieval server from `retrieval/README.md` is up, switch to
`HTTPRetriever(url=...)` (one line below).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from src.searchr1_trainer import SearchR1GRPOTrainer
from src.searchr1_rollout import RolloutConfig
from src.retriever import StubRetriever, HTTPRetriever  # noqa: F401
from src.em_score import reward_em
from src.format_tag_mask import encode_tag_patterns


# =============================================================================
# CONFIG
# =============================================================================
SEED = 3407

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-4B",
    "max_seq_length": 4608,          # 512 prompt + 4096 completion (lowered from 6656: OOM in backward
                                     # — Search-R1 keeps all rollouts near-max with injected <information>)
    "lora_rank": 64,
    "load_in_4bit": False,
    "fast_inference": True,
    # env-overridable: grpo ran fine at 0.32, but the per-token shaped methods
    # carry a bit more live memory and OOM'd the backward on the long step-785
    # batch (missed by ~350 MiB). Lower for shaped runs (e.g. 0.25) to free GPU
    # for the backward — pure memory knob, does not change results.
    "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", 0.32)),
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
    "gradient_accumulation_steps": 4,    # global batch = 4 prompts/grad-update
    "num_generations": int(os.environ.get("SMOKE_NUM_GEN", 4)),
    "max_steps": int(os.environ.get("SMOKE_MAX_STEPS", 1000)),
    "save_steps": 9999,
    "max_grad_norm": 1.0,
    "report_to": "none",
    "seed": SEED,
    "bf16": True,
    "fp16": False,
}

DATASET_CONFIG = {
    "name": "PeterJinGo/nq_hotpotqa_train",
    "split": "train",
    "max_prompt_tokens": 512,
    "subset_size": int(os.environ.get("SMOKE_SUBSET", 2000)),
    "shuffle_seed": SEED,
}

ROLLOUT_CONFIG = RolloutConfig(
    max_turns=4,
    topk=3,
    max_completion_tokens=4096,    # matches MODEL_CONFIG max_seq budget
    per_turn_max_tokens=1280,
    temperature=0.7,
    top_p=0.95,
    seed=None,    # MUST be None — a fixed seed makes the num_generations
                  # rollouts of a prompt identical, zeroing GRPO's within-group
                  # advantage (see RolloutConfig.seed note). Data-order
                  # reproducibility still comes from GRPOConfig(seed=SEED).
)

# conf_micro_bs: chunk size for the per-token-confidence second forward in the
# GTPO trainers. Caps peak memory of the full-vocab (B,L,V) fp32 logits tensor
# that otherwise OOMs the backward on long Search-R1 rollouts. Lower = safer.
_CONF_MICRO_BS = int(os.environ.get("CONF_MICRO_BS", 2))

SHAPING_CONFIG = {
    "grpo_s_entropy":   {"beta1": 1.0, "beta2": 0.1, "reward_threshold": 0.0},
    "gtpo_conf":        {"alpha1": 1.0, "alpha2": 0.1, "top_k": 20, "reward_threshold": 0.0,
                         "conf_micro_bs": _CONF_MICRO_BS},
    "gtpo_ema_flipped": {"alpha1": 0.9, "alpha2": 0.1, "lam": 0.9, "top_k": 20, "reward_threshold": 0.0,
                         "conf_micro_bs": _CONF_MICRO_BS},
}

# Tag mask: structural tokens that shouldn't carry per-token shaping signal.
# Search-R1 protocol tags + Qwen3 chat tokens.
TAG_STRINGS = [
    "<think>", "</think>",
    "<search>", "</search>",
    "<information>", "</information>",
    "<answer>", "</answer>",
    "<|im_start|>", "<|im_end|>",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question by reasoning step by step. "
    "If you need external information, issue a search query like "
    "`<search> your query </search>`. Search results will be returned inside "
    "`<information>...</information>` blocks. Use as many search calls as you "
    "need (up to a small number). Once you have enough information, write "
    "your reasoning inside `<think>...</think>` tags and produce the final "
    "answer inside `<answer>...</answer>` tags. Keep the answer short."
)

# Retriever: StubRetriever for smoke runs without infra; swap for HTTPRetriever
# once retrieval/retrieval_launch.sh is up on port 8000.
DEFAULT_RETRIEVER = StubRetriever()


# =============================================================================
# DATASET
# =============================================================================
def prepare_dataset():
    # Load only train.parquet directly. The repo's test.parquet has a
    # different schema (HotpotQA-style with supporting_facts, question_decomposition,
    # etc.) which crashes HF's default loader when it tries to auto-detect
    # the splits, even if we only ask for split='train'.
    ds = load_dataset(
        "parquet",
        data_files={"train": f"hf://datasets/{DATASET_CONFIG['name']}/train.parquet"},
        split="train",
        token=os.environ.get("HF_TOKEN"),
    )
    ds = ds.shuffle(seed=DATASET_CONFIG["shuffle_seed"])
    ds = ds.select(range(min(DATASET_CONFIG["subset_size"], len(ds))))
    # PeterJinGo/nq_hotpotqa_train uses fields: question, golden_answers (list[str])
    ds = ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": x["golden_answers"],
    })
    print(f"Dataset: {len(ds)} examples (Search-R1 NQ+HotpotQA, seed={DATASET_CONFIG['shuffle_seed']})")
    return ds


# =============================================================================
# REWARD
# =============================================================================
_print_counter = 0
PRINT_EVERY_STEPS = 10


def reward_searchr1_em(completions, gold_answers):
    """Reward fn used by SearchR1GRPOTrainer.

    completions: list[str] of full completion text
    gold_answers: list of (str | list[str]) acceptable gold answers
    Returns list[float] aligned with completions.
    """
    global _print_counter
    scores = [reward_em(c, g) for c, g in zip(completions, gold_answers)]
    if _print_counter % PRINT_EVERY_STEPS == 0 and len(completions) > 0:
        # debug print for the first rollout in this batch
        first = completions[0]
        ans = first.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()[:60] if "<answer>" in first else "(no <answer>)"
        gold = gold_answers[0] if isinstance(gold_answers[0], str) else (gold_answers[0][0] if gold_answers[0] else "?")
        print(f"[Step {_print_counter}] gold={str(gold)[:40]!r} | pred={ans!r} | reward={scores[0]:+.1f}")
    _print_counter += 1
    return scores


# =============================================================================
# TRAINER FACTORY
# =============================================================================
def build_trainer(method, model, tokenizer, args, dataset,
                  retriever, rollout_cfg, reward_fn, format_tag_patterns=None):
    common = dict(
        model=model, tokenizer=tokenizer, args=args, train_dataset=dataset,
        reward_funcs=[],   # we compute reward inside the trainer override
        retriever=retriever, rollout_cfg=rollout_cfg, reward_fn=reward_fn,
    )
    if method == "grpo":
        return SearchR1GRPOTrainer(**common)
    if method == "grpo_s_entropy":
        from src.grpo_s_trainer import GRPOSTrainer
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
    ap.add_argument("--retriever", default="stub", choices=["stub", "http"],
                    help="`stub` = mock docs (smoke runs); `http` = Search-R1 server on :8000")
    args_cli = ap.parse_args()
    method = args_cli.method

    output_dir = f"/workspace/exp_056_searchr1_qwen3_grpo_vs_shaped/outputs_{method}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== exp_056 [{method}] — Search-R1 NQ+HotpotQA, Qwen3-4B native format ===")

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

    # ── max prompt length from data
    lengths = []
    for ex in dataset:
        toks = tokenizer.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=True)
        lengths.append(len(toks))
    lengths.sort()
    p99_len = lengths[int(0.99 * len(lengths))]
    max_prompt_length = min(p99_len + 1, DATASET_CONFIG["max_prompt_tokens"])
    print(f"Max prompt length (99%, capped): {max_prompt_length}")

    grpo_args = GRPOConfig(
        max_prompt_length=max_prompt_length,
        # IMPORTANT: GRPOConfig still needs a max_completion_length even though we override
        # generation; the trainer uses it as a buffer size hint. Set to rollout budget.
        max_completion_length=ROLLOUT_CONFIG.max_completion_tokens,
        output_dir=output_dir,
        **TRAINING_CONFIG,
    )

    # Tag-mask patterns for the per-token shaping trainers
    format_tag_patterns = encode_tag_patterns(tokenizer, TAG_STRINGS)
    print(f"[tagmask] {len(format_tag_patterns)} patterns from {len(TAG_STRINGS)} tag strings")
    for pat in format_tag_patterns:
        print(f"           {pat}  -> {tokenizer.decode(pat)!r}")

    # Retriever
    if args_cli.retriever == "stub":
        retriever = StubRetriever()
        print("[retriever] StubRetriever — mock docs (smoke mode)")
    else:
        retriever = HTTPRetriever(url=os.environ.get("RETRIEVAL_URL", "http://127.0.0.1:8000/retrieve"))
        print(f"[retriever] HTTPRetriever -> {retriever.url}")

    trainer = build_trainer(
        method, model, tokenizer, grpo_args, dataset,
        retriever=retriever, rollout_cfg=ROLLOUT_CONFIG, reward_fn=reward_searchr1_em,
        format_tag_patterns=format_tag_patterns,
    )

    print(f"Starting [{method}] training...")
    trainer.train()
    print(f"Done [{method}]. Saved to: {output_dir}")


if __name__ == "__main__":
    main()
