#!/usr/bin/env python3
"""C2 — BoN with Skywork-Reward-Qwen-2.5-7B-v0.2 reward model.

Loads Skywork-Reward-Qwen-2.5-7B-v0.2 via transformers and uses it for BoN
selection. Compares against SAGE on MATH-500 and IFEval.

Usage:
    python run_updated_rm.py \
        --ip localhost --port 9090 \
        --model-name Qwen/Qwen3-8B-FP8 \
        --rm-device cuda:0 \
        --benchmark math500 \
        --method bon_rm \
        --num-samples 500 \
        --output-path /path/to/outputs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REBUTTAL_ROOT = Path(__file__).resolve().parents[2]
if str(_REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_REBUTTAL_ROOT))

from core.vllm_client import get_engine
from core.math500_eval import Math500Eval
from core.ifeval_eval import IFEval
from core.helpers import ensure_output_dir
from experiments.sage.solver import process_query, load_prompt, load_configurations


RM_MODEL_NAME = "Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2"


def load_reward_model(device: str = "cuda:0"):
    """Load the Skywork reward model using transformers."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"[C2] Loading reward model {RM_MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(RM_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        RM_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        num_labels=1,
    )
    model.eval()
    print("[C2] Reward model loaded.")
    return model, tokenizer


def score_with_rm(model, tokenizer, prompt: str, responses: List[str], device: str) -> List[float]:
    """Score each response with the reward model. Returns a list of scalar reward scores."""
    import torch

    scores = []
    with torch.no_grad():
        for response in responses:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            score = float(output.logits[0, 0].item())
            scores.append(score)
    return scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BoN with Skywork RM vs SAGE")
    p.add_argument("--ip", type=str, default="localhost")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B-FP8")
    p.add_argument("--rm-device", type=str, default="cuda:0",
                   help="Device for the reward model (e.g. cuda:0, cpu)")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--continue-from", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-candidates", type=int, default=7)
    p.add_argument("--num-optimization-epochs", type=int, default=2)
    p.add_argument(
        "--benchmark",
        type=str,
        choices=["math500", "ifeval"],
        default="math500",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["bon_rm", "sage"],
        default="bon_rm",
        help="bon_rm: Best-of-N with Skywork RM; sage: SAGE",
    )
    p.add_argument(
        "--judge-prompt-math500",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config-math500",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "math500_judge_config.json"),
    )
    p.add_argument(
        "--judge-prompt-ifeval",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_prompt.txt"),
    )
    p.add_argument(
        "--judge-config-ifeval",
        type=str,
        default=str(_REBUTTAL_ROOT / "configs" / "ifeval_judge_config.json"),
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    base_url = f"http://{args.ip}:{args.port}/v1"
    out_dir = ensure_output_dir(args.output_path)

    if args.benchmark == "math500":
        judge_prompt_template = load_prompt(args.judge_prompt_math500)
        judge_configurations = load_configurations(args.judge_config_math500)
    else:
        judge_prompt_template = load_prompt(args.judge_prompt_ifeval)
        judge_configurations = load_configurations(args.judge_config_ifeval)

    print(f"[C2] model={args.model_name}, benchmark={args.benchmark}, method={args.method}")

    engine = get_engine(base_url=base_url, model=args.model_name, timeout=300.0, type="aug")

    GEN_PARAMS = {
        "n": args.n_candidates,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": args.seed,
        "max_tokens": 4096,
    }

    if args.method == "bon_rm":
        # Load reward model once (not inside the async solver to avoid repeated loading)
        rm_model, rm_tokenizer = load_reward_model(args.rm_device)

        async def solver(prompt: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            texts = await engine.agenerate(prompt, **GEN_PARAMS)
            if isinstance(texts, str):
                texts = [texts]
            rm_scores = score_with_rm(rm_model, rm_tokenizer, prompt, texts, args.rm_device)
            best_idx = rm_scores.index(max(rm_scores))
            elapsed = time.perf_counter() - t0
            return {
                "output": texts[best_idx],
                "time_s": elapsed,
                "rm_scores": rm_scores,
                "best_rm_score": rm_scores[best_idx],
                "n_candidates": len(texts),
            }
    else:
        async def solver(prompt: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            result = await process_query(
                prompt,
                judge_prompt_templates=[judge_prompt_template],
                judge_configurations=[judge_configurations],
                num_optimization_epochs=args.num_optimization_epochs,
                base_url=base_url,
                model_name=args.model_name,
                number_of_gens_per_epoch=args.n_candidates,
            )
            result["time_s"] = time.perf_counter() - t0
            return result

    wall_start = time.perf_counter()
    if args.benchmark == "math500":
        evaluator = Math500Eval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(out_dir),
            continue_from_file=args.continue_from,
            max_workers=args.batch_size,
        )
        summary = await evaluator.run_async(solver=solver)
        print(f"\n[C2] MATH-500 accuracy ({args.method}): {summary['accuracy']:.4f}")
    else:
        evaluator = IFEval(
            number_of_samples_to_test=args.num_samples,
            seed=args.seed,
            output_path=str(out_dir),
            continue_from_file=args.continue_from,
            batch_size=args.batch_size,
        )
        summary = await evaluator.run_async(solver=solver)
        print(f"\n[C2] IFEval prompt accuracy ({args.method}): {summary['prompt_accuracy']:.4f}")
        print(f"[C2] IFEval instruction accuracy ({args.method}): {summary['instruction_accuracy']:.4f}")

    summary["method"] = args.method
    summary["benchmark"] = args.benchmark
    summary["model"] = args.model_name
    summary["rm_model"] = RM_MODEL_NAME if args.method == "bon_rm" else None
    summary["wall_time_s"] = time.perf_counter() - wall_start

    result_path = out_dir / f"updated_rm_{args.benchmark}_{args.method}.json"
    result_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[C2] Results saved to: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
