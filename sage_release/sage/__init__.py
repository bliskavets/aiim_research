"""SAGE — Self-judged Aspect-Guided Exploration for test-time alignment.

SAGE refines a frozen LLM's outputs at inference time using the *same* model as
both generator and soft judge: candidates are scored with a contrastive
label-margin read from a judging prompt, aggregated across aspects into a
continuous preference signal, grouped into best/worst sets, and distilled into a
short textual "gradient" that steers the next round of sampling — all under a
fixed token budget and with no external reward model.

Quickstart
----------
>>> import asyncio
>>> from sage import process_query, load_preset
>>> kwargs = load_preset("math500")
>>> out = asyncio.run(process_query("What is 2+2?", model_name="Qwen/Qwen3-8B-FP8", **kwargs))
>>> out["output"]
"""
from sage.solver import (
    process_query,
    run_optimization_epoch,
    generate_new_candidates,
    score_and_parse,
    get_contrastive_score,
    get_verified_group,
    form_best_and_worst_groups_strict,
    form_best_and_worst_groups_relaxed,
    IMPROVEMENT_PROMPT,
    APPLY_RECOMMENDATIONS_PROMPT,
    DEFAULT_GEN_PARAMS,
    DEFAULT_JUDGE_PARAMS,
)
from sage.engine import get_engine
from sage.config import load_preset, PRESETS, BEST_SETTINGS

__version__ = "1.0.0"

__all__ = [
    "process_query",
    "run_optimization_epoch",
    "generate_new_candidates",
    "score_and_parse",
    "get_contrastive_score",
    "get_verified_group",
    "form_best_and_worst_groups_strict",
    "form_best_and_worst_groups_relaxed",
    "get_engine",
    "load_preset",
    "PRESETS",
    "BEST_SETTINGS",
    "IMPROVEMENT_PROMPT",
    "APPLY_RECOMMENDATIONS_PROMPT",
    "DEFAULT_GEN_PARAMS",
    "DEFAULT_JUDGE_PARAMS",
    "__version__",
]
