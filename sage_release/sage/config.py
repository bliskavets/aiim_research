"""Benchmark presets and the inference-time budget used in the paper.

Each preset bundles (i) the judge prompt template(s), (ii) the judge scoring
configuration (aspect tags, polarity, and weights), and (iii) the shared
inference-time budget. Loading a preset returns keyword arguments that can be
splatted directly into :func:`sage.solver.process_query`.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict, List

_PKG = "sage"


def _read_prompt(name: str) -> str:
    return (resources.files(_PKG) / "prompts" / name).read_text(encoding="utf-8")


def _read_config(name: str) -> Dict[str, Any]:
    return json.loads((resources.files(_PKG) / "configs" / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Shared inference-time budget (paper setting): N = 7 hypotheses per stage,
# two optimization epochs, automatic best/worst group formation (m_min = 1),
# a temperature schedule for generation, and a low-temperature judge.
# ----------------------------------------------------------------------------
BEST_SETTINGS: Dict[str, Any] = {
    "number_of_gens_per_epoch": 7,
    "num_optimization_epochs": 2,
    "m_min": 1,
    "engine_type": "aug",  # non-thinking chat template; use "think" for reasoning-mode runs
    "judge_params": {
        "n": 1,
        "temperature": 0.1,
        "top_p": 0.95,
        "seed": 7,
        "max_tokens": 4096,
    },
}


# ----------------------------------------------------------------------------
# Per-benchmark judge recipes.
# ----------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, List[str]]] = {
    # Verifiable math: a single verification aspect is strongest (see paper, Sec. Ablations).
    "math500": {"prompts": ["math_verification.txt"], "configs": ["math500.json"]},
    # Open-ended preference: multi-aspect tagging (usefulness / completeness / non-repetition).
    "alpacaeval": {"prompts": ["alpaca_attributes.txt"], "configs": ["alpacaeval.json"]},
    # Strict instruction following: single yes/no verdict aspect.
    "ifeval": {"prompts": ["instruction_following.txt"], "configs": ["general_verdict.json"]},
}


def load_preset(name: str, **overrides: Any) -> Dict[str, Any]:
    """Return process_query() kwargs for a named benchmark preset.

    Args:
        name: one of ``sage.config.PRESETS``.
        **overrides: any keyword that overrides ``BEST_SETTINGS`` (e.g. ``engine_type="think"``).

    Returns:
        A dict ready to splat into :func:`sage.solver.process_query`, e.g.::

            kwargs = load_preset("math500")
            result = await process_query(query, model_name=..., base_url=..., **kwargs)
    """
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    preset = PRESETS[name]
    kwargs: Dict[str, Any] = dict(BEST_SETTINGS)
    kwargs.update(overrides)
    kwargs["judge_prompt_templates"] = [_read_prompt(p) for p in preset["prompts"]]
    kwargs["judge_configurations"] = [_read_config(c) for c in preset["configs"]]
    return kwargs
