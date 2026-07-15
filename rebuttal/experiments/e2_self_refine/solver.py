"""E2 - Self-Refine and Reflexion baselines (same-model, matched budget).

These are the two same-model iterative-refinement baselines requested by reviewer
k8B9. Both reuse the SAGE generation engine (Qwen3 non-thinking via AugEngine) but
replace SAGE's contrastive-margin group signal with a free-form self-critique:

  Self-Refine (Madaan 2023): generate -> self-feedback -> revise, iterated.
  Reflexion (Shinn 2023): same loop but the model keeps a growing scratchpad of
      verbal self-reflections across attempts instead of only the latest feedback.

Budget matching: SAGE runs number_of_gens_per_epoch generations for the initial
round plus one per optimization epoch, i.e. G = n_gens * (1 + epochs) solution
generations (default 7 * (1 + 2) = 21). We match on solution generations: the loop
produces exactly `budget` solutions (1 initial + budget-1 refinements). Self-critique
calls are additional and are counted separately in `num_critiques` so the accounting
is transparent (these methods are inherently sequential, unlike SAGE's parallel
group generation; wall-clock reflects that).
"""
from __future__ import annotations

from typing import Any, Dict, List
from copy import deepcopy

from core.vllm_client import get_engine


GEN_PARAMS: Dict[str, Any] = {
    "n": 1,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 7,
    "max_tokens": 4096,
}
CRITIQUE_PARAMS: Dict[str, Any] = {
    "n": 1,
    "temperature": 0.3,
    "top_p": 0.95,
    "seed": 7,
    "max_tokens": 1024,
}

FEEDBACK_TEMPLATE = (
    "You are reviewing a candidate response to a problem. Point out any mistakes, "
    "unjustified steps, or ways the response could be wrong or improved. Be concrete "
    "and specific. If the response is already fully correct and needs no changes, "
    "reply with exactly the single line: NO_ERRORS\n\n"
    "Problem:\n{problem}\n\nCandidate response:\n{answer}\n\nCritique:"
)

REFINE_TEMPLATE = (
    "Improve the response to the problem using the critique. Output ONLY the improved "
    "response itself: complete, self-contained, and directly answering the problem in "
    "the exact format the problem requires (for a math problem, end with the final "
    "answer in \\boxed{{}}). Do NOT output any commentary, review, or notes about the "
    "critique or the quality of the response.\n\n"
    "Problem:\n{problem}\n\nCurrent response:\n{answer}\n\nCritique:\n{feedback}\n\n"
    "Improved response:"
)

REFLEXION_REFINE_TEMPLATE = (
    "You are retrying a problem. Below are your own reflections from previous attempts. "
    "Use them to avoid repeating mistakes. Output ONLY the new response itself: "
    "complete, self-contained, and directly answering the problem in the exact format "
    "the problem requires (for a math problem, end with the final answer in \\boxed{{}}). "
    "Do NOT output commentary or notes.\n\n"
    "Problem:\n{problem}\n\nMost recent attempt:\n{answer}\n\n"
    "Reflections from previous attempts:\n{reflections}\n\nNew response:"
)

# Critique phrases that indicate the model found no errors (Self-Refine stop signal).
_NO_ERROR_SIGNALS = (
    "no_errors", "no errors", "no mistakes", "already correct", "already fully correct",
    "solution is correct", "response is correct", "no issues", "no changes needed",
)


def _first(texts) -> str:
    if isinstance(texts, str):
        return texts
    if isinstance(texts, (list, tuple)) and texts:
        return texts[0] if isinstance(texts[0], str) else str(texts[0])
    return str(texts)


async def refine_query(
    query: str,
    *,
    base_url: str = "http://localhost:9090/v1",
    model_name: str = "Qwen/Qwen3-8B-FP8",
    mode: str = "self_refine",
    budget: int = 21,
    gen_params: Dict[str, Any] | None = None,
    critique_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run Self-Refine or Reflexion for `budget` solution generations.

    Returns a dict with keys: output (final solution), num_generations,
    num_critiques, all_answers, mode.
    """
    if mode not in ("self_refine", "reflexion"):
        raise ValueError(f"mode must be 'self_refine' or 'reflexion', got {mode!r}")
    gen_params = {**GEN_PARAMS, **(gen_params or {})}
    critique_params = {**CRITIQUE_PARAMS, **(critique_params or {})}

    engine = get_engine(base_url=base_url, model=model_name, timeout=300, type="aug")

    all_answers: List[str] = []
    reflections: List[str] = []
    num_generations = 0
    num_critiques = 0

    current = _first(await engine.agenerate(query, **gen_params))
    num_generations += 1
    all_answers.append(current)

    stopped_early = False
    while num_generations < budget:
        feedback = _first(await engine.agenerate(
            FEEDBACK_TEMPLATE.format(problem=query, answer=current), **critique_params
        ))
        num_critiques += 1

        # Self-Refine stop condition: if the critique reports no errors, keep the
        # current answer rather than over-iterating into drift (Madaan 2023).
        fb_head = feedback.strip().lower()[:80]
        if any(sig in fb_head for sig in _NO_ERROR_SIGNALS):
            stopped_early = True
            break

        if mode == "reflexion":
            reflections.append(feedback.strip())
            refine_prompt = REFLEXION_REFINE_TEMPLATE.format(
                problem=query,
                answer=current,
                reflections="\n".join(f"- {r}" for r in reflections),
            )
        else:
            refine_prompt = REFINE_TEMPLATE.format(
                problem=query, answer=current, feedback=feedback
            )

        # Vary the seed per refinement so retries are not identical.
        gp = deepcopy(gen_params)
        gp["seed"] = gp.get("seed", 7) + num_generations
        current = _first(await engine.agenerate(refine_prompt, **gp))
        num_generations += 1
        all_answers.append(current)

    return {
        "output": current,
        "num_generations": num_generations,
        "num_critiques": num_critiques,
        "all_answers": all_answers,
        "mode": mode,
        "stopped_early": stopped_early,
    }
