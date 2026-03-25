"""Core SAGE algorithm: self-refinement via aspect-guided exploration.

This module provides the process_query() entry point and all supporting
functions for the SAGE best-vs-worst contrastive refinement loop.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.vllm_client import get_engine


# ============================================================
# Prompts
# ============================================================

IMPROVEMENT_PROMPT = """You are an experienced problem solver.
You are given a problem and several solutions to it. These are best solutions and worst solutions.
You have to understand what differentiates the best solutions from the worst solutions. Then, write
the recommendations on how to improve any solution.

Problem:
{problem}

Best solutions:
{best_solutions}

Worst solutions:
{worst_solutions}

You should output your reasoning step by step between tags <reasoning> and </reasoning>, then output the final recommendations between tags <recommendations> and </recommendations> like this:
<reasoning>
..
</reasoning>
<recommendations>
...
</recommendations>"""

APPLY_RECOMMENDATIONS_PROMPT = """You are an experienced problem solver.
You are given a problem and several solutions to it. You are also given recommendations on how to improve any solution.
You have to apply these recommendations and create a new solution that is better than the best of all solutions.

Problem:
{problem}

Best solutions:
{best_solutions}

Recommendations:
{recommendations}

You should think on how to apply recommendations to write the improved solution.
Output your reasoning steps on writing an improvement response between <reasoning> and </reasoning>, then output the refined solution between tags <refined_solution> and </refined_solution> like this:
<reasoning>
..
</reasoning>
<refined_solution>
...
</refined_solution>"""


# ============================================================
# Default generation / judge parameters
# ============================================================

DEFAULT_GEN_PARAMS: Dict[str, Any] = {
    "n": 1,
    "temperature": 0.8,
    "top_p": 0.95,
    "seed": 7,
    "max_tokens": 4096,
}

DEFAULT_JUDGE_PARAMS: Dict[str, Any] = {
    "n": 1,
    "temperature": 0.1,
    "top_p": 0.95,
    "seed": 7,
    "max_tokens": 4096,
}

SINGLE_RESPONSE_PARAMS: Dict[str, Any] = {
    "n": 1,
    "temperature": 0.2,
    "top_p": 0.95,
    "seed": 7,
    "max_tokens": 1536,
}


# ============================================================
# Utility functions
# ============================================================

def load_prompt(prompt_or_path: str) -> str:
    """Load a prompt from a file path or return it directly if it is a string."""
    if os.path.isfile(prompt_or_path):
        with open(prompt_or_path, "r", encoding="utf-8") as f:
            return f.read()
    return prompt_or_path


def load_configurations(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Load judge configuration JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text(maybe: Union[str, List[Any], Tuple[Any, ...], Dict[str, Any]]) -> str:
    """Recursively extract a text string from various return types."""
    if isinstance(maybe, str):
        return maybe
    if isinstance(maybe, (list, tuple)):
        return extract_text(maybe[0]) if maybe else ""
    if isinstance(maybe, dict):
        for key in ("text", "response", "answer", "output"):
            if key in maybe and isinstance(maybe[key], str):
                return maybe[key]
    return str(maybe)


def find_placeholders(text: str) -> List[str]:
    """Find all {placeholder} names in a template string."""
    return [m.group(1).strip() for m in re.compile(r"\{([^{}]+)\}").finditer(text)]


# ============================================================
# Log-probability utilities
# ============================================================

def find_opening_tag(
    tokens: List[str],
    tagname: str,
    closing_tagname: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Locate the token-span positions of opening and closing XML-like tags.

    Returns ((open_start, open_end), (close_start, close_end)) or (-1, -1) pairs on failure.
    """
    start_opening_tag_pos = end_opening_tag_pos = -1
    start_closing_tag_pos = end_closing_tag_pos = -1

    for i in range(0, len(tokens) - 1):
        if i <= end_opening_tag_pos:
            continue
        if tagname.startswith(tokens[i].strip()):
            cur_acc = tokens[i].strip()
            for j in range(i + 1, len(tokens)):
                cur_acc = cur_acc + tokens[j].strip()
                if cur_acc == tagname:
                    start_opening_tag_pos = i
                    end_opening_tag_pos = j
                    break
                elif not tagname.startswith(cur_acc):
                    break
        if (
            end_opening_tag_pos != -1
            and i > end_opening_tag_pos
            and closing_tagname.startswith(tokens[i].strip())
        ):
            cur_acc = tokens[i].strip()
            for j in range(i + 1, len(tokens)):
                cur_acc = cur_acc + tokens[j].strip()
                if cur_acc == closing_tagname:
                    start_closing_tag_pos = i
                    end_closing_tag_pos = j
                    break
                elif not closing_tagname.startswith(cur_acc):
                    break
        if start_opening_tag_pos != -1 and end_closing_tag_pos != -1:
            return (start_opening_tag_pos, end_opening_tag_pos), (start_closing_tag_pos, end_closing_tag_pos)
    return (-1, -1), (-1, -1)


def find_logprobs(
    logprobs: List[Dict[str, float]],
    tokens: List[str],
    start_pos: int,
    end_pos: int,
) -> Optional[Dict[str, float]]:
    """Return the logprob dict at the first non-empty token between start_pos and end_pos."""
    non_empty_pos: Optional[int] = None
    for i in range(start_pos, min(end_pos + 1, len(tokens))):
        if tokens[i].strip():
            non_empty_pos = i
            break
    if non_empty_pos is None:
        return None
    if non_empty_pos < 0 or non_empty_pos >= len(logprobs):
        return None
    return logprobs[non_empty_pos]


def get_contrastive_score(
    logprobs: Optional[Dict[str, float]],
    positive_labels: List[str],
    negative_labels: List[str],
) -> float:
    """Compute mean(pos_logprobs) - mean(neg_logprobs) as the contrastive score."""
    if not logprobs:
        return float("-1e10")
    min_score = min(logprobs.values()) if logprobs else -1e10
    positive_scores = [logprobs.get(l, min_score) for l in positive_labels]
    negative_scores = [logprobs.get(l, min_score) for l in negative_labels]
    return float(np.mean(positive_scores) - np.mean(negative_scores))


def get_parsed_verification_result(
    matching_logprob: Dict[str, float],
    labels: List[str],
) -> Optional[str]:
    """Return the label with the highest log-probability among the candidate labels."""
    label2score: Dict[str, float] = {}
    for label in labels:
        if label in matching_logprob:
            label2score[label] = matching_logprob[label]
    if not label2score:
        return None
    return max(label2score.items(), key=lambda x: x[1])[0]


# ============================================================
# Scoring pipeline
# ============================================================

async def score_answers_async(
    judge_engine,
    judge_prompt_template: str,
    instruction: str,
    answers: List[Dict[str, Any]],
    gen_params_judge: Dict[str, Any],
) -> None:
    """Run the judge on each answer and store raw verification results in-place."""
    placeholders = find_placeholders(judge_prompt_template)

    async def _verify_one(answer_text: str):
        prompt = judge_prompt_template.format(
            prompt=instruction,
            answer=answer_text,
            **{p: f"{{{p}}}" for p in placeholders if p not in ("prompt", "answer")},
        )
        return prompt, (await judge_engine.agenerate(prompt, logprobs=20, return_tokens=True, **gen_params_judge))

    for ans in answers:
        if "verification_result" not in ans:
            ans["verification_result"] = {}
            ans["verification_prompt"] = {}

    tasks = [_verify_one(a["answer"]) for a in answers]
    results = await asyncio.gather(*tasks)
    for ans, (prompt, r) in zip(answers, results):
        ans["verification_result"][judge_prompt_template] = r
        ans["verification_prompt"][judge_prompt_template] = prompt


def assign_scores_to_answers(
    answers: List[Dict[str, Any]],
    configuration_name: str,
    pos_labels: List[str],
    neg_labels: List[str],
    labels: List[str],
    opening_tag: str,
    closing_tag: str,
    prompt_template: str,
) -> List[Dict[str, Any]]:
    """Assign contrastive log-probability scores to each answer for one configuration."""
    for ans in answers:
        if "llm_judge_scores" not in ans:
            ans["llm_judge_scores"] = {}
        ans["llm_judge_scores"][configuration_name] = {}

    for ans in answers:
        if "verification_result" not in ans:
            ans["llm_judge_scores"][configuration_name]["matching_logprob"] = {l: -1e10 for l in labels}
            ans["llm_judge_scores"][configuration_name]["llm_judge_score"] = get_contrastive_score(
                ans["llm_judge_scores"][configuration_name]["matching_logprob"], pos_labels, neg_labels
            )
            continue

        texts, all_logprobs, all_tokens = ans["verification_result"][prompt_template]
        (open_s, open_e), (close_s, close_e) = find_opening_tag(all_tokens[0], opening_tag, closing_tag)
        matching_logprob = find_logprobs(all_logprobs[0], all_tokens[0], open_e + 1, close_s)
        if matching_logprob is None:
            matching_logprob = {l: -1e10 for l in labels}

        ans["llm_judge_scores"][configuration_name]["matching_logprob"] = matching_logprob
        ans["llm_judge_scores"][configuration_name]["llm_judge_score"] = get_contrastive_score(
            matching_logprob, pos_labels, neg_labels
        )
    return answers


def compute_final_llm_judge_score(
    ans: Dict[str, Any],
    configurations: Dict[str, Dict[str, Any]],
) -> float:
    """Compute the weighted sum of per-configuration LLM judge scores."""
    total = 0.0
    for configuration_name, configuration_info in configurations.items():
        total += float(ans["llm_judge_scores"][configuration_name]["llm_judge_score"]) * configuration_info["weight"]
    return total


def score_and_parse(
    answers: List[Dict[str, Any]],
    configurations: Dict[str, Dict[str, Any]],
    judge_prompt_templates: List[str],
) -> None:
    """Assign per-configuration scores and compute final_llm_judge_score for each answer."""
    for prompt_template in judge_prompt_templates:
        for configuration_name, configuration_info in configurations.items():
            assign_scores_to_answers(
                answers=answers,
                configuration_name=configuration_name,
                pos_labels=configuration_info["pos_labels"],
                neg_labels=configuration_info["neg_labels"],
                labels=configuration_info["labels"],
                opening_tag=configuration_info["opening_tag"],
                closing_tag=configuration_info["closing_tag"],
                prompt_template=prompt_template,
            )
    final_scores = [compute_final_llm_judge_score(a, configurations) for a in answers]
    for a, s in zip(answers, final_scores):
        a["final_llm_judge_score"] = float(s)
        parsed_map: Dict[str, Optional[str]] = {}
        for configuration_name in configurations.keys():
            parsed_map[configuration_name] = get_parsed_verification_result(
                a["llm_judge_scores"][configuration_name]["matching_logprob"],
                configurations[configuration_name]["labels"],
            )
        a["short_verification_result"] = parsed_map


# ============================================================
# Group formation
# ============================================================

def get_verified_group(
    answers: List[Dict[str, Any]],
    configurations: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Partition answers into positive, negative, and unparsed groups."""
    verified_positive: List[Dict[str, Any]] = []
    verified_negative: List[Dict[str, Any]] = []
    verified_unparsed: List[Dict[str, Any]] = []

    for ans in answers:
        parsed_map = ans.get("short_verification_result", {})
        if not parsed_map or any(parsed_map.get(c) is None for c in configurations.keys()):
            verified_unparsed.append(ans)
            continue
        is_positive = all(
            parsed_map.get(cname) in cfg["pos_labels"]
            for cname, cfg in configurations.items()
        )
        is_negative = any(
            parsed_map.get(cname) in cfg["neg_labels"]
            for cname, cfg in configurations.items()
        )
        if is_positive:
            verified_positive.append(ans)
        elif is_negative:
            verified_negative.append(ans)
        else:
            verified_unparsed.append(ans)

    return {
        "positive": verified_positive,
        "negative": verified_negative,
        "unparsed": verified_unparsed,
        "all": answers,
    }


def form_best_and_worst_groups_strict(
    verified_group: Dict[str, List[Dict[str, Any]]],
    k_best: int = 4,
    k_worst: int = 3,
    m_min: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """Select top-k_best positives and bottom-k_worst negatives (strict: requires labelled groups).

    m_min: minimum number of answers required in each group to proceed. Returns empty
    groups if either positive or negative pool has fewer than m_min members.
    """
    positives = verified_group["positive"]
    negatives = verified_group["negative"]

    if len(positives) < m_min or len(negatives) < m_min:
        return {"best": [], "worst": []}

    pos_sorted = sorted(positives, key=lambda a: a.get("final_llm_judge_score", -1e10), reverse=True)
    neg_sorted = sorted(negatives, key=lambda a: a.get("final_llm_judge_score", 1e10))
    return {
        "best": pos_sorted[:k_best],
        "worst": neg_sorted[:k_worst],
    }


def form_best_and_worst_groups_relaxed(
    verified_group: Dict[str, List[Dict[str, Any]]],
    configurations: Dict[str, Dict[str, Any]],
    k_best: int = 4,
    k_worst: int = 3,
    m_min: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """Select best and worst from the full answer pool by final score (relaxed: uses all answers).

    m_min: minimum total answers required; returns empty groups otherwise.
    """
    all_answers = verified_group["all"]
    if len(all_answers) < m_min:
        return {"best": [], "worst": []}

    all_sorted = sorted(all_answers, key=lambda a: a.get("final_llm_judge_score", -1e10), reverse=True)
    best = all_sorted[:k_best]
    k_worst_actual = min(k_worst, len(all_sorted) - k_best)
    worst = list(reversed(all_sorted))[:k_worst_actual]
    return {"best": best, "worst": worst}


# ============================================================
# Candidate generation
# ============================================================

def format_answers(answers: List[Dict[str, Any]]) -> str:
    """Format answers for inclusion in improvement prompts."""
    lines: List[str] = []
    for i, a in enumerate(answers):
        score = float(a.get("final_llm_judge_score", float("nan")))
        lines.append(f"### {i + 1}. Score: {score:.2f}")
        lines.append(a.get("answer", ""))
        lines.append("")
    return "\n".join(lines)


def get_answers_by_epochs(sample: Dict[str, Any], epochs: List[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    answers_by_epoch: Dict[int, List[Dict[str, Any]]] = sample.get("answers_by_epoch", {})
    for e in epochs:
        out.extend(answers_by_epoch.get(e, []))
    return out


async def generate_new_candidates(
    generation_engine,
    q: str,
    best: List[Dict[str, Any]],
    worst: List[Dict[str, Any]],
    gen_params: Dict[str, Any],
    n_candidates: int = 1,
) -> List[str]:
    """Generate improved candidate answers using the SAGE contrastive prompt."""
    recommendations_prompt = IMPROVEMENT_PROMPT.format(
        problem=q,
        best_solutions=format_answers(best),
        worst_solutions=format_answers(worst),
    )
    improvement_response = await generation_engine.agenerate(recommendations_prompt, **SINGLE_RESPONSE_PARAMS)
    if isinstance(improvement_response, list):
        improvement_response = improvement_response[0]
    recommendations_pattern = re.compile(r"<recommendations>(.*?)</recommendations>", re.DOTALL)
    recommendations_matches = recommendations_pattern.findall(str(improvement_response))
    recommendations = recommendations_matches[-1] if recommendations_matches else ""

    apply_prompt = APPLY_RECOMMENDATIONS_PROMPT.format(
        problem=q,
        best_solutions=format_answers(best),
        recommendations=recommendations,
    )
    gen_params = deepcopy(gen_params)
    gen_params["n"] = n_candidates
    refined_solutions_raw = await generation_engine.agenerate(apply_prompt, **gen_params)
    if isinstance(refined_solutions_raw, str):
        refined_solutions_raw = [refined_solutions_raw]

    refined_solution_pattern = re.compile(r"<refined_solution>(.*?)</refined_solution>", re.DOTALL)
    refined_solutions: List[str] = []
    for new_answer in refined_solutions_raw:
        matches = refined_solution_pattern.findall(str(new_answer))
        if matches:
            refined_solutions.append(matches[-1])

    iter_ = 1
    max_iters = 10
    while len(refined_solutions) < n_candidates and iter_ <= max_iters:
        gen_params2 = deepcopy(gen_params)
        gen_params2["seed"] = gen_params2.get("seed", 7) * (2 ** iter_) + 1
        refined_solutions2_raw = await generation_engine.agenerate(apply_prompt, **gen_params2)
        if isinstance(refined_solutions2_raw, str):
            refined_solutions2_raw = [refined_solutions2_raw]
        refined_solutions2: List[str] = []
        for new_answer in refined_solutions2_raw:
            matches = refined_solution_pattern.findall(str(new_answer))
            if matches:
                refined_solutions2.append(matches[-1])
        if iter_ == max_iters - 1 and len(refined_solutions2) + len(refined_solutions) < n_candidates:
            refined_solutions2 = [extract_text(x) for x in refined_solutions2_raw]
        refined_solutions += refined_solutions2
        iter_ += 1

    return refined_solutions[:n_candidates]


# ============================================================
# Optimization epoch
# ============================================================

async def run_optimization_epoch(
    generation_engine,
    judge_engine,
    sample: Dict[str, Any],
    epoch: int,
    gen_params: Optional[Dict[str, Any]] = None,
    gen_params_judge: Optional[Dict[str, Any]] = None,
    judge_prompt_templates: Optional[List[str]] = None,
    judge_configurations: Optional[List[Dict[str, Dict[str, Any]]]] = None,
    m_min: int = 1,
) -> List[Dict[str, Any]]:
    """Run one SAGE optimization epoch: form groups, generate improvements, score."""
    prev_epoch_answers = get_answers_by_epochs(sample, list(range(-1, epoch)))

    configurations: Dict[str, Dict[str, Any]] = {}
    for judge_configuration in judge_configurations:
        configurations.update(judge_configuration)

    verified_group = get_verified_group(prev_epoch_answers, configurations)
    best_and_worst_strict = form_best_and_worst_groups_strict(verified_group, m_min=m_min)
    best_and_worst_relaxed = form_best_and_worst_groups_relaxed(verified_group, configurations, m_min=m_min)

    has_strict_groups = len(best_and_worst_strict["best"]) > 0 and len(best_and_worst_strict["worst"]) > 0
    has_relaxed_groups = len(best_and_worst_relaxed["best"]) > 0 and len(best_and_worst_relaxed["worst"]) > 0

    if has_strict_groups:
        task_strict = generate_new_candidates(
            generation_engine,
            sample["prompt"],
            best_and_worst_strict["best"],
            best_and_worst_strict["worst"],
            gen_params=gen_params or DEFAULT_GEN_PARAMS,
            n_candidates=4,
        )
        task_relaxed = generate_new_candidates(
            generation_engine,
            sample["prompt"],
            best_and_worst_relaxed["best"],
            best_and_worst_relaxed["worst"],
            gen_params=gen_params or DEFAULT_GEN_PARAMS,
            n_candidates=3,
        )
        strict_new, relaxed_new = await asyncio.gather(task_strict, task_relaxed)
        new_candidates = strict_new + relaxed_new
    elif has_relaxed_groups:
        new_candidates = await generate_new_candidates(
            generation_engine,
            sample["prompt"],
            best_and_worst_relaxed["best"],
            best_and_worst_relaxed["worst"],
            gen_params=gen_params or DEFAULT_GEN_PARAMS,
            n_candidates=7,
        )
    else:
        print(f"[SAGE] Epoch {epoch}: insufficient candidates for group formation (m_min={m_min}). Skipping.")
        return []

    new_answers = [{"answer": cand, "epoch": epoch} for cand in new_candidates]

    for judge_prompt_template, judge_configuration in zip(judge_prompt_templates, judge_configurations):
        await score_answers_async(
            judge_engine=judge_engine,
            judge_prompt_template=judge_prompt_template,
            instruction=sample["prompt"],
            answers=new_answers,
            gen_params_judge=gen_params_judge or DEFAULT_JUDGE_PARAMS,
        )
    score_and_parse(new_answers, configurations, judge_prompt_templates)
    return sorted(new_answers, key=lambda a: a.get("final_llm_judge_score", -1e10), reverse=True)


# ============================================================
# Main entry point
# ============================================================

async def process_query(
    query: str,
    *,
    judge_prompt_templates: List[str],
    judge_configurations: List[Dict[str, Dict[str, Any]]],
    generation_params_list: Optional[List[Dict[str, Any]]] = None,
    judge_params: Optional[Dict[str, Any]] = None,
    num_optimization_epochs: int = 2,
    base_url: str = "http://localhost:9090/v1",
    model_name: str = "Qwen/Qwen3-8B-FP8",
    number_of_gens_per_epoch: int = 7,
    m_min: int = 1,
    aspects: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full SAGE pipeline: generate, score, refine, return best answer.

    Args:
        query: The problem prompt string.
        judge_prompt_templates: List of judge prompt template strings (must contain {prompt} and {answer}).
        judge_configurations: List of configuration dicts, one per template.
        generation_params_list: Override for initial generation parameters.
        judge_params: Override for judge generation parameters.
        num_optimization_epochs: Number of refinement epochs.
        base_url: vLLM server base URL.
        model_name: Model identifier string.
        number_of_gens_per_epoch: Number of candidate answers per epoch.
        m_min: Minimum group size for best/worst group formation.
        aspects: Optional list of aspect strings (for informational logging only; embed in prompts externally).

    Returns:
        Dict with keys: output (best answer text), all_answers, prompt.
    """
    generation_params_list = generation_params_list or [
        {**DEFAULT_GEN_PARAMS, "n": int(3 / 7 * number_of_gens_per_epoch), "temperature": 0.7},
        {**DEFAULT_GEN_PARAMS, "n": int(2 / 7 * number_of_gens_per_epoch), "temperature": 0.8},
        {**DEFAULT_GEN_PARAMS, "n": int(2 / 7 * number_of_gens_per_epoch), "temperature": 0.9},
    ]
    generation_params_list[0]["n"] = number_of_gens_per_epoch - sum(pl["n"] for pl in generation_params_list[1:])

    judge_params = judge_params or deepcopy(DEFAULT_JUDGE_PARAMS)
    gen_engine = get_engine(base_url=base_url, model=model_name, timeout=300, type="aug")
    judge_engine = get_engine(base_url=base_url, model=model_name, timeout=300, type="aug")

    print("[SAGE] Starting process_query")

    # 1. Generate initial candidates
    candidate_texts: List[str] = []
    for gen_params in generation_params_list:
        texts = await gen_engine.agenerate(query, **gen_params)
        if isinstance(texts, str):
            candidate_texts.append(texts)
        elif isinstance(texts, (list, tuple)):
            for t in texts:
                candidate_texts.append(extract_text(t))
        else:
            candidate_texts.append(extract_text(texts))

    answers: List[Dict[str, Any]] = [{"answer": t, "index": i} for i, t in enumerate(candidate_texts)]
    print(f"[SAGE] Generated {len(answers)} initial candidates")

    configurations: Dict[str, Dict[str, Any]] = {}
    for judge_configuration in judge_configurations:
        configurations.update(judge_configuration)

    # 2. Score initial candidates
    for judge_prompt_template, judge_configuration in zip(judge_prompt_templates, judge_configurations):
        await score_answers_async(
            judge_engine=judge_engine,
            judge_prompt_template=judge_prompt_template,
            instruction=query,
            answers=answers,
            gen_params_judge=judge_params,
        )
    score_and_parse(answers, configurations, judge_prompt_templates)
    print(f"[SAGE] Scored initial candidates")

    # 3. Optimization epochs
    sample: Dict[str, Any] = {
        "prompt": query,
        "answers_by_epoch": {-1: answers},
    }
    for epoch in range(num_optimization_epochs):
        print(f"[SAGE] Running epoch {epoch}")
        new_answers = await run_optimization_epoch(
            generation_engine=gen_engine,
            judge_engine=judge_engine,
            sample=sample,
            epoch=epoch,
            gen_params=DEFAULT_GEN_PARAMS,
            gen_params_judge=judge_params,
            judge_prompt_templates=judge_prompt_templates,
            judge_configurations=judge_configurations,
            m_min=m_min,
        )
        sample["answers_by_epoch"][epoch] = new_answers
        print(f"[SAGE] Finished epoch {epoch} ({len(new_answers)} new candidates)")

    # 4. Select best overall answer
    all_answers: List[Dict[str, Any]] = []
    for _, ans_list in sorted(sample["answers_by_epoch"].items(), key=lambda kv: kv[0]):
        all_answers.extend(ans_list)
    all_answers.sort(key=lambda x: x.get("final_llm_judge_score", -1e10), reverse=True)
    best = all_answers[0] if all_answers else {"answer": "", "final_llm_judge_score": float("-inf")}

    print("[SAGE] Finished process_query")
    return {
        "output": best["answer"],
        "all_answers": all_answers,
        "prompt": query,
    }
