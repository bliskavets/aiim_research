"""Exp 2 — difficulty-axis resolution (no new model runs).

Buckets already-collected v2 agentic accuracy by controllable difficulty knobs and
shows accuracy declines monotonically — evidence that FinOpsBench is a tunable
instrument, not a single fixed score. Difficulty features per environment:
  - n_distractor_tools = augmented tools minus tools used by the reference plan
  - tool_chain_depth   = number of tool calls in the reference plan
  - n_augmented_tools  = total tools exposed

Accuracy comes from committed run files (e8 200-subset gpt-4.1/-mini; e4 DeepSeek/Claude).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = Path("/tmp/FinOpsBench/v2/finqa_agents")
E8 = HERE.parent / "e8_access_ladder" / "results" / "agentic"
E4 = HERE.parent / "e4_new_models" / "results"

RESULT_FILES = {
    "GPT-4.1": E8 / "openai_gpt-4.1.json",
    "GPT-4.1-mini": E8 / "openai_gpt-4.1-mini.json",
    "DeepSeek-V3": E4 / "deepseek_deepseek-chat-v3-0324.json",
    "Claude-Sonnet-4.5": E4 / "anthropic_claude-sonnet-4.5.json",
}


def features(aid):
    d = ROOT / aid
    ta, pa = d / "tools_augmented.py", d / "correct_plan_augmented.py"
    if not (ta.is_file() and pa.is_file()):
        return None
    aug = set(re.findall(r"^def ([a-z_][a-z0-9_]*)", ta.read_text(), re.M))
    plan = pa.read_text()
    core = {t for t in aug if re.search(r"\b" + re.escape(t) + r"\s*\(", plan)}
    depth = sum(len(re.findall(r"\b" + re.escape(t) + r"\s*\(", plan)) for t in core)
    return {"n_aug": len(aug), "n_distractor": len(aug) - len(core), "depth": depth}


def bucket_report(rows, key, edges, labels):
    """rows: list of (feature_value, passed). Returns per-bucket (label, n, acc)."""
    buckets = defaultdict(list)
    for v, ok in rows:
        for i, hi in enumerate(edges):
            if v <= hi:
                buckets[labels[i]].append(ok)
                break
        else:
            buckets[labels[-1]].append(ok)
    return [(lab, len(buckets[lab]), round(100 * sum(buckets[lab]) / len(buckets[lab]), 1))
            for lab in labels if buckets[lab]]


def main():
    feat_cache = {}
    pooled = {"n_distractor": [], "depth": []}
    per_model = {}
    for model, path in RESULT_FILES.items():
        if not path.exists():
            continue
        rows = json.load(path.open())
        d_rows, depth_rows = [], []
        for r in rows:
            aid = r["agent_id"]
            f = feat_cache.get(aid) or features(aid)
            feat_cache[aid] = f
            if not f:
                continue
            ok = bool(r["passed"])
            d_rows.append((f["n_distractor"], ok)); depth_rows.append((f["depth"], ok))
            pooled["n_distractor"].append((f["n_distractor"], ok)); pooled["depth"].append((f["depth"], ok))
        per_model[model] = {"n": len(d_rows),
                            "by_distractor": bucket_report(d_rows, "d", [2, 4, 6, 99], ["0-2", "3-4", "5-6", "7+"]),
                            "by_depth": bucket_report(depth_rows, "x", [3, 5, 7, 99], ["1-3", "4-5", "6-7", "8+"])}

    print("### Accuracy vs number of DISTRACTOR tools (per model)\n")
    print("| Model | 0-2 | 3-4 | 5-6 | 7+ |")
    print("|---|---|---|---|---|")
    for m, d in per_model.items():
        cells = {lab: f"{acc}% (n={n})" for lab, n, acc in d["by_distractor"]}
        print(f"| {m} | " + " | ".join(cells.get(k, "–") for k in ["0-2", "3-4", "5-6", "7+"]) + " |")

    print("\n### Accuracy vs tool-chain depth (reference-plan calls)\n")
    print("| Model | 1-3 | 4-5 | 6-7 | 8+ |")
    print("|---|---|---|---|---|")
    for m, d in per_model.items():
        cells = {lab: f"{acc}% (n={n})" for lab, n, acc in d["by_depth"]}
        print(f"| {m} | " + " | ".join(cells.get(k, "–") for k in ["1-3", "4-5", "6-7", "8+"]) + " |")

    print("\n### Pooled (all models)\n")
    print("- by distractors:", bucket_report(pooled["n_distractor"], "d", [2, 4, 6, 99], ["0-2", "3-4", "5-6", "7+"]))
    print("- by depth:", bucket_report(pooled["depth"], "x", [3, 5, 7, 99], ["1-3", "4-5", "6-7", "8+"]))

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "difficulty_axes.json").write_text(json.dumps({
        "per_model": per_model,
        "pooled_by_distractor": bucket_report(pooled["n_distractor"], "d", [2, 4, 6, 99], ["0-2", "3-4", "5-6", "7+"]),
        "pooled_by_depth": bucket_report(pooled["depth"], "x", [3, 5, 7, 99], ["1-3", "4-5", "6-7", "8+"]),
    }, indent=2))


if __name__ == "__main__":
    main()
