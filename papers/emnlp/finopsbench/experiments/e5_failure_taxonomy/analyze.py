"""Aggregate the failure taxonomy: category distribution per model + process
metrics. Reads classified.jsonl and failures.jsonl; writes summary.json and
prints a markdown table.

Usage: python analyze.py
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CATS = ["wrong_tool_selection", "malformed_arguments", "incomplete_retrieval",
        "calculation_error", "financial_misunderstanding", "format_unit_error",
        "round_limit_exhaustion", "other"]
MODEL_ORDER = ["GPT-5", "o4-mini", "GPT-4.1", "GPT-4.1-mini", "Claude-Sonnet-4.5", "DeepSeek-V3"]


def main():
    classified = [json.loads(l) for l in (HERE / "classified.jsonl").open() if l.strip()]
    failures = [json.loads(l) for l in (HERE / "failures.jsonl").open() if l.strip()]
    fkey = {(r["version"], r["model"], r["query"]): r for r in failures}

    by_model = defaultdict(Counter)
    n_by_model = Counter()
    for c in classified:
        by_model[c["model"]][c["category"]] += 1
        n_by_model[c["model"]] += 1

    # process metrics per model (over the classified sample's failures)
    proc = defaultdict(lambda: defaultdict(list))
    for c in classified:
        r = fkey.get((c["version"], c["model"], c["query"]))
        if not r:
            continue
        proc[c["model"]]["n_tool_calls"].append(r.get("n_tool_calls", 0))
        if r["version"] == "v1":
            proc[c["model"]]["n_sql_errors"].append(r.get("n_sql_errors", 0))
        else:
            proc[c["model"]]["n_tool_errors"].append(r.get("n_tool_errors", 0))
        proc[c["model"]]["round_exhausted"].append(1 if r.get("round_exhausted") else 0)

    models = [m for m in MODEL_ORDER if m in n_by_model] + [m for m in n_by_model if m not in MODEL_ORDER]

    # ---- markdown: category share (%) per model ----
    print("\n### Failure category distribution (% of a model's classified failures)\n")
    header = "| Category | " + " | ".join(models) + " |"
    print(header)
    print("|" + "---|" * (len(models) + 1))
    for cat in CATS:
        row = [cat]
        for m in models:
            tot = n_by_model[m]
            row.append(f"{100*by_model[m][cat]/tot:.0f}%" if tot else "–")
        print("| " + " | ".join(row) + " |")
    print("| **n classified** | " + " | ".join(str(n_by_model[m]) for m in models) + " |")

    # ---- markdown: process metrics ----
    print("\n### Process metrics on failing traces (mean)\n")
    print("| Metric | " + " | ".join(models) + " |")
    print("|" + "---|" * (len(models) + 1))

    def meanrow(label, field, pct=False):
        cells = []
        for m in models:
            vals = proc[m].get(field, [])
            if vals:
                v = statistics.mean(vals)
                cells.append(f"{100*v:.0f}%" if pct else f"{v:.1f}")
            else:
                cells.append("–")
        print(f"| {label} | " + " | ".join(cells) + " |")

    meanrow("tool calls / trace", "n_tool_calls")
    meanrow("SQL errors / trace (v1)", "n_sql_errors")
    meanrow("tool errors / trace (v2)", "n_tool_errors")
    meanrow("round-exhausted share", "round_exhausted", pct=True)

    summary = {
        "n_classified": dict(n_by_model),
        "category_counts": {m: dict(by_model[m]) for m in models},
        "category_share_pct": {
            m: {cat: round(100 * by_model[m][cat] / n_by_model[m], 1) for cat in CATS if by_model[m][cat]}
            for m in models
        },
        "process_metrics_mean": {
            m: {f: round(statistics.mean(v), 2) for f, v in proc[m].items() if v}
            for m in models
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwritten summary.json")


if __name__ == "__main__":
    main()
