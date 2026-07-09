"""E6: quantitative diversity statistics for both FinOpsBench versions.

Covers the axes requested in review: SQL complexity, tool-chain depth,
numerical operations, and template/lexical diversity. Pure offline scripting
over the released benchmark data; no API calls.

Usage: python compute_diversity.py [--benchmark_root /tmp/FinOpsBench]
"""

import argparse
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path

SQL_FEATURES = {
    "JOIN": re.compile(r"\bJOIN\b", re.I),
    "GROUP BY": re.compile(r"\bGROUP\s+BY\b", re.I),
    "ORDER BY": re.compile(r"\bORDER\s+BY\b", re.I),
    "HAVING": re.compile(r"\bHAVING\b", re.I),
    "subquery": re.compile(r"\(\s*SELECT\b", re.I),
    "aggregate": re.compile(r"\b(SUM|COUNT|AVG|MIN|MAX)\s*\(", re.I),
    "CASE": re.compile(r"\bCASE\b", re.I),
    "date function": re.compile(r"\b(DATE|JULIANDAY|STRFTIME)\s*\(", re.I),
}

V2_OPERATIONS = {
    "percent change": re.compile(r"percent[_ ]?change|pct[_ ]?change", re.I),
    "ratio / division": re.compile(r"ratio|divide|/\s*[a-z_]+\)", re.I),
    "sum / aggregation": re.compile(r"\bsum\s*\(|total", re.I),
    "difference": re.compile(r"\b(diff|subtract|change|delta)\b|-\s*[a-z_]+\)", re.I),
    "average": re.compile(r"\b(avg|average|mean)\b", re.I),
}


def distinct_ngram_ratio(texts: list[str], n: int) -> float:
    total, distinct = 0, set()
    for t in texts:
        tokens = re.findall(r"[a-z0-9]+", t.lower())
        grams = list(zip(*[tokens[i:] for i in range(n)]))
        total += len(grams)
        distinct.update(grams)
    return len(distinct) / total if total else 0.0


def describe(values: list) -> dict:
    return {
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "p90": sorted(values)[int(0.9 * len(values))],
        "max": max(values),
    }


def analyze_v1(root: Path) -> dict:
    queries, sql_calls_per_item, sql_feature_counts = [], [], Counter()
    tool_calls_per_item, roles = [], Counter()
    pool = root / "v1" / "data" / "finopsbench_v1_pool.jsonl.gz"
    n = 0
    for line in gzip.open(pool, "rt"):
        item = json.loads(line)
        n += 1
        queries.append(item["query"])
        roles[item.get("user_role", "?")] += 1
        calls = []
        for msg in item.get("agent_dialog") or []:
            for tc in msg.get("tool_calls") or []:
                kwargs = tc.get("tool_kwargs") or {}
                calls.append(kwargs.get("query") or json.dumps(kwargs))
        tool_calls_per_item.append(len(calls))
        sql_calls_per_item.append(len(calls))
        item_features = set()
        for c in calls:
            for name, rx in SQL_FEATURES.items():
                if rx.search(c):
                    item_features.add(name)
        sql_feature_counts.update(item_features)

    return {
        "n_examples": n,
        "distinct_user_roles": len(roles),
        "tool_calls_per_example": describe(tool_calls_per_item),
        "sql_feature_share_of_examples": {
            k: round(v / n, 3) for k, v in sql_feature_counts.most_common()
        },
        "query_lexical_diversity": {
            "distinct_1gram_ratio": round(distinct_ngram_ratio(queries, 1), 4),
            "distinct_3gram_ratio": round(distinct_ngram_ratio(queries, 3), 4),
            "distinct_queries_ratio": round(len(set(queries)) / len(queries), 4),
        },
    }


def analyze_v2(root: Path) -> dict:
    plan_calls, n_tools, op_counts = [], [], Counter()
    dirs = sorted((root / "v2" / "finqa_agents").glob("agent_*"))
    n = 0
    for d in dirs:
        plan_f = d / "correct_plan_augmented.py"
        tools_f = d / "tools_augmented.py"
        if not plan_f.is_file() or not tools_f.is_file():
            continue
        n += 1
        plan = plan_f.read_text()
        # count only invocations of functions actually defined in the tool set
        tool_names = set(re.findall(r"^def ([a-z_][a-z0-9_]*)", tools_f.read_text(), re.M))
        calls = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", plan)
        plan_calls.append(sum(c in tool_names for c in calls))
        n_tools.append(len(tool_names))
        for name, rx in V2_OPERATIONS.items():
            if rx.search(plan):
                op_counts[name] += 1

    return {
        "n_examples": n,
        "reference_plan_calls_per_example": describe(plan_calls),
        "tools_per_example": describe(n_tools),
        "operation_share_of_examples": {
            k: round(v / n, 3) for k, v in op_counts.most_common()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    args = p.parse_args()
    out = {
        "v1": analyze_v1(args.benchmark_root),
        "v2": analyze_v2(args.benchmark_root),
    }
    print(json.dumps(out, indent=2))
    out_path = Path(__file__).parent / "results" / "diversity_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
