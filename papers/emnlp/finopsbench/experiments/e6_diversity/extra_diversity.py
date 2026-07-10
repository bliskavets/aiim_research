"""E6 supplement: two extra diversity axes requested during rebuttal.

1. v1 SQL structural depth  -- join count, subquery nesting, clauses per query,
   tables joined. Deeper than the surface keyword-share table.
2. v2 entity/source diversity -- distinct FinQA source companies and filings the
   environments are drawn from, plus the off-path (distractor + partial-info)
   tool-count distribution per environment.

Pure offline scripting over released data + the FinQA train split. No API calls.

Usage: python extra_diversity.py [--benchmark_root /tmp/FinOpsBench] \
                                 [--finqa /tmp/finqa_train.json]
"""
import argparse, gzip, json, re, statistics
from collections import Counter
from pathlib import Path

CLAUSES = ["SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "JOIN", "LIMIT"]


def bucket(counts, edges, labels):
    out = Counter()
    for c in counts:
        for e, lab in zip(edges, labels):
            if c <= e:
                out[lab] += 1
                break
        else:
            out[labels[-1]] += 1
    n = len(counts) or 1
    return {lab: round(100 * out.get(lab, 0) / n, 1) for lab in labels}


def v1_sql_depth(root):
    pool = root / "v1" / "data" / "finopsbench_v1_pool.jsonl.gz"
    joins_per_q, nest_per_q, clauses_per_q, tables_per_q = [], [], [], []
    item_max_joins = []
    n_items = n_queries = 0
    for line in gzip.open(pool, "rt"):
        item = json.loads(line)
        n_items += 1
        joins_this_item = [0]
        for msg in item.get("agent_dialog") or []:
            for tc in msg.get("tool_calls") or []:
                q = (tc.get("tool_kwargs") or {}).get("query")
                if not q or "select" not in q.lower():
                    continue
                n_queries += 1
                j = len(re.findall(r"\bJOIN\b", q, re.I))
                nest = len(re.findall(r"\(\s*SELECT\b", q, re.I))
                cl = sum(bool(re.search(r"\b" + c.replace(" ", r"\s+") + r"\b", q, re.I)) for c in CLAUSES)
                # tables: FROM/JOIN referenced identifiers (rough)
                t = 1 + j
                joins_per_q.append(j); nest_per_q.append(nest)
                clauses_per_q.append(cl); tables_per_q.append(t)
                joins_this_item.append(j)
        item_max_joins.append(max(joins_this_item))
    return {
        "n_items": n_items,
        "n_reference_queries": n_queries,
        "joins_per_query": bucket(joins_per_q, [0, 1, 2], ["0", "1", "2", "3+"]),
        "subquery_nesting_per_query": bucket(nest_per_q, [0, 1], ["0", "1", "2+"]),
        "tables_joined_per_query": bucket(tables_per_q, [1, 2, 3], ["1", "2", "3", "4+"]),
        "clauses_per_query_mean": round(statistics.mean(clauses_per_q), 2) if clauses_per_q else 0,
        "clauses_per_query_max": max(clauses_per_q) if clauses_per_q else 0,
        "multi_join_item_share_pct": round(100 * sum(m >= 2 for m in item_max_joins) / (n_items or 1), 1),
    }


def v2_entity(root, finqa_path):
    finqa = json.load(open(finqa_path))
    dirs = sorted((root / "v2" / "finqa_agents").glob("agent_*"),
                  key=lambda d: int(d.name.split("_")[1]))
    companies, filings = Counter(), set()
    offpath_counts = []
    n = 0
    for d in dirs:
        idx = int(d.name.split("_")[1])
        if idx >= len(finqa):
            continue
        plan_f = d / "correct_plan_augmented.py"
        tools_f = d / "tools_augmented.py"
        if not plan_f.is_file() or not tools_f.is_file():
            continue
        n += 1
        fn = finqa[idx].get("filename", "")
        comp = fn.split("/")[0] if fn else "?"
        companies[comp] += 1
        filings.add(fn)
        tool_names = set(re.findall(r"^def ([a-z_][a-z0-9_]*)", tools_f.read_text(), re.M))
        tool_names = {t for t in tool_names if not t.startswith("_")}
        used = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", plan_f.read_text()))
        offpath = len(tool_names - used)
        offpath_counts.append(offpath)
    return {
        "n_environments": n,
        "distinct_source_companies": len(companies),
        "distinct_source_filings": len(filings),
        "top_companies": companies.most_common(8),
        "offpath_tools_per_env": {
            "mean": round(statistics.mean(offpath_counts), 2) if offpath_counts else 0,
            "median": statistics.median(offpath_counts) if offpath_counts else 0,
            "max": max(offpath_counts) if offpath_counts else 0,
            "share_with_ge2": round(100 * sum(c >= 2 for c in offpath_counts) / (len(offpath_counts) or 1), 1),
        },
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_root", default="/tmp/FinOpsBench")
    ap.add_argument("--finqa", default="/tmp/finqa_train.json")
    a = ap.parse_args()
    root = Path(a.benchmark_root)
    res = {"v1_sql_depth": v1_sql_depth(root), "v2_entity": v2_entity(root, a.finqa)}
    out = Path(__file__).parent / "results" / "extra_diversity.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
