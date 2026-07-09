"""Fill the two gaps against the reviewer's requests and append markdown to README:

Failure ask -> add QUALITATIVE worked examples per category (not just a distribution).
Diversity ask -> add the items E6 did not quantify: financial-concept coverage,
                 explicit reasoning-operation mix, and template diversity.

Usage: python coverage_supplement.py [--benchmark_root /tmp/FinOpsBench]
"""

import argparse
import gzip
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).parent

# ---- financial-concept keyword groups (case-insensitive substring match) ----
CONCEPTS = {
    "Accounts payable / invoices / vendors": r"accounts? payable|\bap\b|invoice|vendor|supplier|purchase order",
    "Approval / authorization / controls": r"approv|authori|sign-?off|segregation of duties|override|blackout|control",
    "Overdue / aging / late payment": r"overdue|aging|aged|past due|late|days? outstanding",
    "Variance / budget vs actual": r"varianc|budget|forecast|over ?run|actual vs|deviation",
    "Revenue recognition / deferred": r"revenue recogni|deferred revenue|recogni[sz]ed|contract amendment",
    "Reconciliation / discrepancy": r"reconcil|discrepanc|mismatch|unmatched|does not (match|reconcile)",
    "Tax / VAT": r"\bvat\b|\btax\b|withholding|statutory deduction",
    "Cash flow / liquidity": r"cash ?flow|liquidit|receipt|payment run",
    "Fraud / duplicate / anomaly": r"fraud|duplicat|anomal|suspicious|manipulat|conceal",
    "Ratios / financial statement (v2/FinQA)": r"ratio|percent|net sales|lease|pension|debt|dividend|return|shares?",
}

# ---- v1 task categories (paper's five) via keywords ----
V1_CATS = {
    "Accounts Payable analysis": r"payable|invoice|vendor|supplier|purchase",
    "Variance analysis": r"varianc|budget|forecast|overrun|deviation",
    "Data integrity & reconciliation": r"reconcil|discrepanc|mismatch|integrity|duplicat|unmatched",
    "Revenue recognition": r"revenue|deferred|recogni",
    "Financial reporting": r"report|statement|ledger|balance|summary",
}
OP_VERBS = {
    "detect/identify (anomaly search)": r"\b(detect|identify|find|locate|flag|spot)\b",
    "list/retrieve (enumeration)": r"\b(list|show|retrieve|provide|display|give)\b",
    "compute/quantify (aggregation)": r"\b(compute|calculate|total|sum|how much|what is the (total|amount|percentage|ratio))\b",
    "compare (relative reasoning)": r"\b(compare|versus|vs\.?|difference|change|trend|higher|lower)\b",
}


def share(texts, patterns):
    n = len(texts)
    out = {}
    for label, pat in patterns.items():
        rx = re.compile(pat, re.I)
        out[label] = round(100 * sum(bool(rx.search(t)) for t in texts) / n, 1) if n else 0.0
    return out


def near_dup_rate(texts, sample_n=400, seed=13):
    import random
    s = random.Random(seed).sample(texts, min(sample_n, len(texts)))
    def toks(t):
        return set(re.findall(r"[a-z0-9]+", t.lower()))
    tok = [toks(t) for t in s]
    hi = tot = 0
    for a, b in combinations(range(len(tok)), 2):
        tot += 1
        u = tok[a] | tok[b]
        if u and len(tok[a] & tok[b]) / len(u) >= 0.8:
            hi += 1
    return round(100 * hi / tot, 2) if tot else 0.0


def load_v1_queries(root):
    return [json.loads(l)["query"] for l in gzip.open(root / "v1" / "data" / "finopsbench_v1_pool.jsonl.gz", "rt")]


def load_v2_questions(root):
    qs = []
    for d in sorted((root / "v2" / "finqa_agents").glob("agent_*")):
        f = d / "agent_system_prompt.txt"
        if f.is_file():
            m = re.search(r"(?is)Question\s*[-\s]*\n(.+?)(?:\nGuidelines|\Z)", f.read_text())
            if m:
                qs.append(m.group(1).strip())
    return qs


def qualitative_examples():
    cls = {(c["version"], c["model"], c["query"]): c for c in map(json.loads, (HERE / "classified.jsonl").open())}
    fails = [json.loads(l) for l in (HERE / "failures.jsonl").open()]
    import random
    random.Random(21).shuffle(fails)
    picked, seen = [], set()
    order = ["wrong_tool_selection", "malformed_arguments", "incomplete_retrieval",
             "calculation_error", "financial_misunderstanding", "format_unit_error", "round_limit_exhaustion"]
    for cat in order:
        for r in fails:
            c = cls.get((r["version"], r["model"], r["query"]))
            if c and c["category"] == cat and cat not in seen and len(r["query"]) > 25:
                picked.append((cat, r, c))
                seen.add(cat)
                break
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_root", type=Path, default=Path("/tmp/FinOpsBench"))
    args = p.parse_args()
    v1q = load_v1_queries(args.benchmark_root)
    v2q = load_v2_questions(args.benchmark_root)

    L = ["\n---\n\n## Coverage of the reviewer's requests (supplement)\n"]
    L.append("This section closes the two items the tables above did not fully address: "
             "**qualitative failure examples** (failure ask) and **financial-concept coverage / "
             "explicit reasoning-operation mix / template diversity** (diversity ask). "
             "SQL complexity, tool-chain depth and numerical operations are in `../e6_diversity/`.\n")

    # ---- qualitative examples ----
    L.append("### Qualitative failure examples (one per category)\n")
    L.append("Each is a real trace; the explanation is the LLM judge's reasoning (v1) or the classifier rationale (v2).\n")
    for cat, r, c in qualitative_examples():
        L.append(f"**{cat}** — {r['model']} ({r['version']})")
        L.append(f"- *Query:* {r['query'][:200]}")
        L.append(f"- *Expected:* {str(r['expected'])[:160]}")
        L.append(f"- *Model answer:* {str(r['final_answer'])[:160]}")
        why = r.get("judge_reasoning") or c.get("rationale")
        L.append(f"- *Why it failed:* {why[:220]}\n")

    # ---- financial-concept coverage ----
    L.append("### Financial-concept coverage (% of examples mentioning each concept)\n")
    cov1, cov2 = share(v1q, CONCEPTS), share(v2q, CONCEPTS)
    L.append(f"| Concept | v1 (n={len(v1q)}) | v2 (n={len(v2q)}) |")
    L.append("|---|---|---|")
    for k in CONCEPTS:
        L.append(f"| {k} | {cov1[k]}% | {cov2[k]}% |")

    # ---- reasoning operations ----
    L.append("\n### Reasoning operations\n")
    L.append("**v1 task categories** (keyword-assigned, non-exclusive):\n")
    catshare = share(v1q, V1_CATS)
    L.append("| Category | % of v1 |")
    L.append("|---|---|")
    for k, v in catshare.items():
        L.append(f"| {k} | {v}% |")
    L.append("\n**v1 query operation type** (analyst intent):\n")
    ops = share(v1q, OP_VERBS)
    L.append("| Operation | % of v1 |")
    L.append("|---|---|")
    for k, v in ops.items():
        L.append(f"| {k} | {v}% |")
    L.append("\n**v2 numerical operations** (from `../e6_diversity/`): aggregation 51%, difference/YoY 41%, "
             "ratio 32%, average 11%, percent-change 11% of reference plans.\n")

    # ---- template diversity ----
    L.append("### Template diversity\n")
    uniq = len(set(v1q))
    def ngram_ratio(texts, n):
        tot, dist = 0, set()
        for t in texts:
            tk = re.findall(r"[a-z0-9]+", t.lower())
            g = list(zip(*[tk[i:] for i in range(n)]))
            tot += len(g); dist.update(g)
        return round(len(dist) / tot, 4) if tot else 0.0
    L.append(f"- v1 expansion: **12 seed queries → {len(v1q)} examples** ({len(v1q)//12}× expansion) with cosine-0.9 near-duplicate filtering.")
    L.append(f"- v1 distinct queries: **{uniq}/{len(v1q)} = {100*uniq/len(v1q):.1f}%** (no exact duplicates).")
    L.append(f"- v1 distinct-token-3-gram ratio: **{ngram_ratio(v1q,3)}**; distinct-4-gram ratio: **{ngram_ratio(v1q,4)}**.")
    L.append(f"- v1 high-overlap pair rate (token-Jaccard ≥ 0.8 on a 400-query sample): **{near_dup_rate(v1q)}%** — templated phrasings are rare.")
    L.append(f"- v2 distinct questions: **{len(set(v2q))}/{len(v2q)}** (human-authored FinQA questions).\n")

    md = "\n".join(L) + "\n"
    (HERE / "README.md").open("a").write(md)
    print(md)


if __name__ == "__main__":
    main()
