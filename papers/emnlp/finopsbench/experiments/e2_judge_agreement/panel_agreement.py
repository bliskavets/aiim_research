"""Judge-corroboration: how often do the three construction judges agree?

Addresses Reviewer 6zfv's "quality depends on LLM judgments": we show acceptance
is not one model's opinion. Over the released v1 pool, each item carries three
independent panel verdicts (claude-sonnet-4-0, o4-mini, o3-mini) on five binary
criteria. We report per-criterion unanimity across the panel. High unanimity on
objective criteria plus lower unanimity on the subjective one (answer soundness)
is exactly why a majority panel is used rather than a single judge.

Offline over released data, no API calls.

Usage: python panel_agreement.py [--benchmark_root /tmp/FinOpsBench]
"""
import argparse, gzip, json
from pathlib import Path

CRIT = ["data_is_natural", "trace_is_reasonable", "trace_is_sound",
        "reasoning_is_grounded", "answer_is_sound"]


def main(root):
    pool = Path(root) / "v1" / "data" / "finopsbench_v1_pool.jsonl.gz"
    n = 0
    unanimous_all = 0
    per_crit_unanimous = {c: 0 for c in CRIT}
    for line in gzip.open(pool, "rt"):
        d = json.loads(line)
        js = d.get("judgements") or []
        if len(js) < 2:
            continue
        n += 1
        all_u = True
        for c in CRIT:
            vals = [j.get(c) for j in js if c in j]
            if not vals:
                continue
            if len(set(vals)) == 1:
                per_crit_unanimous[c] += 1
            else:
                all_u = False
        if all_u:
            unanimous_all += 1
    res = {
        "n_items": n,
        "judges": ["claude-sonnet-4-0", "o4-mini", "o3-mini"],
        "unanimous_on_all_criteria_pct": round(100 * unanimous_all / n, 1),
        "per_criterion_unanimity_pct": {c: round(100 * per_crit_unanimous[c] / n, 1) for c in CRIT},
    }
    out = Path(__file__).parent / "results" / "panel_agreement.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_root", default="/tmp/FinOpsBench")
    main(ap.parse_args().benchmark_root)
