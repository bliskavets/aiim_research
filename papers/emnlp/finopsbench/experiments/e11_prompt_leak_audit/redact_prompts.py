"""Audit + redact answer leakage in FinOpsBench-v2 system prompts.

Bug found during rebuttal: the Stage-8 system-prompt generator sometimes used the
*actual gold answer* as the output-format example, e.g. `... output the final
percentage value only (e.g. "39.1%")` where 39.1% IS the gold. The agent is thus
told the answer in the format hint. This inflates agentic accuracy.

This script scans every agent_system_prompt.txt, and where the gold answer appears
inside a format-hint context (e.g./for example/…), replaces the leaked value with a
format-preserving neutral placeholder. Originals are backed up to
`agent_system_prompt.txt.orig`. A report is written to results/leak_report.json.

Usage:
    python redact_prompts.py --apply     # redact in place (with .orig backups)
    python redact_prompts.py             # dry-run audit only
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path("/tmp/FinOpsBench/v2/finqa_agents")
CUE = r"(e\.g\.|for example|for instance|example|final (?:answer|number|percentage|value)|respond with|output (?:only|the)|answer as|format(?:ted)? as|should be)"


def neutral(gold: str) -> str:
    g = gold.strip()
    if g.lower() in ("yes", "no"):
        return 'yes" or "no'
    if g.endswith("%"):
        return "45.6%" if g.replace("%", "").lstrip("-") == "12.3" else "12.3%"
    try:
        v = float(g.replace(",", ""))
        return "123" if v == int(v) else "12.3"
    except ValueError:
        return "XX"


def redact(sp: str, gold: str) -> tuple[str, int]:
    """Replace gold inside format-hint contexts with a neutral placeholder."""
    if len(gold) < 2:
        return sp, 0
    n = neutral(gold)
    count = [0]
    g = re.escape(gold)
    # (1) cue ... gold  (small same-line gap), keeping surrounding quotes/parens
    pat1 = re.compile(CUE + r'(?P<mid>[^\n]{0,30}?)' + g + r'(?P<post>["\)\.]*)', re.I)
    def sub1(m):
        count[0] += 1
        return f'{m.group(1)}{m.group("mid")}{n}{m.group("post")}'
    sp = pat1.sub(sub1, sp)
    # (2) parenthetical that both contains a cue AND the gold: (e.g. "gold")
    pat2 = re.compile(r'\(([^\n\)]*?(?:e\.g\.|example|i\.e\.)[^\n\)]*?)' + g + r'([^\n\)]*)\)', re.I)
    def sub2(m):
        count[0] += 1
        return f'({m.group(1)}{n}{m.group(2)})'
    sp = pat2.sub(sub2, sp)
    return sp, count[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    dirs = [d for d in sorted(ROOT.glob("agent_*"))
            if (d / "agent_system_prompt.txt").is_file() and (d / "initial_solution.txt").is_file()]
    changed, still_leaks, report = 0, [], []
    for d in dirs:
        gold = (d / "initial_solution.txt").read_text().strip()
        f = d / "agent_system_prompt.txt"
        sp = f.read_text()
        new, k = redact(sp, gold)
        if k:
            report.append({"agent_id": d.name, "gold": gold, "redactions": k})
            if args.apply:
                bak = d / "agent_system_prompt.txt.orig"
                if not bak.exists():
                    bak.write_text(sp)
                f.write_text(new)
            changed += 1
            # residual: gold still present after redaction (narrative leak)
            if len(gold) >= 3 and gold in new:
                still_leaks.append(d.name)

    out = {"scanned": len(dirs), "prompts_redacted": changed,
           "residual_gold_present_after_redaction": len(still_leaks),
           "residual_examples": still_leaks[:20], "detail": report}
    (Path(__file__).parent / "results" / "leak_report.json").write_text(json.dumps(out, indent=2))
    print(f"scanned={len(dirs)} | prompts with format-hint leak {'redacted' if args.apply else '(dry-run)'}={changed}")
    print(f"residual gold-in-narrative after redaction: {len(still_leaks)} ({still_leaks[:8]})")


if __name__ == "__main__":
    main()
