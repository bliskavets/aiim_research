"""Assemble the information-access ladder and derived metrics.

Rungs (same 200 v2 items, same model, percent-robust scoring):
  (a) question_only  -> parametric / memorised floor
  (c) agentic        -> tools only (our benchmark)
  (b) full_context   -> FinQA narrative+table in prompt, no tools (reading ceiling)

Derived:
  tool_use_necessity = c - a   (how much answering requires tool use)
  agentic_gap        = b - c   (accuracy lost moving from reading to tool retrieval)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/FinOpsBench/v2")
from compare_outputs import compare_answers, extract_number_from_answer  # noqa: E402

HERE = Path(__file__).parent


def robust(pred, gold):
    if compare_answers(pred, gold):
        return True
    pv, _ = extract_number_from_answer(pred)
    gv, gp = extract_number_from_answer(gold)
    if pv == "no answer" or gv == "no answer":
        return False
    tol = 10 ** (-gp) * 0.6
    return abs(pv / 100 - gv) <= tol or abs(pv - gv * 100) <= max(tol * 100, 0.6)


def acc_jsonl(path):
    rows = [json.loads(l) for l in path.open()]
    return round(100 * sum(robust(r["prediction"], r["gold"]) for r in rows) / len(rows), 1), len(rows)


def acc_agentic(path):
    rows = json.load(path.open())
    return round(100 * sum(robust(r["agent_output"], r["ground_truth_answer"]) for r in rows) / len(rows), 1), len(rows)


MODELS = ["openai_gpt-4.1-mini", "openai_gpt-4.1", "deepseek_deepseek-v3.2", "anthropic_claude-haiku-4.5", 'qwen_qwen3-235b-a22b-2507', 'meta-llama_llama-3.3-70b-instruct', 'openai_gpt-oss-120b', 'deepseek_deepseek-v4-flash']
NAMES = {"openai_gpt-4.1-mini": "GPT-4.1-mini", "openai_gpt-4.1": "GPT-4.1", "deepseek_deepseek-v3.2": "DeepSeek-V3.2", "anthropic_claude-haiku-4.5": "Claude-Haiku-4.5", 'qwen_qwen3-235b-a22b-2507': 'Qwen3-235B-A22B', 'meta-llama_llama-3.3-70b-instruct': 'Llama-3.3-70B', 'openai_gpt-oss-120b': 'gpt-oss-120b', 'deepseek_deepseek-v4-flash': 'DeepSeek-V4-Flash'}

ladder = {}
for m in MODELS:
    a, _ = acc_jsonl(HERE / "results" / f"question_only_{m}.jsonl")
    b, _ = acc_jsonl(HERE / "results" / f"full_context_{m}.jsonl")
    fq, _ = acc_jsonl(HERE / "results" / f"finqa_canonical_{m}.jsonl")
    c, n = acc_agentic(HERE / "results" / "agentic" / f"{m}.json")
    ladder[NAMES[m]] = {"n": n, "a_question_only": a, "c_agentic": c,
                        "d_finqa_canonical": fq, "b_full_context": b,
                        "tool_use_necessity_c_minus_a": round(c - a, 1),
                        "agentic_gap_vs_finqa": round(fq - c, 1),
                        "agentic_gap_vs_fullctx": round(b - c, 1)}

(HERE / "results" / "ladder_summary.json").write_text(json.dumps(ladder, indent=2))

print("### Information-access ladder (200 v2 items, percent-robust scoring)\n")
print("| Model | (a) question-only | (c) agentic (tools) | (d) FinQA-native (gold facts) | (b) full-context (whole doc) | tool-use necessity (c−a) | agentic gap (d−c) |")
print("|---|---|---|---|---|---|---|")
for name, d in ladder.items():
    print(f"| {name} | {d['a_question_only']}% | {d['c_agentic']}% | {d['d_finqa_canonical']}% | {d['b_full_context']}% | "
          f"+{d['tool_use_necessity_c_minus_a']} pt | {d['agentic_gap_vs_finqa']} pt |")
print(json.dumps(ladder, indent=2))
