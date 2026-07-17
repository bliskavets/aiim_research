#!/usr/bin/env python3
"""Pairwise AlpacaEval-style judging: model output vs reference, GPT-4 via OpenRouter.
Randomizes A/B position per item (deterministic by index) to cancel position bias.
Reports win rate, avg length (verbosity check), and a length-matched win rate."""
from __future__ import annotations
import argparse, json, os, re, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

JUDGE_MODEL = os.getenv("ALPACA_JUDGE_MODEL", "openai/gpt-4.1")
_client = None
def client():
    global _client
    if _client is None: _client = OpenAI()
    return _client

PROMPT = """You are comparing two AI assistant responses to an instruction. Pick the response that is more helpful, accurate, and appropriate. If they are equal, still pick the marginally better one.

Instruction:
{instruction}

Response A:
{A}

Response B:
{B}

Answer with ONLY a JSON object: {{"better": "A"}} or {{"better": "B"}}."""

def judge(item, i):
    model_is_A = (i % 2 == 0)
    A = item["output"] if model_is_A else item["reference"]
    B = item["reference"] if model_is_A else item["output"]
    msg = PROMPT.format(instruction=item["instruction"][:4000], A=A[:6000], B=B[:6000])
    try:
        r = client().chat.completions.create(model=JUDGE_MODEL,
            messages=[{"role": "user", "content": msg}],
            response_format={"type": "json_object"}, max_tokens=200, temperature=0)
        pick = json.loads(r.choices[0].message.content or "{}").get("better", "")
        model_won = (pick == "A") == model_is_A
        return 1.0 if model_won else 0.0
    except Exception as e:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True); ap.add_argument("--label", default="model")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    items = json.load(open(a.outputs))
    wins = [None]*len(items)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge, it, i): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            wins[futs[f]] = f.result()
    valid = [(w, items[i]) for i, w in enumerate(wins) if w is not None]
    wr = 100 * sum(w for w, _ in valid) / len(valid)
    out_len = statistics.median([it["out_chars"] for _, it in valid])
    ref_len = statistics.median([it["ref_chars"] for _, it in valid])
    # length-matched win rate: only items where model output is not much longer than reference
    matched = [(w, it) for w, it in valid if it["out_chars"] <= 1.5 * it["ref_chars"]]
    wr_matched = 100 * sum(w for w, _ in matched) / len(matched) if matched else float("nan")
    res = {"label": a.label, "n": len(valid), "win_rate_vs_davinci003": round(wr, 1),
           "median_out_chars": out_len, "median_ref_chars": ref_len,
           "len_ratio": round(out_len/ref_len, 2),
           "win_rate_length_matched": round(wr_matched, 1), "n_length_matched": len(matched)}
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
