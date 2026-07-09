# E1 — Closed-book contamination baseline (FinOpsBench-v2)

**Claim tested:** "High risk of data contamination for v2, as core questions
come from widely publicized FinQA training corpus" (Reviewer R3).

**Design.** Every v2 environment prompt (scenario + tool signatures + question)
is given to the model *without any callable tools*; the model must answer
directly and is asked for its best estimate. If FinQA memorization provided an
answer pathway, models could recall gold values without tool access. Keeping
tool signatures in the prompt makes the test conservative (strictly more
information than plain closed-book), so the measured accuracy is an **upper
bound** on what contamination can deliver. Scoring is the standard v2
comparator (`compare_outputs.compare_answers`).

Items: all v2 environments with a system prompt and gold answer on disk
(n=1,174; the paper's benchmark is the 1,108-item validated subset).
Note some scenarios legitimately contain the needed figure in the narrative
excerpt, so a non-zero closed-book score is expected even without any
contamination; this further inflates the upper bound.

**Run:**
```bash
export OPENROUTER_API_KEY=...
python run_closed_book.py --model openai/gpt-5-mini --reasoning_effort low
python run_closed_book.py --model qwen/qwen3-30b-a3b
python run_closed_book.py --model openai/gpt-4.1 --sample 300   # seed 13
python analyze.py
```

Results: `results/<model>.jsonl` (per item: prediction, passed, cost),
`results/summary.json` (accuracy vs the paper's agentic numbers, per-item
overlap with the released agentic gpt-4.1 run).
