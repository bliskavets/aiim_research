# E2 — LLM judge vs deterministic numeric scoring (FinOpsBench-v1)

**Claims tested:** "FinOpsBench-v1 evaluation itself relies on another LLM
judge rather than deterministic correctness whenever possible" (Reviewer PVoW);
"v1 lacks machine-verifiable hard ground truth" (Reviewer R3).

**Design.** The released v1 pool (8,233 items) pairs each query with an
expected answer and a reference agent trace. The trace's final message is
scored two ways: (a) deterministically — the answer must contain a number
matching the expected answer's single numeric value within the v2 tolerance
rule; (b) by the LLM judge, using the *verbatim* grading prompt from the v1
evaluation harness (judge: o4-mini, as in the paper).

**Key finding on applicability:** only 363/8,233 (4.4%) of expected answers
contain exactly one number; the rest are multi-entity analyst answers (lists
of invoice IDs, per-vendor tables, month ranges, policy descriptions) for
which token-level numeric matching is undefined. This is the design reason
v1 uses an LLM comparator while v2 (plain-number answers by construction)
is scored deterministically.

Even within the scalar subset the single number is often incidental to the
answer (a variance/payment ID rather than the asked-for value), so the two
scorers disagree on 93/362 items. `results/disagreements_for_human_annotation.jsonl`
contains all disagreement cases with a `human_label` field — filling it
(is the trace's answer actually correct?) turns this into a targeted human
check of judge accuracy on exactly the cases where scoring is contested.

**Run:**
```bash
export OPENROUTER_API_KEY=...
python run_agreement.py --judge_model openai/o4-mini --sample 363
```

Results: `results/agreement_scalar_openai_o4-mini.jsonl`, `results/summary.json`,
`results/pilot_lastnumber_rule_openai_o4-mini.jsonl` (discarded first design:
last-number rule, invalid for verbose answers — kept for the record).
