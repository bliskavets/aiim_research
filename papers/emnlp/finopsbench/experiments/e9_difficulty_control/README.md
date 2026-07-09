# E9 — Difficulty control (Reviewer R2 / paper's "tunable difficulty" claim)

The paper claims FinOpsBench can regulate task difficulty. We verify this on v2.

## What the code lets you dial (from `v2/pipeline/prompts.py`)
The generator builds a **basic** environment then an **augmented** one whose explicit
goal is "a more complicated environment that is harder to solve": it splits tables to
lengthen the solution path, adds tools "with similar names but different arguments",
and adds tools that "provide information not directly relevant ... to fall into a trap".
Measured over 200 environments: **basic 3.9 tools → augmented 8.9 tools** (mean 4.0 core
used by the reference plan + 4.9 extra), reference-plan **tool-chain depth mean 4.8**.

## Exp 2 — accuracy scales with tool-chain depth (clean, observational, no new runs)
Bucketing already-collected agentic accuracy (e8 + e4 runs) by required reference-plan
depth shows a monotonic decline — harder items are measurably harder:

| Model | depth 1-3 | 4-5 | 6-7 | 8+ |
|---|---|---|---|---|
| DeepSeek-V3 | 61.9% | 58.6% | 54.6% | **43.4%** |
| Claude-Sonnet-4.5 | 72.4% | 69.2% | 76.3% | **42.9%** |
| GPT-4.1 | 59.5% | 65.3% | 58.8% | 58.3% |
| Pooled | 61.6% | 61.1% | 57.0% | **45.8%** |

(By contrast, bucketing by *raw distractor-tool count* is **confounded** — distractor
count co-varies with task type — and shows no clean trend; see `difficulty_axes.json`.
This is exactly why a controlled manipulation is needed, which the access ladder provides.)

## Exp 3 — distractor ablation: attempted, reported as INVALID (honest negative)
We tried removing the augmentation's distractor tools (expose only reference-plan tools)
and re-running gpt-4.1-mini: 52.0% vs 61.5% with all tools. This is **not** a valid
distractor-tax measurement, for two reasons, so we do not report a tax number:
1. the per-example system prompt still advertises the removed tools, so the agent calls
   an absent tool and errors (prompt/tool mismatch artifact);
2. "not used by the optimal reference plan" ≠ "distractor": some plan-external tools are
   discovery helpers a real agent needs (e.g. `list_all_company_names` when the plan
   hardcodes the entity name). A clean ablation would require regenerating the system
   prompt per condition. See `ablation_note.json`.

## Clean difficulty-control evidence for the rebuttal
- **Access ladder (e8):** a controlled manipulation of the same items — accuracy moves
  1.5% → 62% → 65% across information-access modes.
- **Tool-chain depth scaling (above):** ~16-point drop from shallow to deep chains.
- **Designed basic→augmented knob:** +5 tools, split tables, longer paths, by construction.
