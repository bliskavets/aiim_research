# E11 — Answer-leak audit & fix in FinOpsBench-v2 system prompts

**Found during rebuttal (Reviewer-driven check):** the Stage-8 system-prompt
generator sometimes used the *actual gold answer* as the output-format example,
e.g. `output the final percentage value only (e.g. "39.1%")` where 39.1% is the
gold. The agent is thus handed the answer in the format hint, inflating agentic
accuracy.

## Scale
- Exact gold answer present in the system prompt: **345/1174 (29.4%)** before fix.
- Classified: ~300 are format-hint leaks (`e.g./for example/…`), ~45 narrative
  or coincidental (mostly `yes`/`no` word matches).

## Fix (`redact_prompts.py --apply`)
Replaces the leaked value inside format-hint contexts with a format-preserving
neutral placeholder (e.g. `12.3%`), backing up each original to
`agent_system_prompt.txt.orig`. **305 prompts redacted**; exact-answer-in-prompt
dropped **29.4% → 6.3%** (residual = narrative/`yes`-`no` coincidences).

## Impact (re-run on cleaned prompts)
DeepSeek-V4-Flash agentic on the 200-item ladder subset:

| | accuracy | agentic gap (reading − agentic) |
|---|---|---|
| leaky prompts | 71.0% | −3.0 (looked "faithful") |
| **cleaned prompts** (n=162) | **54.3%** | **+13.7** |
| same-item overlap (n=78) | 69.2% → **59.0%** (−10.3 pt) | |

The leak inflated agentic accuracy by ~10 pt. Since the same ~26% of prompts were
seen by every model, all previously-reported agentic numbers are inflated by a
similar amount; the corrected numbers make agentic gaps **larger**, i.e. the
benchmark is *harder* and *more discriminating* than the leaky numbers suggested.

Files: `redact_prompts.py` (audit+fix), `results/leak_report.json`,
`results/deepseek_v4_clean_vs_leaky.json`. Prompt fixes live in the benchmark
repo (v2/finqa_agents/*/agent_system_prompt.txt, with .orig backups) and should
be pushed to the FinOpsBench release.
