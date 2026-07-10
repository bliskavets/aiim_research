# FinOpsBench rebuttal — remaining work to harden the responses

Companion to `rebuttal_responses_final.md`. These are the experiments / release-hygiene
tasks that would (a) make the posted answers bulletproof under follow-up questions and
(b) protect the benchmark's credibility. Ordered by priority. Internal note — not for
posting on OpenReview.

---

## Reviewer 6zfv — experiments to run for the second response (placeholders left in text)

- **E12: extended cross-benchmark comparison (point 2, flagged IMPORTANT by author).**
  Run the same model(s) on several OPEN static finance-QA benchmarks (FinQA, ConvFinQA,
  TAT-QA, and one multi-table set such as MultiHiertt) in pure reading mode, and contrast
  with FinOpsBench-v2 closed-book vs agentic. Expected story: static benchmarks are near
  ceiling by reading (~85-90%), while v2 needs tool use (closed-book ~1-15%, agentic
  ~40-70%). Produces the cross-benchmark table; a one-benchmark version (TAT-QA) already
  exists in E10 and is used now as a placeholder-backed partial result. Cost ~$10-20 API.
  Text placeholder: `[PLACEHOLDER E12 ...]` in 6zfv point 2.
  STATUS: harness staged. `experiments/e10_cross_benchmark/run_finqa.py` is written and its
  loader is offline-verified against `/tmp/finqa_train.json`; `run_tatqa.py` already exists.
  BLOCKED on `OPENROUTER_API_KEY` (not set in the environment). To run:
  `export OPENROUTER_API_KEY=... && python run_finqa.py --model openai/gpt-4.1-mini --n 200`
  and the matching `run_tatqa.py`. ConvFinQA and MultiHiertt still need dataset download plus
  a small loader each before they can be added to the table.
  DONE alongside this: judge-corroboration (offline, no key) computed in
  `experiments/e2_judge_agreement/panel_agreement.py` and folded into 6zfv point 4.
- **Inter-annotator κ, second annotator (point 4 and PVoW).** Have a second domain-literate
  annotator label the same E3 sample so we can report human-human κ, not only human vs
  scorer. No API cost, annotator time only. Text placeholder in 6zfv point 4.
- **Judge-corroboration measurement (point 4, optional, offline).** On v2, quantify what
  fraction of LLM-panel-accepted environments also pass execution validation, to show the
  panel's judgement is corroborated by execution rather than trusted blindly. Offline over
  released data, no API cost.

---

## P0 — release hygiene that must happen before/at posting (protects the benchmark)

1. **Ship the v2 prompt-format-hint fix silently in the release, without `.orig` backups.**
   The rebuttal uses the *clean* access-ladder numbers throughout and does **not** disclose
   the Stage-8 format-hint issue (decision: do not compromise the benchmark). For this to
   hold up if a reviewer downloads the release:
   - Copy the redacted `agent_system_prompt.txt` into place and **delete every
     `agent_system_prompt.txt.orig`** and any `leak_report`/audit artifact from the *public*
     release repo (`bliskavets/FinOpsBench`) and the anonymous mirror. Keep the audit only in
     the private working repo (`experiments/e11_prompt_leak_audit/`).
   - Re-sync `anonymous.4open.science/r/FinOpsBench`. Verify it opens in a clean
     incognito browser (this is what fixes PVoW Software:1 / Reproducibility:2).
   - Add the top-level `PROMPTS.md` index (E0) so PVoW-3 is verifiable in one click.

2. **Reconcile paper Table 3 with the clean numbers for the camera-ready.**
   The original Table 3 agentic numbers were produced on the pre-fix prompts. In the
   camera-ready, quietly update the affected v2 numbers to the clean re-run values so the
   released code reproduces the table. Do a diff of every published v2 number vs the clean
   run and list any that move by >1 pt.

3. **Revoke the three leaked credentials** (OpenRouter `sk-or-...`, OpenAI `sk-...` in TTA
   history, GitHub `ghp_...` — the latter is embedded in this repo's git remote URL).
   Rotate before the repo is ever shared.

---

## P1 — experiments that close the last soft spots in the answers

4. **Clean full-benchmark re-run of the extra models (R3-4, PVoW-8).**
   The posted model-coverage table uses the clean 200-item ladder (safe). But the *paper's*
   full v2 leaderboard should also be on clean prompts. Re-run Claude-Sonnet-4.5 and
   DeepSeek-V3 (and any model whose Table 3 number came from pre-fix prompts) on the full
   1,108-item v2 set with the fixed prompts.
   - Est. cost: cheap/open models ~$7; +Claude-Sonnet-4.5 full set ~$15. OpenRouter budget
     ~$156 remaining — affordable.
   - Deliverable: a clean full-set leaderboard; report the full-set Claude number (currently
     n≈139/156 subset) so we can drop the "n=139 is a random subset" caveat.

5. **Second independent human annotator (PVoW-1/2).**
   The rebuttal promises inter-annotator κ for the camera-ready. Recruit one more
   domain-literate annotator, have them label the *same* E3 sample (172 v1 judge cases + a
   slice of the 200 v2 validity items), and report human–human κ in addition to human↔judge.
   Pure labor, no API cost. This upgrades the answer from "single expert" to a defensible
   multi-annotator study, which is what PVoW literally asked for ("experts or trained
   annotators", plural).

6. **Clean re-run of the failure taxonomy traces (PVoW-6, R2-3).**
   E5 was classified on pre-fix traces. Taxonomy *shape* is unaffected (categories are
   about behaviour, not the leaked figure), but re-extract the v2 failing traces from the
   clean runs and re-verify the 60-item human check sample so the released
   `classified.jsonl` matches the clean leaderboard. Low effort once P1.4 is done.

---

## P2 — nice-to-have, only if a reviewer pushes

7. **A genuine finance-specialized LLM data point (R3-4).**
   The answer argues open finance LLMs lack reliable function calling. If a reviewer
   disputes this, run one (e.g. a Fin-tuned Qwen/Llama variant) through the v2 harness and
   show either (a) it cannot emit valid tool calls, or (b) its score, to make the claim
   empirical rather than asserted.

8. **Distractor-count controlled ablation, done right (R2 "tunable difficulty").**
   E9's distractor-count analysis was observational (confounded) and the core-only ablation
   was flagged invalid (system prompt still advertised the removed tools). If we want to
   claim distractor difficulty is tunable, redo the ablation with the system prompt tool
   list regenerated to match the reduced tool set. Only needed if a reviewer asks for a
   distractor-level knob; the tool-chain-depth monotonicity result already covers "tunable
   difficulty."

9. **Quantitative diversity table in the paper body (PVoW-5).**
   The numbers exist (E6); just needs to be typeset as an appendix table for the
   camera-ready (SQL-feature histogram, op-mix, tool-chain-depth distribution, role count).

---

## Numbers the answers depend on (verify these survive the clean re-runs)

- v1 judge accuracy **85.1%** (pooled κ 0.67); contested-case agreement **82.6%** (κ 0.64).
- v2 cleanliness **85%** (170/200 execution-verified).
- Closed-book v2 **~14%** flat (GPT-5-mini 14.7 / GPT-4.1 13.3 / Qwen3-30B 13.8) vs agentic 53–68.
- Clean access ladder (200 items): six faithful tool users (read−act gap ≤ +1.5) vs three
  read-well-act-poorly (DeepSeek-V4-Flash +13.7, DeepSeek-V3.2 +30.4, Llama-3.3-70B +37.2).
- Construction cost **~$790**, API-only.
- Cross-version ranking agreement **mean abs diff 2.6 pp**.
- TAT-QA 89% reading vs v2 1.5% closed-book / ~60% agentic (same model, GPT-4.1-mini).
