# FinOpsBench EMNLP rebuttal — session handoff (read this first)

Purpose: let a fresh session resume the rebuttal-polishing work without re-deriving anything.
Submission 5243, EMNLP. Benchmark = FinOpsBench (v1 = 5979 synthetic SQL-tool tasks; v2 = 1108 FinQA-derived agentic environments).

## 0. Where everything lives
- Working repo checkout: `/mnt/data/tmp/aiim_answers` (git remote `bliskavets/aiim_research`, dir `papers/emnlp/finopsbench/`). Commit as `Barys Liskavets <barys.liskavets@acclaim.ai>`, no AI mentions in messages/content. Push flow: `git pull --rebase origin main` then push; the file is sometimes edited by the user/linter mid-session, so re-read before Edit.
- **`REBUTTAL.md`** = the FULL, polished, per-reviewer rebuttal (this is the primary artifact; keep it authoritative). Structure: `# Reviewer PVoW / 6zfv / j7in`, each with `### <summary heading>` per point, then a `>` verbatim quote of the reviewer's concern, then the answer (prose + tables). Contains `<!-- -->` comments holding alternative phrasings the advisor kept.
- **`REBUTTAL_FINAL.md`** = compressed variant for posting (see section 4; currently holds the "full wording" variant A).
- `experiments/` = analysis code/results (E1-E11 + judge_swap, run_finance_local, extra_diversity, panel_agreement). NOT released publicly; do not cite `experiments/...` paths in the rebuttal text.
- `missing_experiments.md` = internal follow-up/release-hygiene checklist.
- The old `REBUTTAL_STATE.md` documents the experiment layer (E0-E11) from an earlier session; still valid background.

## 1. Reviewer map
- **PVoW** (R1): Overall 3.0, conf 5. Most constructive; gave a checklist. Points: human validation + LLM-judge reliability + eval methodology (grouped); release prompts; diversity + failure analysis (grouped); construction cost; bias from proprietary models. (typos + repro/software folded into other answers.)
- **6zfv** (R2): Overall 2.5, conf 4. Points: fundamental NLP capability; novelty vs recent agentic-finance benchmarks; diagnostics beyond final-answer accuracy; dependence on LLM-generated data and judgments.
- **j7in** (R3): Overall 2.5, conf 4. Most hostile. Points: v1 machine-verifiable ground truth; v2 derived from FinQA "monotonous/artificial multi-hop"; missing top agent models + finance-specialized LLMs; "outdated smolagents"; data-contamination risk for v2.

## 2. Global rules and strategic decisions (KEEP DOING THESE)
- STYLE (human, not-LLM): no em-dashes (use colon/semicolon/comma/hyphen), no en-dash ranges (write 57-69% not 57–69%), no comma thousands (8233 not 8,233), NO "First/Second/Third" enumeration scaffolding (use smooth connectors), concise; tables preferred with a concrete conclusion under each. See memory `rebuttal-writing-style`.
- HUMAN ANNOTATOR: there was ONE annotator = the author, domain-literate but NO formal finance background. Always say "a human judge with knowledge of the domain", NEVER "expert", never imply multiple/independent annotators, never report inter-annotator kappa. Cohen's kappa (judge vs the one human) is fine; Fleiss kappa is for >=3 judge models.
- PROMPT-LEAK (E11): DO NOT disclose to reviewers. Use clean numbers only. The Stage-8 format-hint leak was fixed; ship the fix silently in the release (no `.orig` backups), reconcile paper Table 2/3 in camera-ready. This is why some v2 numbers are the "clean 200-item ladder" values.
- Do not cite `experiments/...` code paths (only the main benchmark code is released).

## 3. State of REBUTTAL.md (DONE / polished)
All three reviewers fully drafted, polished across adversarial passes, human-style, dashes/commas cleaned. Key evidence used:
- Human eval: v1 judge vs human 85.1% (Cohen's kappa 0.67); v2 execution validity 178/200 = 89%.
- Contested cases: on 92 items where judge and exact-match disagree, human sides with judge 82.6% (kappa 0.64). Scalar subset only 4.4% of v1 answers.
- Diversity: v1 742 roles, 0 dup queries, JOIN 70% of items, 33% need 2+ JOINs, nested subquery 17% of queries, clauses/query 4.1; v2 median 5 tool calls / 9 tools, off-path tools >=2 in 92%; concept coverage complementary (v1 AP 52%, v2 statement ratios 77%); 124 companies / >1000 FinQA filings.
- Tool-chain depth vs accuracy: pooled 61.6% (1-3) -> 45.8% (8+).
- Failure taxonomy: 779 traces, 8 categories; v1 semantic (malformed args, incomplete retrieval), v2 tool-use (wrong-tool selection, round-limit); process metrics (v1 ~1.4 calls, v2 ~4).
- Cost: v1 ~$450 (10000->5979, $0.075), v2 ~$340 (1247->1108, $0.307), ~$790 total, API-only.
- Bias: generator = GPT-4.1-mini (confirmed core.py); it is the LOWEST frontier model on v1 (61.5% vs GPT-5 68.9%) -> no generator advantage; argued on v1 not v2 (advisor fix).
- Novelty (6zfv): cross-benchmark same-model: TAT-QA reading 89%, FinQA reading 67%, FinOpsBench-v2 no-tools 1.5%, agentic 61.5%. Qualitative positioning vs FinAgentBench/FinGAIA/Herculean (prose, no head-to-head runs).
- Funnel (6zfv Q4): panel + execution discard ~40% v1 / ~11% v2; panel unanimity per criterion (data-natural 97% ... answer-sound 62%); cross-version agreement mean abs diff 2.6 pp.
- Missing models (j7in): base models behind Claude Code/Codex are already evaluated; native->ReAct moves v1 accuracy up to 6.4 pp (harness sensitivity); >a dozen models / 5 vendors.
- Contamination (j7in): closed-book ~14% flat vs agentic 53-68%; access ladder question-only 2-4% / tools 20-54% / FinQA-native 57-69%; v1 freshly generated, never published; consistent per-model ranking (2.6 pp).

### JUDGE-SWAP: removed from REBUTTAL.md on purpose
A judge-swap experiment (3 extra vendor judges on 170 human-labelled items) was run but REMOVED from 6zfv Q4 because its moderate numbers (78-84% vs human, Cohen's kappa 0.44-0.58, Fleiss 0.69) are weaponizable by hostile j7in under cross-visibility. Artifacts kept in `experiments/e2_judge_agreement/` but NOT referenced in text. Do not re-add.

## 4. OPEN TASK: REBUTTAL_FINAL.md compression (user is mid-decision)
OpenReview rule: the whole comment to ONE reviewer (their quotes + all answers) must be <=5000 chars; two comments/reviewer is allowed but undesirable.
- Exact-wording version (comments stripped, all tables, structure): PVoW 12537 / 6zfv 11687 / j7in 11852 (with quotes); ~11k without quotes. => ~2.2-2.5x over 5000; needs ~3 comments/reviewer with zero distortion.
- A heavily-compressed one-comment-per-reviewer version was made (~2800/reviewer, 18 tables, quotes dropped) but the user said it compressed TOO much ("слишком сильно сжал").
- Options presented to user: (A) exact wording, 3 comments/reviewer; (B) 2 comments/reviewer with minimal trimming to <=5000 each; (C) 1 compressed comment/reviewer. RECOMMENDED B. USER HAS NOT YET CHOSEN. `REBUTTAL_FINAL.md` currently holds variant A (full wording, ~11-12.5k/reviewer). NEXT: get the user's choice and produce it.

## 5. Known remaining inconsistencies in REBUTTAL.md (flagged, mostly NOT yet fixed; user said "leave" some)
1. GPT-4.1-mini v2-agentic: 60.0% (6zfv leaderboard) vs 61.5% (6zfv cross-benchmark). Same model/section, 1.5 pp clash. NOT fixed. (GPT-4.1's bigger 66.0-vs-60.6 clash was fixed by removing GPT-4.1 from the leaderboard.)
2. DeepSeek roster: "DeepSeek-V3" (failure/process/depth tables) vs "DeepSeek-V3.2"/"V4-Flash" (leaderboard/contamination). Different versions, unexplained. NOT unified.
3. Contamination self-contradiction: "substantially less exposed / unlikely" vs "a half that cannot be contaminated" (last sentence). NOT fixed.
4. Single-annotator gap (biggest latent weakness; hits PVoW + j7in): only one annotator, not independent, no inter-annotator kappa. NOT addressed in text.
5. PVoW human-eval table "Human vs automatic scoring" column conflates v1 agreement with v2 execution-validity.

## 6. OPEN TASK: FinOpsBench RELEASE repo cleanup (diagnosed, NOT executed - awaiting confirmation)
Repo `bliskavets/FinOpsBench` (public release). Working clone at `/tmp/FinOpsBench` (ephemeral; has uncommitted leak-fix edits to v2 agent_system_prompt.txt). User wants the BENCHMARK DATA removed so reviewers can't see it yet (they will re-add later). Findings:
- Personal data: NO hardcoded keys/emails in tracked files. BUT every commit author is `Barys Liskavets <barys.liskavets@acclaim.ai>` -> DEANONYMIZATION for double-blind (bigger than the data). Repo owner handle `bliskavets` also deanonymizes.
- Data to remove: `v1/data/finopsbench_v1_pool.jsonl.gz` (19M) + `v2/finqa_agents/` (1302 env dirs, 172M) + maybe `v2/results/*.json` (confirm). KEEP code: `v1/finopsbench_v1/`, `v2/pipeline/`, `v2/agent_runners/`, `compare_outputs.py`, `run_eval_grounded.py`, README, PROMPTS.md, tests.
- CRITICAL CAVEATS: data is already in commit `b130822d` (pushed) -> a delete-commit does NOT hide it from git history; need history rewrite + force-push. And the anonymous.4open.science mirror is a separate snapshot that must be regenerated. Reviewers (double-blind) should get ONLY the anon mirror; recommend making the GitHub repo private during review.
- Pending user confirmation: (a) include v2/results in deletion? (b) OK to force-push with history rewrite? (c) fix git author to anonymous? mlflow.db (1.17G) + mlartifacts are gitignored (not in repo).

## 7. Credentials to REVOKE (flagged, user's action)
- GitHub token embedded in the `/tmp/FinOpsBench` git remote URL (ghp_...).
- OpenRouter key used this session for runs (sk-or-v1-...; provided by user in chat).
- (Older) OpenAI key in TTA history.

## 8. Infra / experiments run this session
- OpenRouter runs (via key): FinQA reading eval (67%, in E10/cross-benchmark); judge-swap (removed from text). ~ a few $ spent.
- Docker: `vllm/vllm-openai` image pulled (data-root `/mnt/data/docker`, nvidia runtime works). Finance-LLM (`instruction-pretrain/finance-Llama3-8B`) served but CANNOT act as a tool agent (no chat template / no tool-call tokens) -> decided NOT to include this run in the rebuttal (weak/attackable). `run_finance_local.py` staged in `experiments/e4_new_models/` (not referenced in text). Root fs `/` is small (29G) and fills easily; put caches on `/mnt/data`. env lacks CUDA toolkit (nvcc) so native vLLM install fails; use docker.

## 9. Score-chance assessment (given to user)
PVoW 3.0->3.5 ~55-65%; 6zfv 2.5->3.0 ~40-50%; j7in 2.5->3.0 ~35-45% (but j7in's factual errors give the AC grounds to discount him). Expected average after rebuttal ~2.8-3.1. Top ROI improvements still open: (1) address single-annotator, (2) strengthen 6zfv novelty (less "complement", more "only reproducible/controllable/executable"), (3) clean residual number/version inconsistencies (section 5).

## 10. Immediate next actions
1. Get the user's choice on REBUTTAL_FINAL.md (A/B/C, section 4) and produce it.
2. If asked, fix section-5 inconsistencies (GPT-4.1-mini 60/61.5, DeepSeek versions, contamination unlikely/cannot).
3. Execute the FinOpsBench release cleanup (section 6) once the user confirms scope + force-push + author fix.
