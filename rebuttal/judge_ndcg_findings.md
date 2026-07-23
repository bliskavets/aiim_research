# Judge-quality NDCG analysis + SAGE+Llama root cause (2026-07-23)

Harness: rebuttal-external repro_out/judge_ndcg/. For each of 48 problems (hard MATH-500
seed-7 subset), took the 7 initial (epoch=-1) SAGE generations (Llama-3.1-8B-w8a16),
labelled each is_correct vs gold with gpt-4.1-mini (153/336 = 45.5% correct), then scored
each generation's self-judge contrastive margin under 4 judge prompts, and computed NDCG
(relevance = is_correct, score = margin) per problem, averaged over the 28 problems that
have both correct and incorrect gens (ranking signal).

## NDCG by judge prompt (chat-mode judge)
| prompt | NDCG | tag coverage | median judge chars |
|---|---|---|---|
| v2_brief  (one-line CoT then verdict) | 0.790 | 93.2% | 13015 |
| v3_strict (skeptical, <=3 sentences)  | 0.768 | 100%  |  4373 |
| v0_original (verbose template, 75% gate) | 0.721 | 100% |    34 |
| v1_terse  (verdict only, no CoT)      | 0.719 | 56.5% |    33 |

Takeaways:
- A short reasoning step helps the judge rank correct-vs-incorrect answers: v2/v3 (~0.77-0.79)
  clearly beat the no-reasoning judges v0/v1 (~0.72).
- v3_strict is the efficient sweet spot: NDCG 0.768, 100% tag coverage, only ~4.4k chars
  (v2 reaches 0.790 but at ~13k chars and 7% truncation).
- v1_terse's low coverage (56.5%) shows a verdict-only prompt often fails to emit a parseable tag.

## ROOT CAUSE of poor SAGE+Llama (0.34 on the hard subset)
Confirmed by a direct 2-call test on "What is 2+2?":
- SAGE_ORIG_NOTHINK=1 (the mode the SAGE runner uses): judge output = 18817 chars — the Llama
  judge in raw-completion mode RE-SOLVES the problem instead of verifying, rambling/looping,
  and truncates at max_tokens. ~20% of judge calls in the actual run never emit </verification>
  -> contrastive margin degenerates to 0 -> candidate ranking becomes ~random -> SAGE < baseline.
- SAGE_ORIG_NOTHINK=0 (chat template): judge output = 541 chars — clean, contained verification.

So the failure is NOT max_tokens (raising 4096->10240 does not help; 18k>10240 still truncates)
and NOT the method itself: it is the raw-completion judge mode forcing Llama to re-derive.
FIX: run the judge in chat mode (decouple judge engine from the SAGE_ORIG_NOTHINK generation
switch), or use a judge prompt that forbids re-solving (e.g., v3_strict). Under a chat-mode
judge the self-judge ranks with NDCG ~0.72-0.79, so SAGE selection should recover.

CAVEAT: the NDCG table above was produced with the judge in chat mode (the good mode). The
SAGE run that scored 0.34 used the orig raw-completion judge. A fully aligned rerun would use
the chat-mode judge inside SAGE.

## SAGE selection on the initial 7 with a FIXED (chat-mode) judge — reusing gens.json
Reused the 7 initial generations per problem; scored self-judge margins under the two
reasoning judges (chat mode) and selected argmax. 48 problems, hard seed-7 subset.
| method | accuracy |
|---|---|
| oracle@7 (any of 7 correct = selection ceiling) | 0.792 |
| SAGE-select@7, v2_brief judge  | 0.542 |
| SAGE-select@7, v3_strict judge | 0.542 |
| random-pick expectation        | 0.455 |
| greedy baseline (ref)          | 0.420 |
| SAGE with broken orig-mode judge (v10 run) | ~0.34 |
Conclusion: fixing the judge (chat mode + reasoning prompt) turns SAGE selection from
BELOW baseline (0.34, broken raw-completion judge that re-solves) to clearly ABOVE baseline
(0.542, +12pt over greedy). Both reasoning judges tie on top-1 selection (0.542) despite a
small NDCG gap (v2 0.79 vs v3 0.77); v3_strict is preferable (100% tag coverage, ~4.4k chars
vs 13k). Still well below oracle 0.79 -> the self-judge captures part, not all, of the signal.
This is the SELECTION step only (initial epoch, no textual-gradient refinement); full SAGE
with refinement on the fixed judge would likely be higher and is a separate longer run.

## Why ranking fails — case analysis (top-group had more wrong than right despite enough correct)
Severe failures are rare: v3_strict 1/48, v2_brief 3/48 (problems 468, 341, 81). Re-scoring
those with judge reasoning reveals ONE dominant cause:

**The self-judge rubber-stamps: it emits `<verification>yes</verification>` for EVERY candidate,
correct and wrong alike.** No candidate ever gets "no". So the contrastive margin never encodes
a correctness verdict — it only ranks by the *confidence of the yes*, which tracks superficial
features (fluency, decisive phrasing, formatting, even degenerate "/nothink ... Goodbye [close]"
repetition), not the math.

Concrete (prob 468: sqrt(t) in (2,3.5) -> t in (4,12.25) -> integers 5..12 = 8; gt=8):
- wrong  boxed=7, margin 3.25 (TOP)  judge: "yes ... $\boxed{7}$ /nothink Goodbye [close]..."
- correct boxed=8, margin 2.75       judge: "yes ... $\boxed{8}$"
Both affirmed "yes"; the off-by-one wrong answer (7) is phrased slightly more confidently and
outranks the correct 8. The judge cannot tell 7 from 8 because it does not actually recount --
it just affirms. Same pattern in 341 (gt -2, wrong +1 ranked above), 81 (gt 3, wrong 3.14 top).

Root pattern: failures concentrate on **plausible-wrong answers** (off-by-one 7 vs 8, sign -1 vs
-2, alt value 1/6/15 vs the truth) -- errors that require re-doing the computation to catch. The
brief/strict judge does NOT recompute, so it affirms all and ranks by confidence noise.

The deeper tension (ties the whole investigation together): a BRIEF judge does not discriminate
(rubber-stamps yes -> ranking = fluency noise); a VERBOSE judge that re-solves DOES discriminate
but in raw-completion mode rambles 18k chars and truncates before the verdict (the original
SAGE+Llama failure). v3_strict is the best compromise found (skeptical, bounded), but oracle 0.79
>> select 0.54 shows the self-judge still leaves ~25 points of achievable accuracy on the table
on this subset -- because it affirms rather than verifies.
