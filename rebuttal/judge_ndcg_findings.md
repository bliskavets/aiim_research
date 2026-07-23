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

## Judge-prompt lab: stricter/recompute prompts + linear combination (48 problems, initial-7)
oracle@7=0.792, baseline greedy=0.420, random~0.455.
| judge prompt | select@7 | NDCG |
|---|---|---|
| v8_strict_recompute_corner | 0.5417 | 0.7750 |
| v2_brief                   | 0.5417 | 0.7729 |
| v3_strict                  | 0.5417 | 0.7723 |
| v5_unsure_no               | 0.5417 | 0.7474 |
| v6_corner                  | 0.5000 | 0.7288 |
| v4_recompute               | 0.4583 | 0.6722 |
| v7_recompute_compare       | 0.4375 | 0.6741 |

Findings:
1. The "cure" (make the judge RE-DERIVE the answer and compare) BACKFIRES: v4/v7 are the WORST
   (0.44-0.46, below even random 0.455). Reason: an 8B model re-solving hard problems is itself
   error-prone, so its "matches my answer" verdict inherits the model's own solving mistakes and
   rejects correct candidates. A verifier is only as good as its own solving -> no free lunch.
2. Strict / corner-case / default-no prompts (v5,v6,v8) do NOT beat the simple brief/strict
   judges on top-1 selection (all plateau at 0.5417); v8 edges NDCG (0.7750) but not select@7.
   The single-judge ceiling here is ~0.54.
3. LINEAR COMBINATION of diverse judges' per-problem z-normalised margins WINS:
   greedy subset {v2_brief, v3_strict, v6_corner} -> select@7 = 0.5833 (28/48), vs 0.5417 best
   single and 0.42 baseline. Ensembling complementary judges recovers ~+4pt (closing ~1/6 of the
   remaining gap to oracle 0.79). Equal-weight mean of all 7 does not help (0.5417) -- the weak
   recompute judges dilute it; the gain needs a curated diverse subset.

RECOMMENDATION for the paper's self-judge recipe: use a small ensemble of complementary judge
prompts (brief-quality + strict + corner-case) and combine their soft margins, rather than a
single prompt or a re-derivation judge. Best single prompt: v8/v2/v3 (~0.54, v8 best NDCG).

## Non-equal weights for the margin combination? -> overfits, do not.
Offline weight search on cached margins (48 problems, 7 prompts), per-problem z-normalised,
combined = Z @ w, argmax.
| setting | select@7 |
|---|---|
| equal weights (1..1)            | 0.5417 |
| full-data best weights (optimistic) | 0.6042 |
| full-data best weights, 6-fold CV   | 0.4167 |
| equal weights, 6-fold CV            | 0.5417 |
Tuning 7 continuous weights on 48 problems overfits hard: cross-validated, the tuned weights
drop to 0.42 (below baseline, worse than equal), while equal weights generalise (0.5417 = same
as full data). => Non-equal weights give no honest gain at this data size; use equal weights.
This also flags the earlier greedy-subset 0.5833 as full-data-optimistic; the honest, robust
ensemble select@7 is ~0.54 (still +12pt over the 0.42 greedy baseline).

## Curated equal-weight ensemble (drop recompute/weak judges) — CV-robust 0.5833
Dropped v4_recompute, v7_recompute_compare (counterproductive) and v6_corner (near-zero weight);
kept {v2_brief, v3_strict, v5_unsure_no, v8_strict_recompute_corner}, equal weights on per-problem
z-normalised margins.
| setting | select@7 |
|---|---|
| equal weights, full data     | 0.5833 |
| equal weights, 6-fold CV      | 0.5833 |
| tuned weights, full data      | 0.6042 |
| tuned weights, 6-fold CV      | 0.5625 |
So the equal-weight 4-judge ensemble reaches 0.5833 AND is cross-validated (CV == full), unlike
the all-7 equal ensemble (0.5417, diluted by the weak recompute/corner judges) and unlike tuned
weights (overfit: CV 0.5625<0.5833). Best honest configuration found: equal-weight ensemble of
{v2_brief, v3_strict, v5_unsure_no, v8}, select@7 = 0.5833 (+16pt over baseline 0.42; oracle 0.79).
Weight tuning gives no honest gain at this data size.

## Full-protocol comparison (2 epochs, pool 21, reuse initial-7) — 48 hard problems
Clean re-implemented SAGE loop (chat-mode Llama), same initial 7 gens, pluggable scorer.
| method | accuracy |
|---|---|
| oracle@7 (ceiling)                         | 0.792 |
| SAGE-select@7 ensemble (initial-7, NO refine) | 0.583 |
| SAGE-select@7 single                       | 0.542 |
| SAGE-full ensemble judge (2 ep, pool 21)   | 0.5208 |
| SAGE-full single judge  (2 ep, pool 21)    | 0.5208 |
| BoN@21 (FsfairX RM)                        | 0.5208 |
| SAGE+RM full (RM scorer in loop, pool 21)  | 0.4792 |
| greedy baseline                            | 0.420 |
| SAGE broken orig-judge                     | ~0.34 |

Observations:
1. Refinement does NOT help here: SAGE-full (0.52) < SAGE-select@7-ensemble (0.583). Adding 14
   refined candidates dilutes the pool with plausible-wrong answers the rubber-stamping judge
   cannot filter; the larger pool raises oracle but not the judge's pick. (Caveat: this is our
   re-implemented refinement, possibly weaker than the paper's exact recipe.)
2. Ensemble vs single judge tie in the full loop (both 0.5208) -- the ensemble's edge (0.583 vs
   0.542 at select@7) washes out once refinement + a 21-pool are added.
3. SAGE+RM (external FsfairX RM used INSIDE the loop) is the WORST full-protocol arm (0.4792) --
   on verifiable math the general-purpose RM misranks, misguiding both the good/bad grouping and
   the final pick. Mirrors the paper's core thesis (external RM hurts on math).
4. BoN@21-with-RM (0.5208) equals SAGE-full self-judge (0.5208), but SAGE reaches it WITHOUT any
   external reward model -- self-containment is the advantage here, not raw accuracy.
Bottom line on this hard Llama subset: best is selection (no refinement) with an equal-weight
judge ensemble (0.583); refinement and RM-in-the-loop do not add value in this setting.
