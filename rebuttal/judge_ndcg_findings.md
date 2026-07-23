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
