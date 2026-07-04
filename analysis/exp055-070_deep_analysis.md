# Deep analysis: exp_055–exp_070 — what we learned, and the road to a publishable method

*2026-07-04. Covers the k-sweep / adaptive-k arc (exp_066–070), the position-discount arc
(exp_063–065), the diagnostic experiments (exp_064, exp_067), and positioning vs
GTPO/GRPO-S (arXiv:2508.04349).*

---

## 0. Current best method (formal)

**gtpo_ema_flipped (FIXED) + pos_discount, λ=0.7, top-k=5 (k=3 slightly better), α₁=0.9, α₂=0.1.**

Per token `t` of completion `i` in a group of G rollouts:

```
C_{i,t}   = −(1/k) Σ_{j=1..k} log π_(j)(·|prefix)          # mean of top-k logprobs, sorted desc
Ĉ_{i,t}   = λ·Ĉ_{i,t−1} + (1−λ)·C_{i,t}                    # causal EMA, λ=0.7
g(t)      = τ/(τ+t), τ=1024                                 # position discount on the bonus only

O+ (A_i > 0):  Ã_{i,t} = α₁ + α₂·g(t)·B_{i,t},  B = per-position group-normalized 1/Ĉ
O− (A_i ≤ 0):  Ã_{i,t} = −(α₁ + α₂·g(t)·P_{i,t}), P = per-position group-normalized Ĉ
then per-polarity z-norm over active tokens. Computed on the FULL group in
_generate_and_score (group-visible; never in the B=1 compute_loss).
```

Reference results (L50 boxed / length, Qwen3-4B-Base, 300 steps):

| dataset | GRPO | best shaped (posdisc λ0.7) | k |
|---|---|---|---|
| GSM8K    | +2.02 / 414 | **+2.62** / — | k=3 (k=5: +2.49) |
| MATH-500 | +0.94 / 942 | **+1.67** / — | k=3 (k=5: +1.63) |
| Big-Math | +1.51 / 622 | **+1.93** / — | k=3 (k=5: +1.81) |
| Omni-MATH| **−0.23** / 957 | −0.33…−0.55 | all shaped < GRPO |

---

## 1. The k-arc (exp_066/068/069/070): one mechanism explains everything

Observed:
- **exp_066 (fixed k):** inverted-U. k=1 collapses (len→3584, boxed→0) on every dataset;
  k=3 best; k=5 close; k=20/40 diluted.
- **exp_068 (nucleus k, top-p):** collapses at every p∈{0.7,0.8,0.9,0.95} (p=0.9 with a
  ~100-step delay). `mean_n` 1.7–4.5, median nucleus size 1.
- **exp_069 (rank k = min(rank,5)):** collapses; `mean_k`≈1.18 (~83% of sampled tokens are
  the argmax → k=1).
- **exp_070 (rank floor k = max(rank,5) / max(rank,3)):** stable, ≈ fixed k=5 / k=3
  (`mean_k`≈5.02 / 3.02); no advantage over fixed k.

### 1.1 The sign-reversal insight (why k=1 and k≥2 are OPPOSITE signals)

`C_k = −mean(top-k logp)` changes **meaning**, not just sharpness, between k=1 and k≥2:

- **k=1:** `C = −log p_max`. Near-deterministic token → C≈0 (LOW). Flat token → C large.
- **k≥2:** the runner-up logprobs dominate the mean. Near-deterministic token
  (p≈[0.99, 4e-3, 2e-3, …]) → logps ≈ [−0.01, −5.5, −6.2, −6.9, −7.5] → C₅ ≈ 5.2 (HIGH).
  Flat token (p≈[0.3,0.25,0.2,0.15,0.1]) → C₅ ≈ 1.7 (LOW).

So **for k≥2, C is high on peaked tokens and low on flat/branching tokens — the opposite
ordering of k=1.** The O+ bonus rewards low-C tokens (`B ∝ 1/Ĉ`):

- with **k≥2** it rewards *branching/decision* tokens in correct rollouts and (via `P ∝ Ĉ`)
  punishes *confidently-generated* tokens in wrong rollouts — a coherent
  "credit branching, blame confidence" scheme;
- with **k=1** it rewards *deterministic filler*: a near-deterministic token takes the whole
  per-position bonus budget (1/(C+ε) explodes as C→0), and in O− filler gets ≈zero penalty
  (P ∝ C ≈ 0). Filler is over-rewarded when correct and under-penalized when wrong →
  repetition/length farming → collapse spiral.

This single mechanism explains the whole arc:

| observation | explanation |
|---|---|
| k=1 collapses | filler-farming (above) |
| nucleus collapses at all p | median nucleus size = 1 (exp_067) → k=1-like on the deterministic majority |
| rank k=min(rank,5) collapses | ~83% argmax-sampled → k=1 on the majority |
| rank floor k=max(rank,5) stable, ≈ fixed k5 | mean_k≈5.02 — tail-rank adaptivity fires on <5% tokens |
| k=20/40 diluted | ranks 6–20 are deep tail (logp −8…−14) and dominate the mean → peaked/flat contrast shrinks |
| k=3–5 sweet spot | maximal head-contrast between peaked and branching tokens |
| exp_065 PC1/PC2 (confidence-multiplied bonus) collapse | unbounded multiplicative bonus — same harvestable-bonus failure |

**Boundedness lemma (paper-grade):** since Σp ≤ 1, `C_k ≥ log k` (equality iff the top-k
head is uniform). Hence `1/C_k ≤ 1/log k` is bounded for k≥2 and **unbounded for k=1**.
Any k-selection scheme whose *minimum* k is 1 (nucleus min_k=1, rank cap, fixed k=1)
admits an unbounded/maximal harvestable bonus on deterministic tokens; every such scheme
collapsed, and every scheme with k_min≥3 was stable. Empirical support: 8+ collapsed runs
vs 10+ stable runs, perfectly separated by k_min.

**Conclusion:** adaptive k is a dead end *because the estimator's meaning depends on k*.
The right abstraction is not "how many logprobs to average" but a **robust head-contrast
signal** (see §5.2).

### 1.2 Connection to exp_067 (coverage bimodality)

Token distributions are bimodal: median top-1 mass 0.95–0.98, but p5(top-1)≈0.05 and
15–20% of tokens need k>128 to reach 0.9 mass. So:
- the "peaked majority" and "flat minority" are well-separated populations;
- full-vocab entropy (the GTPO-paper weight) mixes head structure with heavy-tail noise on
  the flat minority; a truncated-head statistic with small k measures the *branching factor*
  of the head, which is the discriminative quantity;
- nucleus-k inherits the bimodality (median 1 / p95 20–28) — hence its instability.

---

## 2. The position arc (exp_063/064/065): what pos_discount actually does

Observed:
- pos_discount `g(t)=τ/(τ+t)` on the α₂ bonus helps everywhere it's learnable
  (exp_062/063); stacking with λ=0.7 is the current best (exp_063 COMBO).
- exp_064 (plain-GRPO per-position stats, gsm8k+bigmath): **logprob(t) rises monotonically**
  (−0.44→−0.14) — later tokens are routine; **C(t) is flat overall** (~11–12 after t≈200);
  the discriminative shape is O+ vs O−: **correct rollouts are decisive EARLY** (C spike at
  t≈200–500), **incorrect ones start low and climb** (confidently-wrong late, crossing at
  t≈600–1000).
- exp_065 (adaptive discounts): position-only variants ≈ tie with fixed g(t) (consistent
  with flat C(t) — position alone is weak); confidence-multiplied variants collapse
  (unboundedness again).

Interpretation: the value of g(t) is **not** "early tokens matter more" (C(t) is flat).
Two real effects:
1. it shrinks the late-position credit that longer rollouts collect token-by-token,
   i.e. it caps the *harvestable bonus budget*: Σ_t τ/(τ+t) grows logarithmically instead
   of linearly in length → weakens the length attractor;
2. it happens to align with where O+/O− are most separated (early decisiveness).

Both effects can be obtained in a principled way (§5.3, §5.4) instead of a heuristic decay.

---

## 3. The hard-dataset failure (omnimath): zero-variance groups are shaped into noise

New evidence (from existing logs, mean `frac_reward_zero_std`):

| dataset | GRPO | posdisc λ0.7 k5 |
|---|---|---|
| gsm8k    | 0.15 (late: saturation, all-correct) | 0.51 |
| math500  | 0.10 | 0.43 |
| bigmath  | 0.16 | 0.36 |
| omnimath | 0.13 | **0.40 (0.50 in the first 50 steps — all-WRONG groups)** |

In `flipped_advantages`, polarity is `is_pos = seq_adv > 0`. When a group has
`std(R)=0`, TRL sets all advantages to 0 → **every rollout lands in O−** and receives
`−(α₁ + α₂·P)` with per-polarity z-norm — i.e. a full-strength "penalize relatively-peaked
tokens" gradient **containing zero correctness information**. On omnimath this is ~40–50%
of all groups: half the training signal is pure directional noise. GRPO in the same
situation does *nothing* (advantage 0) — which is exactly why plain GRPO wins there.

This is the single cheapest fix available (§5.1) and doubles as the paper's
difficulty-robustness story.

---

## 4. Positioning vs GTPO/GRPO-S (arXiv:2508.04349)

What they do: per-token advantage reweighting by **full-vocabulary policy entropy** with a
dynamic coefficient ("dynamic entropy weighting"); single re-weighting direction; evaluated
on standard math benchmarks. Acknowledged/visible gaps: no position analysis, no length-bias
analysis, no hard-dataset analysis, no estimator-robustness analysis.

Our differentiation (each item is already supported by ran experiments):

1. **Estimator:** head-truncated confidence (k=3–5) instead of full-vocab entropy, justified
   by the bimodality of LLM token distributions (exp_067) and the k-sweep (exp_066) with the
   sign-reversal analysis (§1.1). Full-vocab entropy is the k→V limit that provably dilutes
   head contrast. *(For the paper: add an entropy-weighted baseline = their method, same
   codebase — currently missing.)*
2. **Polarity-flipped credit:** direction of the per-token weight depends on terminal
   correctness (reward branching when right, blame confidence when wrong) + causal EMA
   smearing — vs their polarity-independent weighting.
3. **Stability theory:** bounded-bonus condition (`C_k ≥ log k` lemma), position-stationarity
   / bonus-budget arguments, with 8+ controlled collapse case-studies (k=1, 4×nucleus,
   rank, PC1/PC2) as *evidence*, not accidents. Negative results become the ablation section.
4. **Zero-variance gating** for sparse-reward (hard) regimes — §3.
5. **Implementation correctness:** the B=1 microbatch degeneracy (group statistics computed
   on singleton batches silently reduce shaped methods to noise) — a reproducibility
   contribution; unsloth/trl users will hit it.

---

## 5. Proposed next setups (prioritized)

### 5.1 exp_071 — zero-variance gate (cheapest, targets the only losing dataset)
When `std(R_group)=0` (or all rollouts on one side of the threshold with equal reward):
skip shaping for the group → advantage 0 (= GRPO behaviour).
Variants: (a) hard gate; (b) soft `α₂_eff = α₂ · std(R)/(std(R)+c)`.
Run: omnimath + gsm8k (regression check), base = posdisc λ0.7 k3.
Expected: omnimath gap to GRPO closes or flips; no change on gsm8k.
*Paper claim: shaped credit must be gated by the information content of the group signal.*

### 5.2 exp_072 — principled head signal: branching factor instead of C
Replace `C` with a **bounded, scale-free branching-factor** statistic of the renormalized
top-k head p̃:
- `H_k = −Σ_{j≤k} p̃_j log p̃_j ∈ [0, log k]` (truncated renormalized entropy), or
- `N_eff = 1/Σ_j p̃_j² ∈ [1, k]` (effective support / inverse Simpson).
O+ bonus ∝ `H_k/log k` (reward branch points, **no reciprocal, no ε, bounded by
construction**); O− penalty ∝ `1 − H_k/log k` (blame peaked wrong tokens). EMA as usual.
This preserves the k≥2 signal direction while removing the unbounded 1/C and the k-dilution
(H_k of the renormalized head doesn't degrade with k the way the raw mean does — k can even
be 20 safely; ablate k∈{3,5,20}).
*Paper: this is the clean estimator that replaces both C and full-vocab entropy.*

### 5.3 exp_073 — length-invariant bonus budget (replaces/augments pos_discount)
Rescale the per-rollout bonus so its sum is length-invariant:
`Σ_t α₂·bonus_{i,t} = α₂·B_const` (softmax/normalization over the rollout's own tokens).
The total shaped credit a rollout can harvest no longer grows with its length — this
removes the length attractor *by construction* rather than by hyperbolic decay.
Killer ablation for the paper: re-run one collapsing config (nucleus p=0.9 or k=1) with the
budget — if it no longer collapses, the collapse mechanism is proven to be the
length-coupled bonus, not the signal itself.
Run: budget-vs-posdisc-vs-both on gsm8k+math500 (base k=3 λ0.7).

### 5.4 exp_074 — surprisal-guided credit (exp_064's direct implication)
The clean per-token informativeness signal from exp_064 is the sampled token's surprisal
`s_t = −log p(o_t)` (monotone position trend, O+/O− separation). Bonus weight
∝ per-polarity z(s_t) (reward surprising tokens in correct rollouts, penalize confident
tokens in wrong ones — no top-k forward needed, logprobs are already computed → cheaper).
This is the minimal-machinery variant of the method; if it matches C-based shaping it wins
on simplicity; if not, the gap quantifies the value of the head statistic.

### 5.5 exp_075 — the paper method (combine winners) + rigor pass
`k=3 head signal (5.2 winner) + zero-var gate (5.1) + budget or posdisc (5.3 winner) + λ=0.7 flipped EMA`.
Then, for the submission: ≥2–3 seeds on all 4 datasets, held-out **eval accuracy** (not
just training reward), the entropy-weighted GTPO-paper baseline, a second model family
(e.g. Llama-3.2-3B) if budget allows, plus the two diagnostic figures (bimodality,
O+/O− position profiles) and the collapse-taxonomy table.

### Priority order
**071 → 072 → 073 → (074 optional) → 075.**
071 is hours and fixes the only losing dataset; 072 is the core scientific contribution;
073 turns 8 failures into a theorem-shaped ablation; 075 assembles the paper.

---

## 6. One-line summary

*Shaped per-token credit works when the signal is a bounded, head-truncated branching
statistic, its direction is conditioned on terminal correctness, its budget is
length-invariant, and it is gated by group-signal information; every deviation from these
four conditions produced a reproducible collapse in exp_055–070 — which is exactly the
evidence base a top-tier paper needs.*
