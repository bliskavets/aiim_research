# exp_064 — GRPO per-position C / logprob profile (for adaptive pos_discount)

Diagnostic: run **plain GRPO** (loss unchanged) and log, per rollout token,
`C_{i,t} = -mean_topk log p` (confidence/peakedness) and `logprob_{i,t}` (of the
sampled token), accumulated by ABSOLUTE position into count/sum/sumsq for three
groups: all / correct / incorrect rollouts (correctness = exact boxed match).
Goal: see how C and logprob depend on generation position, to design a more
adaptive pos_discount (currently a fixed g(t)=τ/(τ+t)).

- `src/grpo_posstats_trainer.py` — GRPOPosStatsTrainer (observes in
  _generate_and_score, one extra no-grad forward for C+logprob; saves diag/posstats_<ds>.npz).
- accumulator CPU unit-tested (`tests/test_posstats.py`).
- `run_posstats.sh` — gsm8k + bigmath, 300 steps.
- `analyze_posstats.py` — figures/exp064_posstats_<ds>.png (C, logprob, coverage vs position).

Same hyperparameters as exp_058/062 (Qwen3-4B-Base, ng=4, bs=1, ga=4, 300 steps).

## Results (gsm8k 300 steps, bigmath 300 steps)

Per-position C(t) and logprob(t), split overall / O+ (correct) / O- (incorrect).
Figures: `figures/exp064_posstats_<ds>_{overall,Oplus,Ominus,OplusOminus}.png`.

**Robust pattern across both datasets:**
- **logprob(t) rises monotonically toward 0** with position (gsm8k -0.44→-0.14,
  bigmath -0.45→-0.18): later tokens are more predictable/routine, earlier tokens
  more surprising/decisive.
- **C(t) overall is roughly flat** (~11-12) after the first ~200 tokens — position
  alone is a weak signal; the value is in the O+/O- SHAPE.
- **O+ vs O- early-decisiveness (the discriminative signal):** correct rollouts
  (O+) are decisive EARLY — C spikes to ~13 by pos~200-500 then settles; incorrect
  (O-) start LOW (C~9-9.6) and climb monotonically, CROSSING O+ around pos~600-1000
  (confidently-wrong late). O+ also commits harder at the answer (logprob ends
  higher: gsm8k -0.05 vs -0.17).

**Implication for adaptive pos_discount:** the current fixed g(t)=τ/(τ+t) discounts
by absolute position, but C(t) overall is flat so position per se is weak. The real
signal is (a) surprisal −logp(o_t) (early tokens surprising → informative) and
(b) the sign of early C relative to the group (high early C ⇒ O+-like). A more
adaptive scheme should up-weight early decisive tokens / surprisal and avoid
rewarding late confidence, rather than a pure positional decay.
