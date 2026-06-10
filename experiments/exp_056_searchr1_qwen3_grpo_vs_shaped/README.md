# exp_056 — Search-R1 with our 4-method shaping (Qwen3-4B)

Port of our GRPO + shaped-methods toolkit to **Search-R1** style training: the model interleaves `<think>`, `<search>`, retrieved `<information>`, and `<answer>` blocks to answer open-domain QA from Wikipedia.

We rebuilt this on the unsloth + TRL stack we already use (vs. the official Search-R1 implementation on verl), because all our shaping trainers (`GRPOSTrainer`, `GTPOConfTrainer`, `GTPOEMAFlippedTrainer`) plug into TRL's `GRPOTrainer` via a single `_compute_loss` override. Porting them to verl would mean rewriting everything; here they slot in unchanged.

## Status

This commit ships the **scaffolding + tests** for exp_056. The training run itself depends on a Wikipedia retrieval server (E5 + wiki-18 FAISS index, ~30 GB download) which is set up in a follow-up step. The rollout / reward / mask logic is fully unit-tested against a stub retriever; once the real retrieval server is up, swap `StubRetriever` for `HTTPRetriever` in `train.py`.

## Design

### Output format (Search-R1 official, verbatim)

```
<think> ... reasoning ... </think>
<search> query </search>
[server injects <information> ... </information>]
<think> ... </think>
<search> ... </search>          (optional second hop)
[server injects <information> ... </information>]
<answer> final answer </answer>
```

Stop strings: `</search>` and `</answer>`. On `</search>` we hit the retriever, format the docs as `<information>...</information>`, append, continue. On `</answer>` we stop.

### Rollout

`src/searchr1_rollout.py` exposes `run_rollouts(prompts, generate_fn, encode_fn, retriever, cfg)`:

- `generate_fn(prompts, sp)` — any function returning `[GenerationResult(text, token_ids, ...)]`. Lets us back the same rollout by unsloth+vllm in training and by a scripted fake LLM in tests.
- `retriever` — `StubRetriever` (mocks) or `HTTPRetriever` (talks to Search-R1's FastAPI server on `http://127.0.0.1:8000/retrieve`).
- Returns `RolloutTrace(completion_text, token_ids, model_mask, n_turns, n_searches, finish_reason, queries)`.

`model_mask` is the key trainer artefact: 1 for tokens the model produced, 0 for retrieval-injected `<information>` blocks. Used as `completion_mask` so loss and shaping only act on model tokens.

### Reward

`src/em_score.py` ports the official Search-R1 `qa_em.py` semantics (SQuAD-style normalization → lowercase, strip punctuation, drop articles, collapse whitespace):

- `reward_em(completion, gold)`: extract last `<answer>...</answer>`, normalize, return `1.0` if exact match else `0.0`. `format_score=0.0` if no `<answer>` tag at all.
- `reward_subem(...)`: looser variant, substring match.

This is the binary "did you get the right answer" signal used in the paper.

### Methods (same 4 as exp_054/055)

| method | shaping | tag-mask effect |
|---|---|---|
| `grpo` | none — baseline | n/a |
| `grpo_s_entropy` | seq-level entropy weighting (GRPO-S) | mask no-op (seq-level) |
| `gtpo_conf` | per-token confidence bonus | mask active |
| `gtpo_ema_flipped` | per-token EMA-flipped advantages | mask active |

Tag-mask in shaping trainers covers Search-R1 native tokens + Qwen3 chat tokens:
`<think>`, `</think>`, `<search>`, `</search>`, `<information>`, `</information>`,
`<answer>`, `</answer>`, `<|im_start|>`, `<|im_end|>` (configured in `train.py`).

### Dataset

`PeterJinGo/nq_hotpotqa_train` (HuggingFace) — Natural Questions + HotpotQA merged training split used by Search-R1. Fields: `question`, `golden_answers` (list).

### Retrieval (deferred)

The full Search-R1 setup expects E5 + wiki-18 FAISS index served on port 8000.
Setup steps (NOT YET RUN — see `retrieval/README.md`):

1. `pip install datasets faiss-gpu` (or `faiss-cpu`)
2. Download `PeterJinGo/wiki-18-corpus/wiki-18.jsonl.gz` (~3GB)
3. Download `PeterJinGo/wiki-18-e5-index/{part_aa,part_ab}` and `cat` them → `e5_Flat.index` (~30GB)
4. Launch:
   ```
   python search_r1/search/retrieval_server.py \
     --index_path e5_Flat.index --corpus_path wiki-18.jsonl \
     --topk 3 --retriever_name e5 --retriever_model intfloat/e5-base-v2 --faiss_gpu
   ```
5. Server exposes `POST http://127.0.0.1:8000/retrieve` with body `{"queries": [...], "topk": 3}`.

Once running, switch `StubRetriever()` → `HTTPRetriever()` in `train.py`.

## Files

```
README.md                       this file
run_056.sh                      docker launcher, all 4 methods sequential
plot_metrics.py                 4-metric grid
plot_reward_dynamics.py         single-panel rolling-20 reward
train.py                        method-switch trainer, dataset prep, reward wiring
src/
  searchr1_rollout.py           multi-turn rollout (think→search→info→answer)
  retriever.py                  StubRetriever + HTTPRetriever
  em_score.py                   Search-R1 EM reward port
  format_tag_mask.py            same as exp_054/055 but Search-R1 tags
  entropy_utils.py              (unchanged from exp_002)
  grpo_s_trainer.py             same shaping core
  confidence_utils.py
  gtpo_conf_trainer.py
  ema_flipped_utils.py
  gtpo_ema_flipped_trainer.py
tests/
  test_em_score.py              6 tests on EM normalization + extraction
  test_searchr1_rollout.py      5 tests with scripted fake LLM
  test_format_tag_mask.py       (inherited)
  test_methods.py               (inherited)
retrieval/
  README.md                     server setup steps
```

## Hypothesis

Same as the past two months of shaping work: per-token shaping (gtpo_conf, gtpo_ema_flipped) with tag-masking on structural tokens may help on a not-yet-saturated baseline. Search-R1 on Qwen3-4B is harder than Big-Math int-2000 (multi-hop, retrieval-augmented), so the baseline likely won't saturate — should give shaping room to act.

## Next steps

1. Set up E5 + wiki-18 retrieval server (see `retrieval/README.md`)
2. Implement `train.py` Search-R1 wiring: tokenizer, system prompt, custom rollout integration with TRL GRPOTrainer
3. Reparent shaping trainers to a `SearchR1GRPOTrainer` base that overrides `_generate_and_score_completions` with our multi-turn rollout
4. Smoke run (50 steps grpo) once retrieval is up
5. Full 4-method run

## Results

(to be filled in)

| method | reward L50 | EM accuracy | # searches/rollout | finish_reason | KL L50 |
|---|---|---|---|---|---|
| grpo               | tbd | tbd | tbd | tbd | tbd |
| grpo_s_entropy     | tbd | tbd | tbd | tbd | tbd |
| gtpo_conf          | tbd | tbd | tbd | tbd | tbd |
| gtpo_ema_flipped   | tbd | tbd | tbd | tbd | tbd |
