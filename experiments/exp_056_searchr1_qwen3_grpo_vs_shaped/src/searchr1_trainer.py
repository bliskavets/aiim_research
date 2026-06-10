"""
searchr1_trainer.py — TRL GRPOTrainer extended with Search-R1 multi-turn rollouts.

Design
======

TRL's `GRPOTrainer._generate_and_score_completions` produces, for one
training step, a dict containing:
    prompt_ids        (B, P)
    prompt_mask       (B, P)
    completion_ids    (B, T)
    completion_mask   (B, T)   1 = train on this token, 0 = ignore
    advantages        (B,)
    old_per_token_logps (B, T)
    ref_per_token_logps (B, T)
    + reward-component aggregates for logging

We override this method to:
  1. Decode prompts to strings.
  2. Call `run_rollouts()` from searchr1_rollout.py for each prompt (×num_generations).
     The rollout chains vLLM `generate(stop=["</search>", "</answer>"])` calls,
     hitting the retriever between turns.
  3. Each rollout returns a `RolloutTrace` with token_ids + model_mask.
     model_mask = 1 for tokens the model produced, 0 for retrieval-injected
     `<information>` content. We use model_mask as TRL's completion_mask so
     the loss / shaping signal only acts on tokens the policy controlled.
  4. Build the input dict TRL expects, recomputing `old_per_token_logps`
     and `ref_per_token_logps` (vs. trying to capture them during rollout)
     because the multi-turn structure makes mid-rollout logp accumulation
     fragile.

Subclasses (GRPOSTrainer, GTPOConfTrainer, GTPOEMAFlippedTrainer) keep their
`_compute_loss` overrides unchanged — the new completion_ids/completion_mask
shape is invariant.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from trl import GRPOTrainer

from .retriever import Retriever
from .searchr1_rollout import (
    GenerationResult, RolloutConfig, RolloutTrace, run_rollouts,
)


def _build_vllm_generate_fn(llm, tokenizer) -> Callable[[List[str], dict], List[GenerationResult]]:
    """Wrap a vLLM `LLM` instance into the GenerateFn protocol used by run_rollouts."""
    from vllm import SamplingParams  # imported lazily so tests don't need vllm

    def _gen(prompts: List[str], sp_dict: dict) -> List[GenerationResult]:
        sp = SamplingParams(
            n=1,
            max_tokens=sp_dict["max_tokens"],
            temperature=sp_dict["temperature"],
            top_p=sp_dict["top_p"],
            stop=sp_dict["stop"],
            include_stop_str_in_output=sp_dict.get("include_stop_str_in_output", True),
            seed=sp_dict.get("seed"),
        )
        outs = llm.generate(prompts, sp, use_tqdm=False)
        results: List[GenerationResult] = []
        for o in outs:
            sample = o.outputs[0]
            results.append(GenerationResult(
                text=sample.text,
                token_ids=list(sample.token_ids),
                finish_reason=sample.finish_reason,
                stopped_at=sample.stop_reason,
            ))
        return results

    return _gen


class SearchR1GRPOTrainer(GRPOTrainer):
    """TRL GRPOTrainer with Search-R1 multi-turn rollouts.

    Extra kwargs:
        retriever: Retriever  — instance the rollout calls each search turn
        rollout_cfg: RolloutConfig — turns/topk/budget
        reward_fn: callable(completions, gold_answers) -> List[float]
                   wraps em_score.reward_em or similar
    """

    def __init__(self, *args,
                 retriever: Retriever,
                 rollout_cfg: RolloutConfig,
                 reward_fn: Optional[Callable[[List[str], List[Any]], List[float]]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._retriever = retriever
        self._rollout_cfg = rollout_cfg
        self._sr1_reward_fn = reward_fn

    # ─────────────────────────────────────────────────────────────────────
    # Rollout override
    # ─────────────────────────────────────────────────────────────────────
    def _generate_and_score_completions(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop-in replacement for the TRL method, doing multi-turn search rollouts.

        Build of `inputs` follows the TRL parent: prompts are pre-tokenised by
        the data collator under `inputs['prompt_ids']` etc., but we re-decode
        to text because the rollout needs string-level stop matching.
        """
        device = self.accelerator.device
        tokenizer = self.processing_class
        prompt_ids: torch.Tensor = inputs["prompt_ids"].to(device)
        prompt_mask: torch.Tensor = inputs["prompt_mask"].to(device)

        # 1. Decode prompts to text — one per row of the (B, P) batch.
        prompt_texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)

        # 2. Repeat each prompt G times for the GRPO group.
        G = self.num_generations
        grouped_prompts = [p for p in prompt_texts for _ in range(G)]

        # 3. Pull the live vLLM engine from the parent (Unsloth keeps one).
        llm = getattr(self, "llm", None) or getattr(self, "vllm_engine", None)
        if llm is None:
            raise RuntimeError(
                "SearchR1GRPOTrainer requires a vLLM engine on `self.llm` or "
                "`self.vllm_engine`. Make sure FastLanguageModel was loaded with "
                "fast_inference=True.")
        generate_fn = _build_vllm_generate_fn(llm, tokenizer)

        encode_fn = lambda txt: tokenizer.encode(txt, add_special_tokens=False)

        # 4. Run the rollouts.
        traces: List[RolloutTrace] = run_rollouts(
            prompts=grouped_prompts,
            generate_fn=generate_fn,
            encode_fn=encode_fn,
            retriever=self._retriever,
            cfg=self._rollout_cfg,
        )

        # 5. Pad completions to a common (B*G, T) tensor.
        completion_ids_list = [torch.tensor(t.token_ids, dtype=torch.long) for t in traces]
        model_mask_list = [torch.tensor(t.model_mask, dtype=torch.long) for t in traces]
        max_T = max((x.shape[0] for x in completion_ids_list), default=1)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        completion_ids = torch.full((len(traces), max_T), pad_id, dtype=torch.long)
        completion_mask = torch.zeros((len(traces), max_T), dtype=torch.long)
        for i, (c, m) in enumerate(zip(completion_ids_list, model_mask_list)):
            completion_ids[i, : c.shape[0]] = c
            completion_mask[i, : m.shape[0]] = m
        completion_ids = completion_ids.to(device)
        completion_mask = completion_mask.to(device)

        # 6. Repeat prompts for the group too.
        prompt_ids_rep = prompt_ids.repeat_interleave(G, dim=0)
        prompt_mask_rep = prompt_mask.repeat_interleave(G, dim=0)

        # 7. Compute rewards. The data collator stores per-row gold answers
        #    under inputs['answer'] (string or list of strings).
        completion_texts = [t.completion_text for t in traces]
        golds_per_prompt = inputs.get("answer", [None] * len(prompt_texts))
        grouped_golds = [g for g in golds_per_prompt for _ in range(G)]
        if self._sr1_reward_fn is None:
            rewards = torch.zeros(len(traces), device=device)
        else:
            rewards = torch.tensor(
                self._sr1_reward_fn(completion_texts, grouped_golds),
                device=device, dtype=torch.float32,
            )

        # 8. Group-normalize → advantages (same as standard GRPO).
        rewards_grouped = rewards.view(-1, G)
        mean_g = rewards_grouped.mean(dim=1, keepdim=True)
        std_g = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-6)
        advantages = ((rewards_grouped - mean_g) / std_g).view(-1)

        # 9. Compute old_per_token_logps + ref_per_token_logps under no-grad.
        input_ids = torch.cat([prompt_ids_rep, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask_rep, completion_mask.clamp(max=1)], dim=1)
        logits_to_keep = completion_ids.shape[1]
        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
            )
            ref_per_token_logps = None
            if self.beta != 0.0 and self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model, input_ids, attention_mask, logits_to_keep, compute_entropy=False,
                )

        # 10. Logging aggregates.
        mode = "train" if self.model.training else "eval"
        n_search_mean = float(sum(t.n_searches for t in traces)) / max(1, len(traces))
        finish_pct_answer = sum(1 for t in traces if t.finish_reason == "answer") / max(1, len(traces))
        finish_pct_trunc = sum(1 for t in traces if t.finish_reason == "truncated") / max(1, len(traces))
        finish_pct_max = sum(1 for t in traces if t.finish_reason == "max_turns") / max(1, len(traces))
        self._metrics[mode].setdefault("searchr1/n_searches_mean", []).append(n_search_mean)
        self._metrics[mode].setdefault("searchr1/frac_finish_answer", []).append(finish_pct_answer)
        self._metrics[mode].setdefault("searchr1/frac_finish_truncated", []).append(finish_pct_trunc)
        self._metrics[mode].setdefault("searchr1/frac_finish_max_turns", []).append(finish_pct_max)
        self._metrics[mode].setdefault("rewards/em/mean", []).append(rewards.mean().item())

        out: Dict[str, Any] = {
            "prompt_ids": prompt_ids_rep,
            "prompt_mask": prompt_mask_rep,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }
        if ref_per_token_logps is not None:
            out["ref_per_token_logps"] = ref_per_token_logps
        return out
