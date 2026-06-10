"""
searchr1_trainer.py — TRL GRPOTrainer extended with Search-R1 multi-turn rollouts.

Contract of `_generate_and_score_completions` in modern TRL/Unsloth:
    inputs : list[dict]  — one dict per row from the dataloader.
                           Each dict has at least "prompt" (list of {role, content})
                           plus any extra columns from the dataset row (we use "answer").
    returns : dict[str, Tensor] — (B*G, ...) tensors used by `_compute_loss`:
        prompt_ids, prompt_mask, completion_ids, completion_mask,
        advantages, old_per_token_logps, [ref_per_token_logps]
"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from trl import GRPOTrainer

from .retriever import Retriever
from .searchr1_rollout import (
    GenerationResult, RolloutConfig, RolloutTrace, run_rollouts_batched,
)


def _build_vllm_generate_fn(llm, tokenizer):
    from vllm import SamplingParams

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
    """GRPOTrainer with Search-R1 multi-turn rollouts via run_rollouts()."""

    def __init__(self, *args,
                 retriever: Retriever,
                 rollout_cfg: RolloutConfig,
                 reward_fn: Optional[Callable[[List[str], List[Any]], List[float]]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._retriever = retriever
        self._rollout_cfg = rollout_cfg
        self._sr1_reward_fn = reward_fn

    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        tokenizer = self.processing_class

        # 1. Apply chat template to every row.
        from trl.data_utils import maybe_apply_chat_template
        prompts_text = [
            maybe_apply_chat_template(ex, tokenizer)["prompt"] for ex in inputs
        ]

        # 2. Tokenize with left padding (so the prompts align right-side, matching TRL).
        prompt_inputs = tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)

        # Trim to max_prompt_length if set.
        if self.max_prompt_length is not None and prompt_ids.shape[1] > self.max_prompt_length:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # 3. Run multi-turn rollouts; repeat each prompt num_generations times.
        G = self.num_generations
        grouped_prompts = [p for p in prompts_text for _ in range(G)]

        # Push the latest LoRA weights into vLLM.
        if hasattr(self, "_move_model_to_vllm"):
            if getattr(self, "_last_loaded_step", None) != self.state.global_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

        llm = getattr(self, "llm", None) or getattr(self, "vllm_engine", None)
        if llm is None:
            raise RuntimeError(
                "SearchR1GRPOTrainer needs a vLLM engine on `self.llm` or "
                "`self.vllm_engine`. Load the model with FastLanguageModel(fast_inference=True).")
        generate_fn = _build_vllm_generate_fn(llm, tokenizer)
        encode_fn = lambda txt: tokenizer.encode(txt, add_special_tokens=False)

        traces: List[RolloutTrace] = run_rollouts_batched(
            prompts=grouped_prompts,
            generate_fn=generate_fn,
            encode_fn=encode_fn,
            retriever=self._retriever,
            cfg=self._rollout_cfg,
        )

        # 4. Build padded completion tensors.
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        max_T = max((len(t.token_ids) for t in traces), default=1)
        max_T = max(max_T, 1)
        BG = len(traces)
        completion_ids = torch.full((BG, max_T), pad_id, dtype=torch.long, device=device)
        completion_mask = torch.zeros((BG, max_T), dtype=torch.long, device=device)
        for i, t in enumerate(traces):
            L = len(t.token_ids)
            if L == 0:
                continue
            completion_ids[i, :L] = torch.tensor(t.token_ids, dtype=torch.long, device=device)
            completion_mask[i, :L] = torch.tensor(t.model_mask, dtype=torch.long, device=device)

        # 5. Repeat prompts for the group dimension.
        prompt_ids_rep = prompt_ids.repeat_interleave(G, dim=0)
        prompt_mask_rep = prompt_mask.repeat_interleave(G, dim=0)

        # 6. Compute reward per rollout from our reward_fn.
        completion_texts = [t.completion_text for t in traces]
        # Gold answers: pull from inputs and repeat per generation.
        golds = [x.get("answer") for x in inputs]
        grouped_golds = [g for g in golds for _ in range(G)]
        if self._sr1_reward_fn is None:
            rewards = torch.zeros(BG, device=device)
        else:
            rewards = torch.tensor(
                self._sr1_reward_fn(completion_texts, grouped_golds),
                device=device, dtype=torch.float32,
            )

        # 7. Group-normalise → advantages (mirrors TRL's "group" scale_rewards).
        rewards_grouped = rewards.view(-1, G)
        mean_g = rewards_grouped.mean(dim=1, keepdim=True)
        std_g = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-4)
        advantages = ((rewards_grouped - mean_g) / std_g).view(-1)

        # 8. old_per_token_logps + (optional) ref_per_token_logps. When
        # beta != 0.0 and ref_model is None, fall back to the policy model
        # with LoRA disabled (matches TRL's default behaviour).
        full_ids = torch.cat([prompt_ids_rep, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask_rep, completion_mask.clamp(max=1)], dim=1)
        L_comp = completion_ids.shape[1]
        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, full_ids, full_mask, L_comp, compute_entropy=False,
            )
            ref_per_token_logps = None
            if self.beta != 0.0:
                if getattr(self, "ref_model", None) is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, full_ids, full_mask, L_comp, compute_entropy=False,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, full_ids, full_mask, L_comp, compute_entropy=False,
                        )

        # 9. Logging metrics.
        n_search_mean = float(sum(t.n_searches for t in traces)) / max(1, BG)
        finish_pct_answer = sum(1 for t in traces if t.finish_reason == "answer") / max(1, BG)
        finish_pct_trunc = sum(1 for t in traces if t.finish_reason == "truncated") / max(1, BG)
        finish_pct_max = sum(1 for t in traces if t.finish_reason == "max_turns") / max(1, BG)
        completion_lengths = completion_mask.sum(dim=1).float()
        self._metrics[mode].setdefault("searchr1/n_searches_mean", []).append(n_search_mean)
        self._metrics[mode].setdefault("searchr1/frac_finish_answer", []).append(finish_pct_answer)
        self._metrics[mode].setdefault("searchr1/frac_finish_truncated", []).append(finish_pct_trunc)
        self._metrics[mode].setdefault("searchr1/frac_finish_max_turns", []).append(finish_pct_max)
        self._metrics[mode].setdefault("rewards/em/mean", []).append(rewards.mean().item())
        self._metrics[mode].setdefault("completions/mean_length", []).append(completion_lengths.mean().item())
        self._metrics[mode].setdefault("completions/min_length", []).append(completion_lengths.min().item())
        self._metrics[mode].setdefault("completions/max_length", []).append(completion_lengths.max().item())
        self._metrics[mode].setdefault("reward", []).append(rewards.mean().item())
        self._metrics[mode].setdefault("reward_std", []).append(rewards.std().item())
        self._metrics[mode].setdefault("frac_reward_zero_std",
            []).append(((std_g.squeeze(1) < 1e-3).float().mean()).item())

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
