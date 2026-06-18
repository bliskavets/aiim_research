"""
gtpo_ema_lenpen_gated_trainer.py
--------------------------------
exp_058 NEW method #2: gtpo_ema_flipped + Lagrange-like length penalty that is
GATED by low-temperature success (multi-temperature heuristic).

Per generation batch, in addition to the `num_generations` training completions
(sampled at the training temperature t=1.0), we sample 2 EXTRA completions per
prompt that are NOT used in the gradient update:
    - one greedy (t = 0)
    - one at t2 (0 < t2 < t,  default 0.5)
both with the same max_tokens. If EITHER extra completion gives the exact answer
(boxed integer == gold), the problem is *concisely solvable*, so we apply the
length penalty  alpha_len * max(0, |o| - L)  to ALL of that prompt's training
completions. If neither low-temp sample solves it, the penalty is skipped for
that prompt (it may genuinely need long exploration).

Implementation: override `_generate_and_score_completions` to (a) run the 2 extra
vLLM generations on the same engine/LoRA (wake → generate → sleep), (b) score
them, (c) attach a per-completion gate `len_gate` that flows through the buffer to
`compute_loss`. If the extra generation or the gate propagation fails, we fall
back to the ungated penalty (= method #1) and flag it via a metric, so the run
never crashes silently.
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages


def _answers_equal(guess, gold):
    if guess is None:
        return False
    try:
        return float(str(guess).strip().replace(",", "")) == float(str(gold).strip().replace(",", ""))
    except (ValueError, TypeError):
        return False


class GTPOEMAFlippedLenPenGatedTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        alpha_len            = kwargs.pop("alpha_len", 0.0015)
        length_L             = kwargs.pop("length_L", 1024)
        gate_temps           = kwargs.pop("gate_temps", (0.0, 0.5))
        gate_max_tokens      = kwargs.pop("gate_max_tokens", 3584)
        answer_extractor     = kwargs.pop("answer_extractor", None)
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs
        self.alpha_len, self.length_L = alpha_len, length_L
        self.gate_temps, self.gate_max_tokens = tuple(gate_temps), gate_max_tokens
        self.answer_extractor = answer_extractor
        self._gate_ok_logged = False

    # ── low-temp gate: extra vLLM generations per prompt ─────────────────────
    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        ng = self.num_generations
        n_comp = out["completion_ids"].shape[0]
        device = out["completion_ids"].device
        try:
            # `inputs` is already expanded: one entry per completion (the same
            # prompt repeated num_generations times). Dedupe by prompt text so we
            # run the gate generations once per UNIQUE prompt, then map back per
            # completion. Robust to ordering / batch shape (no P*ng assumption).
            assert len(inputs) == n_comp, f"len(inputs)={len(inputs)} != n_comp={n_comp}"
            keys = [str(x["prompt"]) for x in inputs]
            uniq = {}                       # prompt-text -> first index
            for i, k in enumerate(keys):
                uniq.setdefault(k, i)
            uniq_keys = list(uniq.keys())
            prompt_texts = [
                self.processing_class.apply_chat_template(
                    inputs[uniq[k]]["prompt"], tokenize=False, add_generation_prompt=True)
                for k in uniq_keys
            ]
            uniq_answers = [inputs[uniq[k]].get("answer") for k in uniq_keys]
            from vllm import SamplingParams
            if hasattr(self.llm, "wake_up"):
                try: self.llm.wake_up()
                except Exception: pass
            lora = self.model.load_lora("grpo_trainer_lora_model", load_tensors=True)
            solved = [False] * len(uniq_keys)
            for T in self.gate_temps:
                sp = SamplingParams(temperature=float(T), max_tokens=self.gate_max_tokens, n=1)
                gens = self.llm.generate(prompt_texts, sampling_params=sp, lora_request=lora, use_tqdm=False)
                for j, g in enumerate(gens):
                    guess = self.answer_extractor(g.outputs[0].text) if self.answer_extractor else None
                    if _answers_equal(guess, uniq_answers[j]):
                        solved[j] = True
            if hasattr(self.llm, "sleep"):
                try: self.llm.sleep(level=1)
                except Exception: pass
            gate_by_key = {k: solved[j] for j, k in enumerate(uniq_keys)}
            gate = torch.tensor([1.0 if gate_by_key[keys[i]] else 0.0 for i in range(n_comp)],
                                device=device, dtype=torch.float32)
            out["len_gate"] = gate
        except Exception as e:
            # fall back to ungated (= method #1): gate all ones
            out["len_gate"] = torch.ones(n_comp, device=device, dtype=torch.float32)
            if not self._gate_ok_logged:
                print(f"[gtpo_ema_lenpen_gated] WARN gate disabled (fallback to ungated): {e!r}", flush=True)
                self._gate_ok_logged = True
        return out

    def _length_penalty(self, completion_mask):
        lengths = completion_mask.sum(dim=1)
        return (self.alpha_len * (lengths - self.length_L).clamp(min=0.0)).unsqueeze(1)  # (B,1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        completion_ids  = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_advantages  = inputs["advantages"]
        input_ids      = torch.cat([inputs["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        confidence = confidence_from_model_chunked(
            model, input_ids, attention_mask, logits_to_keep, top_k=self.top_k,
            pass_logits_to_keep=("logits_to_keep" in self.model_kwarg_keys),
            micro_bs=self.conf_micro_bs)

        token_advantages = compute_gtpo_ema_flipped_advantages(
            rewards=seq_advantages, confidence=confidence, completion_mask=completion_mask,
            alpha1=self.alpha1, alpha2=self.alpha2, lam=self.lam,
            reward_threshold=self.reward_threshold)
        if self.format_tag_patterns:
            tag_mask = build_tag_mask(completion_ids, self.format_tag_patterns)
            token_advantages = apply_tag_mask_to_token_advantages(
                token_advantages, seq_advantages, tag_mask)

        # gated Lagrange length penalty
        pen = self._length_penalty(completion_mask)                       # (B,1)
        gate = inputs.get("len_gate")
        B = completion_mask.shape[0]
        if gate is None or gate.shape[0] != B:
            gate = torch.ones(B, device=completion_mask.device, dtype=token_advantages.dtype)
            gate_present = 0.0
        else:
            gate = gate.to(token_advantages.dtype); gate_present = 1.0
        token_advantages = token_advantages - pen * gate.unsqueeze(1) * completion_mask

        # ── metrics ──
        mode = "train" if model.training else "eval"
        tot = completion_mask.sum().clamp(min=1.0)
        ema = compute_ema_vectorized(confidence, completion_mask, lam=self.lam)
        self._metrics[mode].setdefault("gtpo_ema_lenpen_gated/mean_ema", []).append(
            self.accelerator.gather((ema * completion_mask).sum() / tot).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen_gated/mean_len", []).append(
            self.accelerator.gather(completion_mask.sum(dim=1).float().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen_gated/gate_frac", []).append(
            self.accelerator.gather(gate.mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_lenpen_gated/gate_present", []).append(gate_present)
        self._metrics[mode].setdefault("gtpo_ema_lenpen_gated/mean_len_penalty_applied", []).append(
            self.accelerator.gather((pen.squeeze(1) * gate).mean()).mean().item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
