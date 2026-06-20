"""
gtpo_ema_adaptlen_gated_trainer.py
----------------------------------
exp_058 method #4: the ADAPTIVE length penalty of method #3 (knee L from each
group's own length distribution), GATED by the same low-temperature difficulty
heuristic as gtpo_ema_lenpen_gated.

Per generation batch we sample 2 EXTRA completions per UNIQUE prompt — greedy
(t=0) and t2 (default 0.5) — NOT used in the update. If EITHER gives the exact
boxed answer within gate_max_tokens, the problem is concisely solvable, so we
apply the adaptive penalty to that prompt's training completions; otherwise we
skip it (the problem may genuinely need long exploration).

    pen_rel_i  = adaptive_length_penalty(group lengths)   # group-centered, [-0.5..0.5]
    len_pen_i  = pen_rel_i * gate(prompt_i)               # gate ∈ {0,1} per prompt
    Ã_{i,t}   <- Ã_{i,t} - len_pen_i                      (applied in compute_loss)

Robust fallback: on any gate failure the penalty stays ungated (= method #3),
flagged via a metric.
"""
import torch
from trl import GRPOTrainer
from .ema_flipped_utils import (
    confidence_from_model_chunked, compute_ema_vectorized,
    compute_gtpo_ema_flipped_advantages, EPS,
)
from .format_tag_mask import build_tag_mask, apply_tag_mask_to_token_advantages
from .shaped_loss import inject_advantages
from .adaptive_lenpen_utils import adaptive_length_penalty


def _answers_equal(guess, gold):
    if guess is None:
        return False
    try:
        return float(str(guess).strip().replace(",", "")) == float(str(gold).strip().replace(",", ""))
    except (ValueError, TypeError):
        return False


class GTPOEMAFlippedAdaptLenGatedTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        alpha1               = kwargs.pop("alpha1", 0.9)
        alpha2               = kwargs.pop("alpha2", 0.1)
        lam                  = kwargs.pop("lam", 0.9)
        top_k                = kwargs.pop("top_k", 20)
        reward_threshold     = kwargs.pop("reward_threshold", 0.0)
        format_tag_patterns  = kwargs.pop("format_tag_patterns", None)
        conf_micro_bs        = kwargs.pop("conf_micro_bs", 2)
        gate_temps           = kwargs.pop("gate_temps", (0.0, 0.5))
        gate_max_tokens      = kwargs.pop("gate_max_tokens", 3584)
        answer_extractor     = kwargs.pop("answer_extractor", None)
        super().__init__(*args, **kwargs)
        self.alpha1, self.alpha2, self.lam, self.top_k = alpha1, alpha2, lam, top_k
        self.reward_threshold = reward_threshold
        self.format_tag_patterns = format_tag_patterns
        self.conf_micro_bs = conf_micro_bs
        self.gate_temps, self.gate_max_tokens = tuple(gate_temps), gate_max_tokens
        self.answer_extractor = answer_extractor
        self._gate_warned = False

    def _compute_gate_per_completion(self, inputs, n_comp, device):
        """Return gate (n_comp,) in {0,1}; gate[i]=1 if a low-temp sample of
        completion i's prompt gave the exact answer."""
        keys = [str(x["prompt"]) for x in inputs]
        uniq = {}
        for i, k in enumerate(keys):
            uniq.setdefault(k, i)
        uniq_keys = list(uniq.keys())
        prompt_texts = [
            self.processing_class.apply_chat_template(
                inputs[uniq[k]]["prompt"], tokenize=False, add_generation_prompt=True)
            for k in uniq_keys
        ]
        answers = [inputs[uniq[k]].get("answer") for k in uniq_keys]
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
                if _answers_equal(guess, answers[j]):
                    solved[j] = True
        if hasattr(self.llm, "sleep"):
            try: self.llm.sleep(level=1)
            except Exception: pass
        gate_by_key = {k: solved[j] for j, k in enumerate(uniq_keys)}
        return torch.tensor([1.0 if gate_by_key[keys[i]] else 0.0 for i in range(n_comp)],
                            device=device, dtype=torch.float32)

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        cm = out["completion_mask"]
        n_comp = cm.shape[0]
        device = cm.device
        # adaptive group-relative length penalty
        lengths = cm.sum(dim=1).float()
        pen_rel, L_per = adaptive_length_penalty(lengths, self.num_generations)
        # low-temperature gate
        try:
            assert len(inputs) == n_comp
            gate = self._compute_gate_per_completion(inputs, n_comp, device)
        except Exception as e:
            gate = torch.ones(n_comp, device=device, dtype=torch.float32)
            if not self._gate_warned:
                print(f"[gtpo_ema_adaptlen_gated] WARN gate disabled (ungated fallback): {e!r}", flush=True)
                self._gate_warned = True
        out["len_pen"] = pen_rel * gate
        out["adapt_L"] = L_per
        out["gate_dbg"] = gate
        return out

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

        B = completion_mask.shape[0]
        pen_rel = inputs.get("len_pen")
        gate = inputs.get("gate_dbg")
        if pen_rel is None or pen_rel.shape[0] != B:
            pen_rel = torch.zeros(B, device=completion_mask.device, dtype=token_advantages.dtype)
            pen_present = 0.0
        else:
            pen_rel = pen_rel.to(token_advantages.dtype); pen_present = 1.0
        token_advantages = token_advantages - pen_rel.unsqueeze(1) * completion_mask

        mode = "train" if model.training else "eval"
        self._metrics[mode].setdefault("gtpo_ema_adaptlen_gated/mean_len", []).append(
            self.accelerator.gather(completion_mask.sum(dim=1).float().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_adaptlen_gated/pen_rel_absmean", []).append(
            self.accelerator.gather(pen_rel.abs().mean()).mean().item())
        self._metrics[mode].setdefault("gtpo_ema_adaptlen_gated/pen_present", []).append(pen_present)
        adapt_L = inputs.get("adapt_L")
        if adapt_L is not None and adapt_L.shape[0] == B:
            self._metrics[mode].setdefault("gtpo_ema_adaptlen_gated/mean_L", []).append(
                self.accelerator.gather(adapt_L.to(token_advantages.dtype).mean()).mean().item())
        if gate is not None and gate.shape[0] == B:
            self._metrics[mode].setdefault("gtpo_ema_adaptlen_gated/gate_frac", []).append(
                self.accelerator.gather(gate.to(token_advantages.dtype).mean()).mean().item())

        inputs = inject_advantages(inputs, token_advantages, logits_to_keep)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
