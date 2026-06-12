"""
diagnose_real.py — REAL-model audit of the gtpo_ema_flipped shaping.

Loads Qwen3-4B exactly like train.py, runs a handful of GRPO steps with the
GTPO-EMA-flipped trainer, and inside _compute_loss captures REAL numbers:
  - tag-mask coverage on real completion_ids
  - per-token confidence std across the batch (real logits)
  - shaped token_advantages vs the GRPO seq-advantage broadcast on the SAME
    batch: correlation, ||shaped - seq||/||seq|| (the "addition" norm relative
    to the base reward signal), per-polarity means
  - activation norm: ||per_token_logps|| and hidden logp std
  - grad norm is logged by TRL itself (compare in the printed metric dicts)

One model load. ~6 steps. Run with the GPU free.
"""
import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

# load train.py as a module to reuse its config + dataset + rewards
spec = importlib.util.spec_from_file_location("t057", os.path.join(HERE, "train.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

from unsloth import FastLanguageModel
from trl import GRPOConfig
from src.gtpo_ema_flipped_trainer import GTPOEMAFlippedTrainer
from src.format_tag_mask import build_tag_mask, encode_tag_patterns
from src.ema_flipped_utils import compute_gtpo_ema_flipped_advantages

CAP = []  # captured per-step diagnostics


class DiagTrainer(GTPOEMAFlippedTrainer):
    def _compute_loss(self, model, inputs):
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        seq_adv = inputs["advantages"]
        prompt_ids = inputs["prompt_ids"]; prompt_mask = inputs["prompt_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        ltk = completion_ids.size(1)
        if not CAP:
            print(f"[shaping params] alpha1={getattr(self,'alpha1',None)} alpha2={getattr(self,'alpha2',None)} "
                  f"lam={getattr(self,'lam',None)} top_k={getattr(self,'top_k',None)} "
                  f"reward_threshold={getattr(self,'reward_threshold',None)} "
                  f"tag_patterns={'set' if getattr(self,'format_tag_patterns',None) else 'NONE'}", flush=True)
        with torch.no_grad():
            ptl, _ = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, ltk, compute_entropy=False)
            mi = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "logits_to_keep" in self.model_kwarg_keys:
                mi["logits_to_keep"] = ltk + 1
            logits = model(**mi).logits[:, :-1, :][:, -ltk:, :]
            from src.ema_flipped_utils import confidence_from_logits
            tk = getattr(self, "top_k", None) or 20
            conf = confidence_from_logits(logits, top_k=tk)

            tok_adv = compute_gtpo_ema_flipped_advantages(
                seq_adv, conf, completion_mask,
                getattr(self, "alpha1", None) or 0.9, getattr(self, "alpha2", None) or 0.1,
                getattr(self, "lam", None) or 0.9, getattr(self, "reward_threshold", None) or 0.0)
            tagm = build_tag_mask(completion_ids, self.format_tag_patterns)
            seq_bcast = seq_adv.view(-1, 1).expand_as(tok_adv)
            tok_adv_masked = torch.where(tagm, seq_bcast, tok_adv)

            m = completion_mask.bool()
            mv = m & ~tagm  # content (non-tag) valid tokens
            s = tok_adv_masked[m]; g = seq_bcast[m]
            corr = torch.corrcoef(torch.stack([s.float(), g.float()]))[0, 1].item() if s.numel() > 1 else float("nan")
            add_norm = (tok_adv_masked - seq_bcast)[m].norm().item()
            seq_norm = seq_bcast[m].norm().item()
            is_pos = (seq_adv > self.reward_threshold)
            d = {
                "n_seq": int(seq_adv.numel()),
                "frac_pos": is_pos.float().mean().item(),
                "tag_frac": tagm[m].float().mean().item() if m.any() else 0.0,
                "conf_mean": conf[m].mean().item(),
                "conf_std_across_tokens": conf[m].std().item(),
                "seq_adv_std": seq_adv.std().item(),
                "tok_adv_std": s.std().item(),
                "within_seq_tok_std": tok_adv_masked.std(dim=1).mean().item(),
                "corr_tokadv_vs_seq": corr,
                "add_over_seq_norm": add_norm / (seq_norm + 1e-8),
                "ptl_norm": ptl[m].norm().item(),
                "ptl_std": ptl[m].std().item(),
                "mean_tokadv_Opos": tok_adv_masked[is_pos.unsqueeze(1).expand_as(tok_adv) & m].mean().item() if (is_pos.any() and m.any()) else float("nan"),
            }
            CAP.append(d)
            print(f"[diag step {len(CAP)}] frac_pos={d['frac_pos']:.2f} tag_frac={d['tag_frac']:.3f} "
                  f"conf={d['conf_mean']:.2f}±{d['conf_std_across_tokens']:.2f} | "
                  f"seq_adv_std={d['seq_adv_std']:.2f} tok_adv_std={d['tok_adv_std']:.2f} "
                  f"within_seq_std={d['within_seq_tok_std']:.2f} | corr(tokadv,seq)={d['corr_tokadv_vs_seq']:+.3f} "
                  f"||add||/||seq||={d['add_over_seq_norm']:.2f} | meanadv_O+={d['mean_tokadv_Opos']:+.3f} | "
                  f"||logp||={d['ptl_norm']:.1f}", flush=True)
        return super()._compute_loss(model, inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()

    ds = T.prepare_dataset().select(range(128))

    model, tok = FastLanguageModel.from_pretrained(
        model_name=T.MODEL_CONFIG["model_name"],
        max_seq_length=T.MODEL_CONFIG["max_seq_length"],
        load_in_4bit=False, fast_inference=True,
        max_lora_rank=T.MODEL_CONFIG["lora_rank"],
        gpu_memory_utilization=T.MODEL_CONFIG["gpu_memory_utilization"])
    model = FastLanguageModel.get_peft_model(
        model, r=T.LORA_CONFIG["r"], target_modules=T.LORA_CONFIG["target_modules"],
        lora_alpha=T.LORA_CONFIG["lora_alpha"],
        use_gradient_checkpointing="unsloth", random_state=T.SEED)

    cfg = dict(T.TRAINING_CONFIG); cfg["max_steps"] = a.steps
    args = GRPOConfig(max_prompt_length=512,
                      max_completion_length=T.DATASET_CONFIG["max_completion_tokens"],
                      output_dir="/tmp/diag_out", **cfg)
    pats = encode_tag_patterns(tok, ["<think>", "</think>", "<|im_start|>", "<|im_end|>"])
    trainer = DiagTrainer(model=model, tokenizer=tok, args=args, train_dataset=ds,
                          reward_funcs=T.REWARD_FUNCS_FULL,
                          **T.SHAPING_CONFIG["gtpo_ema_flipped"], format_tag_patterns=pats)
    trainer.train()

    import statistics as st
    print("\n==== REAL-DATA SUMMARY (mean over steps) ====")
    for k in CAP[0]:
        print(f"  {k:28s} {st.mean([c[k] for c in CAP]):+.4f}")


if __name__ == "__main__":
    main()
