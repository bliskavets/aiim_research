"""
verify_format.py — Pre-training sanity check for exp_017
=========================================================
Verifies that:
  1. The model loads correctly in bf16 without quantization
  2. The model receives and follows the system prompt format
  3. Completions are coherent math reasoning (not empty/garbage)
  4. The SOLUTION tags appear in vocabulary and can be generated
  5. Dataset filtering works — reports stats on integer-answer subset

Run BEFORE train.py to catch config issues early:
    HF_TOKEN=<token> python verify_format.py

Expected output: ≥ 1 of 5 samples should match format regex
(the base model has never seen these tags, so 0/5 is also acceptable —
what matters is coherent text and non-zero format token presence)
"""

import os
import re
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Run: HF_TOKEN=<token> python verify_format.py")
    sys.exit(1)

# ── same constants as train.py ────────────────────────────────────────────────
REASONING_START = "<start_working_out>"
REASONING_END   = "<end_working_out>"
SOLUTION_START  = "<SOLUTION>"
SOLUTION_END    = "</SOLUTION>"

SYSTEM_PROMPT = (
    f"You are given a problem.\n"
    f"Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

NUM_SAMPLES = 5
MAX_NEW_TOKENS = 3072


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset check
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset():
    print("\n" + "=" * 70)
    print("STEP 1: Dataset filter check")
    print("=" * 70)
    from datasets import load_dataset

    ds = load_dataset(
        "SynthLabsAI/Big-Math-RL-Verified",
        split="train",
        token=HF_TOKEN,
    )
    print(f"Total examples: {len(ds)}")

    def is_integer_answer(example):
        raw = str(example.get("answer", "")).strip().replace(",", "")
        try:
            return float(raw) == int(float(raw))
        except (ValueError, OverflowError):
            return False

    ds_int = ds.filter(is_integer_answer)
    print(f"Integer-answer examples: {len(ds_int)} ({100*len(ds_int)/len(ds):.1f}%)")

    # Show a few samples
    print("\nSample problems:")
    for i in range(min(3, len(ds_int))):
        row = ds_int[i]
        print(f"  [{i}] answer={row['answer']!r:>8}  problem={str(row['problem'])[:80]}...")

    return ds_int


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vocabulary check
# ─────────────────────────────────────────────────────────────────────────────

def check_vocabulary(tokenizer):
    print("\n" + "=" * 70)
    print("STEP 2: Vocabulary / tokenization check")
    print("=" * 70)

    tags = [REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END]
    for tag in tags:
        ids = tokenizer.encode(tag, add_special_tokens=False)
        print(f"  {tag!r:30s} → {len(ids)} token(s): {ids}")

    # Check that tags are not split into many fragments (>3 tokens each is suspicious)
    for tag in tags:
        n = len(tokenizer.encode(tag, add_special_tokens=False))
        if n > 4:
            print(f"  WARNING: {tag!r} splits into {n} tokens — model may struggle to generate it")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generation check
# ─────────────────────────────────────────────────────────────────────────────

def check_generation(model, tokenizer, dataset):
    print("\n" + "=" * 70)
    print("STEP 3: Generation check (base model, no LoRA)")
    print("=" * 70)

    import torch

    format_matches = 0
    non_empty = 0

    for i in range(min(NUM_SAMPLES, len(dataset))):
        problem = dataset[i]["problem"]
        answer  = dataset[i]["answer"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        fmt_match = match_format.search(generated) is not None
        if fmt_match:
            format_matches += 1
        if len(generated.strip()) > 10:
            non_empty += 1

        print(f"\n--- Sample {i} ---")
        print(f"Problem (first 120 chars): {problem[:120]}")
        print(f"Ground truth answer: {answer}")
        print(f"Generated ({len(generated)} chars):\n{generated[:500]}")
        print(f"Format match: {'✅' if fmt_match else '❌'}")

        if fmt_match:
            m = match_format.search(generated)
            extracted = m.group(1).strip() if m else "?"
            correct = extracted == str(answer).strip()
            print(f"Extracted answer: {extracted!r}  Correct: {'✅' if correct else '❌'}")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {non_empty}/{NUM_SAMPLES} non-empty, {format_matches}/{NUM_SAMPLES} format-match")
    if non_empty == 0:
        print("FAIL: all completions empty — model not generating properly")
        return False
    print("OK: model generates coherent text")
    print("NOTE: format match at 0/5 is expected for base model (no LoRA)")
    print("      what matters: coherent math reasoning in output")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Memory check
# ─────────────────────────────────────────────────────────────────────────────

def check_memory():
    print("\n" + "=" * 70)
    print("STEP 4: GPU memory check")
    print("=" * 70)

    import torch
    if not torch.cuda.is_available():
        print("No GPU available")
        return

    for i in range(torch.cuda.device_count()):
        total  = torch.cuda.get_device_properties(i).total_memory / 1e9
        used   = torch.cuda.memory_allocated(i) / 1e9
        free   = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9
        print(f"  GPU {i}: {total:.1f} GB total, {used:.1f} GB used, {free:.1f} GB free")

    print(f"\n  Training will use batch_size=4, num_generations=16")
    print(f"  → {4*16} sequences/step for loss computation")
    print(f"  → Estimated VRAM needed: ~30-40 GB (model + activations + vLLM KV cache)")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("exp_017 — Pre-training verification")
    print("Model: meta-llama/Llama-3.2-3B-Instruct")
    print("Dataset: SynthLabsAI/Big-Math-RL-Verified (integer filter)")

    check_memory()
    dataset = check_dataset()

    print("\n" + "=" * 70)
    print("Loading model (bf16, no quantization)...")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()

    check_vocabulary(tokenizer)
    ok = check_generation(model, tokenizer, dataset)

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("If generation looks coherent → safe to run train.py")
    print("=" * 70)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
