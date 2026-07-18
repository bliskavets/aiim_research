# ICLR 2027 Submission — PACT: Polarity-Aware Confidence-Shaped Token-Level Credit Assignment

Initial draft generated from the experimental record (exp_001–exp_083) and `analysis/*.md`.

## Files
- `main.tex` — the paper (ICLR 2026 style, as provided in `papers/iclr27/iclr2026.zip`; swap in the ICLR 2027 style file when released)
- `references.bib` — bibliography
- `main.pdf` — compiled draft (11 pages incl. appendix)
- `figures/` — placeholder figures copied from experiments (not yet wired into the tex)

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## What the draft claims (all backed by existing runs)
1. **Composability**: one fixed config (k=5, λ=0.7, α=0.9/0.1, τ=1024, zero-var gate) improves peak reward for GRPO/DAPO/Dr.GRPO on GSM8K/MATH-500/Big-Math (9/9 cells) and reaches baseline peaks 2–5× faster (exp_076–079, `analysis/baseline_tables_qwen3-4b-base.md`).
2. **Stability theory**: C_k ≥ log k boundedness lemma; length-invariant budget; zero-variance gate — validated by the collapse taxonomy (exp_066/068/069/070/065/058).
3. **Diagnostics**: distribution bimodality (exp_067), positional decisiveness (exp_064), surprisal ablation (exp_074).
4. **Honest boundaries**: GSPO absorbs the signal (exp_078), Omni-MATH unchanged (gated), Llama cold-start polarity failure (exp_080–083), implementation hazards appendix (exp_057/058 bugs).

## Known gaps to close before submission (priority order)
1. **Held-out eval accuracy** (currently training reward only) — biggest reviewer risk.
2. **Multi-seed** (currently seed 3407 only) — need ≥3 seeds on the main table.
3. **GTPO entropy-weighted baseline in our codebase** (the closest prior work is cited but not run head-to-head under identical conditions).
4. **Correctness-grounded polarity (exp_084)** — would convert the Llama negative result into a fixed-and-validated section.
5. Figures: training curves for Table 1, bimodality histogram, O+/O− position profiles, collapse gallery.
6. Full-finetune or ≥7B run if budget allows.
