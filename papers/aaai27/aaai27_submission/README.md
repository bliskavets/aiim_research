# SAGE — AAAI-2027 Anonymous Submission

LaTeX project ported from the EMNLP submission (`final_submission_fonts_corrected.tex`)
into the AAAI-2027 author kit format.

## Files

- `main.tex` — single main source file (compile this in Overleaf; pdfLaTeX)
- `custom.bib` — bibliography (fixed a missing comma in `huang2023affective`)
- `aaai2027.sty` / `aaai2027.bst` — unmodified AAAI-2027 style files
- `figures/` — only the figures actually used by the paper
- `ReproducibilityChecklist.tex` — from the author kit; NOT included yet
  (uncomment the `\input` near the end of `main.tex` if AAAI-27 requires it inline)
- `main.pdf`, `main.bbl`, `main.aux` — build artifacts (AAAI asks for .bbl/.aux in the final archive)

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Changes made for AAAI compliance (desk-reject prevention)

1. Preamble replaced with the AAAI-2027 template preamble (`[submission]` mode,
   letterpaper, natbib, no hyperref).
2. Removed forbidden/unneeded packages: `times`, `inconsolata`, `microtype`,
   `latexsym`, `fontenc`/`inputenc` (loaded by aaai2027.sty), `xcolor`,
   `subcaption`, `multirow`, `amsthm` (all unused in the body), `tikz`/`pgfplots`
   (forbidden by AAAI).
3. The inline pgfplots bar chart (MATH-500 headline results) was pre-rendered
   externally into `figures/headline_math500.pdf` and is now included via
   `\includegraphics`, as AAAI requires. Source of that figure: `figures_src/`
   is not shipped; regenerate from the old EMNLP tex if numbers change.
4. All figure PDFs had CID/Identity-H (and one Type 3) fonts — forbidden by AAAI.
   All fonts in all figures were converted to vector outlines with Ghostscript
   (`-dNoOutputFonts`); the final PDF embeds only Type 1 fonts.
5. Removed `height=0.2\textheight` from one includegraphics (forbidden `\textheight`).
6. Section numbering enabled (`secnumdepth=2`) because the paper cross-references
   sections; appendix sections are lettered automatically.
7. Title changed to Chicago Title Case: "Gradient-based" → "Gradient-Based".
8. Author = "Anonymous Submission", empty affiliations (AAAI anonymous format).
9. Added AAAI `links` environment with the anonymous code URL after the abstract.
10. Replaced U+2011 non-breaking hyphens (91×) with ASCII hyphens; stripped all
    commented-out legacy blocks (they contained forbidden commands like
    `\resizebox`, `\vskip -`, `pgfplots` that automated checks may flag).
11. Fixed overfull hboxes in the appendix verbatim prompt blocks.

## Verified

- Compiles cleanly with pdfLaTeX + BibTeX (TeX Live), 0 errors, 0 overfull boxes,
  no undefined references/citations.
- US letter, two-column AAAI layout; 16 pages total: main content fills
  pages 1-8 (references start in the second half of p. 8), technical appendix follows.
- `pdffonts`: all fonts Type 1, all embedded, no Type 3, no Identity-H.

## Revision state (2026-07-17)

Appendix material promoted to the main text to bring content close to the
8-page limit (ends on p. 8, does not exceed it). All moved/edited passages are
marked BLUE (`\color{blue}` / `\textcolor{blue}`) for review:
AIME 2026 reasoning-mode results, 1.7B/32B model scales, IFEval + MMLU-Pro STEM
(now subsections "Reasoning mode and model scale" and "Instruction following and
broader STEM reasoning" with Tables 3-5). The IFEval table was reformatted to
single column. Temperature and prompt-calibration ablations stayed in the appendix.
Added (blue): a "Discussion" paragraph closing the Experiments section and an
unnumbered "Ethical Statement" before References. Main content now ends exactly
at the bottom of p. 8; References start on p. 9. Note: everything before
References (incl. Limitations and Ethical Statement) counts toward the content
page limit per the AAAI kit.
**Remove the blue markup before submitting** (AAAI forbids colored text).

## Revision 2 (2026-07-18): campaign results woven in (GREEN)

Safe (non-contradicting) results from the July-2026 experiment campaign are
integrated in GREEN (`\textcolor{green!50!black}`): oracle/judge-vs-gold analysis,
modern-RM Best-of-N failure on MATH, aspect/m_min robustness, SPO same-model
optimizer failure, AlpacaEval head-to-head + length check, XSTest per-category
deltas, inference cost note, thinking-mode positioning, and the missing
citations (Self-Refine, Reflexion, Jiang/Wataoka/Pan/Tian). Absolute numbers
that clash with the paper's current tables were deliberately NOT added (see
rebuttal/results_summary_aaai2027.md for the contradicting list). To fit the
8-page content limit the filler "Discussion" paragraph and the "Ethical
Statement" were removed and redundant prose was tightened. Blue = moved from
appendix (rev 1), green = new campaign results (rev 2); remove all color
before submission.

## Before submitting — manual checklist

- [ ] Check AAAI-27 page limit and whether the technical appendix may stay in the
      same PDF after the references (current structure) or must be separate.
- [ ] Check whether the Reproducibility Checklist must be appended (see comment
      at the end of `main.tex`).
- [ ] Clear PDF metadata before uploading (anonymity requirement).
- [ ] For camera-ready: rename `main.tex` to the first author's last name, switch
      `[submission]` off, add real authors/acknowledgments.
