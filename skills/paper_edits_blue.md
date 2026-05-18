# Skill: Mark Paper Edits in Blue for Professor Review

**Use this skill EVERY time you modify any LaTeX paper in this repo.** The professor reviews the PDF
and needs to see at a glance what changed since the previous version. Any text Claude introduces
must be wrapped so it renders in **blue**.

## When this applies

- Every paper under `papers/` — SAGE, FinAgent-Bench, future submissions.
- Both the working source (`*.tex` in the repo) and the Overleaf-synced copy.
- All kinds of edits: reviewer-comment responses, polish, formatting changes, new sections,
  added citations.
- Applies until the user explicitly says "remove the blue markup" / "prepare for final submission"
  / equivalent.

## How to mark edits

### One-time preamble setup (per paper)

In the preamble, immediately before the paper's `\newcommand` block, add:

```latex
\usepackage{xcolor}
% \edit{...} marks Claude's changes in blue for the professor's review pass.
% Strip via the script in skills/strip_edit_markup.sh before final submission.
\newcommand{\edit}[1]{\textcolor{blue}{#1}}
```

If `xcolor` (or `color`) is already loaded, skip the `\usepackage` line; ACL's `acl.sty` does NOT
load `xcolor` itself.

### Wrap every change

- **New text**: wrap in `\edit{...}`.

  ```latex
  We introduce \datasetname \edit{and motivate the design choice of two complementary datasets}.
  ```

- **Replaced text**: wrap the new wording in `\edit{...}`. Do not strike through the old text;
  it's already gone from the diff — the blue markup tells the professor "this is the new version
  of something".

- **Reordered bullets / list items**: wrap each moved item in `\edit{...}` so the professor can
  see which entries are new in this position.

- **Multi-paragraph edits**: `\textcolor` cannot span a blank line. For each paragraph use a
  separate `\edit{...}`. Inside an `itemize`/`description` environment, wrap each `\item`'s body
  separately:

  ```latex
  \begin{itemize}
      \item \edit{\textbf{\synthsubset}: A synthetic dataset of natural-language financial
      questions over relational databases...}
      \item \edit{\textbf{\groundedfinqa}: A grounded dataset derived from FinQA. Each
      \emph{sample} is a self-contained task instance...}
  \end{itemize}
  ```

- **Section/subsection titles**: wrap only the changed portion if the heading is partially new,
  otherwise wrap the whole title.

  ```latex
  \subsection{\synthsubset: Synthetic \edit{Database-Query} Dataset}
  ```

- **Table cells / captions / figure labels**: same rule — wrap the changed text in `\edit{...}`.

- **`\edit` is safe inside**: `\textbf`, `\emph`, `\texttt`, `\citep`, `\cite`, math mode (with
  caveats — prefer `\edit{}` outside `$...$`), captions, table cells.
- **`\edit` is NOT safe across**: blank lines (use one `\edit{}` per paragraph),
  `\section`/`\paragraph` boundaries, verbatim environments.

## Bibliography edits

Bib entries don't render in the body — to flag a new citation, wrap the in-text `\citep{key}` or
`\citet{key}` in `\edit{...}`:

```latex
prior work \edit{\citep{newpaper2026}}
```

## Removing the markup before final submission

When the user says "prepare for final submission" / "remove the blue" / "clean up for camera-ready",
run:

```bash
bash /mnt/data/papers/aiim_research/skills/strip_edit_markup.sh <paper.tex>
```

The script:
1. Removes `\edit{...}` wrapping while preserving inner content (handles nested braces).
2. Removes the `\newcommand{\edit}{...}` line.
3. Removes the `\usepackage{xcolor}` line if it was added by us (only if no other `\textcolor` use).
4. Diffs before/after for sanity.

If the script isn't there yet, do it manually with `sed` — but verify with `git diff` and a recompile.

## Retroactive marking

If asked to "mark my recent changes in blue" and the changes are already committed without
markup, identify them via `git diff <prev-commit>..HEAD -- <paper.tex>` and wrap each added
hunk's new lines in `\edit{...}`.

## Why this rule exists

The professor reads the compiled PDF, not the diff. Blue text gives them a visual signal of
exactly what shifted since the last version — drastically reducing review friction and the
risk of important changes being missed. This is a permanent collaboration convention, not a
per-paper toggle.
