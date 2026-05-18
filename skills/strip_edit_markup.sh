#!/usr/bin/env bash
# Strip \edit{...} markup before final paper submission.
#
# Usage:
#   bash skills/strip_edit_markup.sh path/to/paper.tex
#
# What it does (in order):
#   1. Removes \edit{...} wrapping while keeping the inner content (handles
#      one level of nested braces correctly).
#   2. Removes the \newcommand{\edit}[1]{\textcolor{blue}{#1}} definition line.
#   3. Removes \usepackage{xcolor} ONLY if no other \textcolor or \color
#      reference remains in the file.
#   4. Shows a unified diff and asks for confirmation before writing.
#
# Safety: edits a .bak copy first; original is replaced only after confirmation.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <paper.tex>" >&2
    exit 2
fi

TEX="$1"
if [[ ! -f "$TEX" ]]; then
    echo "file not found: $TEX" >&2
    exit 1
fi

TMP="${TEX}.stripped.$$"
cp "$TEX" "$TMP"

# Step 1: strip \edit{...} — Perl handles one level of nested braces.
perl -i -0pe 's/\\edit\{((?:[^{}]|\{[^{}]*\})*)\}/$1/g' "$TMP"

# Step 2: remove the \newcommand line.
sed -i '/\\newcommand{\\edit}\[1\]{\\textcolor{blue}{#1}}/d' "$TMP"
sed -i '/^% \\edit{\.\.\.} marks/d' "$TMP"
sed -i '/^% Strip via the script in skills\/strip_edit_markup\.sh/d' "$TMP"

# Step 3: drop \usepackage{xcolor} if no remaining color use.
if ! grep -qE '\\(textcolor|color\b|colorbox|definecolor)' "$TMP"; then
    sed -i '/^\\usepackage{xcolor}/d' "$TMP"
fi

echo "--- diff ($TEX -> stripped) ---"
diff -u "$TEX" "$TMP" || true
echo "--- end diff ---"

read -rp "apply changes to $TEX ? [y/N] " ans
case "$ans" in
    y|Y|yes) mv "$TMP" "$TEX"; echo "stripped $TEX" ;;
    *) rm -f "$TMP"; echo "aborted, no changes" ;;
esac
