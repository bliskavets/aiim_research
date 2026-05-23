"""
Per-category radar chart for FinOpsBench-v1 with **5 axes** — one per
category from §3.2 of the paper:
  1. Accounts Payable (AP) analysis
  2. Revenue recognition
  3. Variance analysis
  4. Data integrity and reconciliation
  5. Financial reporting

The original 4-axis script (make_radar.py) used the four seed
categories from seed-queries.jsonl. That file does not separate
"Variance analysis" from "Financial reporting" (they collapse into a
single "Financial Analysis" seed bucket). To get a 5-way split we
use short anchor descriptions for each category and assign every
test query to its highest-cosine anchor via TF-IDF.

Outputs:
  fig_synth_categories_radar_5cat.png
  fig_synth_categories_radar_5cat.pdf
  category_distribution_5cat.json  (used for the appendix table)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

DATA = Path("/tmp/tta_eval/exp/expand-10k")
OUT = Path("/mnt/data/papers/aiim_research/papers/emnlp2026/figures")

EVAL_FILES = {
    "GPT-5":         "eval_sample_evaluated_gpt_5.jsonl",
    "GPT-5-mini":    "eval_sample_gpt-5-mini.jsonl",
    "o4-mini":       "eval_sample_evaluated_o4_mini.jsonl",
    "GPT-4.1":       "eval_sample_evaluated_gpt_4.1.jsonl",
    "GPT-4.1-mini":  "eval_sample_evaluated_gpt-4.1_mini.jsonl",
    "Qwen3-30B-A3B": "eval_sample_qwen3-30b-a3b.jsonl",
    "Qwen3-8B":      "eval_sample_qwen3-8b.jsonl",
    "Llama-3.1-8B":  "eval_sample_llama-3.1-8b.jsonl",
}

CATEGORY_ANCHORS = [
    ("Accounts Payable (AP) analysis",
     "AP analysis",
     "accounts payable invoices vendor supplier payments overdue accruals "
     "aging reconciliation late payment approvals duplicates"),
    ("Revenue recognition",
     "Revenue\nrecognition",
     "revenue recognition deferred sales contracts performance obligations "
     "subscription income earned billing"),
    ("Variance analysis",
     "Variance\nanalysis",
     "variance analysis budget actual forecast quarter over quarter year "
     "over year comparison drivers fluctuation"),
    ("Data integrity and reconciliation",
     "Data integrity\n& reconciliation",
     "reconciliation ledger balance integrity discrepancy mismatch missing "
     "duplicate journal entries totals check audit"),
    ("Financial reporting",
     "Financial\nreporting",
     "financial reporting balance sheet income statement cash flow disclosure "
     "kpis dashboard compliance gaap report consolidated"),
]


# ---------------------------------------------------------------------------
# Step 1: load test set.
# ---------------------------------------------------------------------------

with (DATA / "eval-queries.txt").open() as f:
    eval_qs = {line.strip() for line in f if line.strip()}

valid_qs = set()
with (DATA / "11_check_no_tools.jsonl").open() as f:
    for line in f:
        valid_qs.add(json.loads(line)["query"])

test_qs = eval_qs & valid_qs
print(f"Test set: {len(test_qs)} queries")


# ---------------------------------------------------------------------------
# Step 2: TF-IDF categorise.
# ---------------------------------------------------------------------------

TOK_RE = re.compile(r"[a-z0-9]{2,}")


def tokenize(text: str) -> list[str]:
    return TOK_RE.findall(text.lower())


docs = [tokenize(a) for _, _, a in CATEGORY_ANCHORS] + \
       [tokenize(q) for q in valid_qs]
df: Counter[str] = Counter()
for d in docs:
    for w in set(d):
        df[w] += 1
N = len(docs)
idf = {w: math.log((N + 1) / (df_w + 1)) + 1 for w, df_w in df.items()}


def tfidf_vec(toks: list[str]) -> dict[str, float]:
    tf = Counter(toks)
    if not tf:
        return {}
    vec = {w: c * idf.get(w, 0.0) for w, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {w: v / norm for w, v in vec.items()}


def cos(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(w, 0.0) for w, v in a.items())


anchor_vecs = [(cat, tfidf_vec(tokenize(anchor)))
               for cat, _, anchor in CATEGORY_ANCHORS]


def categorise(query: str) -> str:
    qv = tfidf_vec(tokenize(query))
    best_cat, best_sim = None, -1.0
    for cat, av in anchor_vecs:
        sim = cos(qv, av)
        if sim > best_sim:
            best_sim, best_cat = sim, cat
    return best_cat


# Categorise BOTH the test set (for the radar) and the full v1 stream
# (for the appendix distribution table).
cat_of_test: dict[str, str] = {q: categorise(q) for q in test_qs}
cat_of_full: dict[str, str] = {q: categorise(q) for q in valid_qs}

print("\nCategory distribution across the test set:")
for cat, _, _ in CATEGORY_ANCHORS:
    n = sum(1 for v in cat_of_test.values() if v == cat)
    print(f"  {cat:<40} {n:>4}  ({100 * n / len(test_qs):>5.1f}%)")

print("\nCategory distribution across the full v1 stream:")
full_pct = {}
for cat, _, _ in CATEGORY_ANCHORS:
    n = sum(1 for v in cat_of_full.values() if v == cat)
    pct = 100 * n / len(valid_qs)
    full_pct[cat] = pct
    print(f"  {cat:<40} {n:>4}  ({pct:>5.1f}%)")

with (OUT / "category_distribution_5cat.json").open("w") as f:
    json.dump({"per_category_percent": full_pct,
               "n_examples": len(valid_qs)}, f, indent=2)


# ---------------------------------------------------------------------------
# Step 3: per-(model, category) accuracy.
# ---------------------------------------------------------------------------

def is_correct(item: dict) -> bool | None:
    ev = item.get("evaluation")
    if ev is not None:
        return ev.get("correct")
    ome = item.get("output_matches_expected")
    if ome is not None:
        return ome.get("is_correct")
    return None


per_model_cat = {}
for model, fn in EVAL_FILES.items():
    counts = defaultdict(lambda: [0, 0])
    answered = set()
    with (DATA / fn).open() as f:
        for line in f:
            item = json.loads(line)
            q = item["query"]
            if q not in test_qs:
                continue
            answered.add(q)
            cat = cat_of_test[q]
            counts[cat][1] += 1
            if is_correct(item):
                counts[cat][0] += 1
    for q in test_qs - answered:
        counts[cat_of_test[q]][1] += 1
    per_model_cat[model] = {cat: (c, t) for cat, (c, t) in counts.items()}


# ---------------------------------------------------------------------------
# Step 4: radar.
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE = {
    "GPT-5":         "#1f3a93",
    "GPT-5-mini":    "#3b6fb0",
    "o4-mini":       "#5fa8d3",
    "GPT-4.1":       "#7fb7c4",
    "GPT-4.1-mini":  "#a4cab1",
    "Qwen3-30B-A3B": "#d97a3f",
    "Qwen3-8B":      "#b1551c",
    "Llama-3.1-8B":  "#7a2a0e",
}

cats = [c for c, _, _ in CATEGORY_ANCHORS]
labels = [lbl for _, lbl, _ in CATEGORY_ANCHORS]
n = len(cats)
angles = [i / n * 2 * math.pi for i in range(n)] + [0.0]

fig, ax = plt.subplots(figsize=(8.5, 6.5), subplot_kw={"projection": "polar"})

for model in EVAL_FILES:
    counts = per_model_cat[model]
    values = [100 * counts[c][0] / counts[c][1] if counts[c][1] else 0
              for c in cats]
    values_closed = values + [values[0]]
    ax.plot(angles, values_closed,
            color=PALETTE[model], linewidth=1.8, label=model, zorder=3)
    ax.fill(angles, values_closed, color=PALETTE[model], alpha=0.08, zorder=2)
    ax.scatter(angles[:-1], values, color=PALETTE[model], s=20, zorder=4)

ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80])
ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="#666")
ax.set_rlabel_position(225)
ax.grid(color="#cccccc", linewidth=0.6)
ax.spines["polar"].set_color("#999999")

ax.set_title("Per-category accuracy on FinOpsBench-v1 (% correct)",
             y=1.10)
ax.legend(loc="center left", bbox_to_anchor=(1.10, 0.5), frameon=False)

fig.savefig(OUT / "fig_synth_categories_radar_5cat.png")
fig.savefig(OUT / "fig_synth_categories_radar_5cat.pdf")
plt.close(fig)
print(f"\nSaved {OUT / 'fig_synth_categories_radar_5cat.png'}")
print(f"Saved {OUT / 'category_distribution_5cat.json'}")
