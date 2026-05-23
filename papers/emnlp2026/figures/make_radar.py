"""
Per-category radar chart for FinOpsBench-v1.

Inputs (extracted from the collaborator's experiments.zip into
.tta_eval_extracted/):
  - eval-queries.txt              : the 729 test queries
  - 11_check_no_tools.jsonl       : the 5,803 valid dataset queries
  - eval_sample_*_<model>.jsonl   : per-model agent eval outputs
  - seed-queries.jsonl            : the 13 seeds with their categories
                                    (fetched separately from the repo)

Pipeline:
  1. Intersect eval-queries.txt with 11_check_no_tools.jsonl → 729 test set.
  2. Categorise each test query by nearest-seed cosine similarity over
     TF-IDF features computed by hand (no sklearn available).
  3. For each model, compute per-category accuracy.
  4. Render a radar chart in the style of the user's reference image.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("/mnt/data/papers/aiim_research/.tta_eval_extracted")
OUT = Path("/mnt/data/papers/aiim_research/papers/emnlp2026/figures")

# Files we'll consume.
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

# 4 actual seed categories from the repo's seed-queries.jsonl. Names tightened
# for the radar.
CATEGORY_LABELS = {
    "Accounts Payable (AP) Analysis": "AP analysis",
    "Data Integrity & Reconciliation": "Data integrity\n& reconciliation",
    "Financial Analysis": "Variance &\nfinancial analysis",
    "Revenue Recognition Analysis": "Revenue\nrecognition",
}
CATEGORY_ORDER = [
    "Accounts Payable (AP) Analysis",
    "Data Integrity & Reconciliation",
    "Financial Analysis",
    "Revenue Recognition Analysis",
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
assert len(test_qs) == 729, f"expected 729 test queries, got {len(test_qs)}"
print(f"Test set: {len(test_qs)} queries")


# ---------------------------------------------------------------------------
# Step 2: load seeds and categorise test queries.
# ---------------------------------------------------------------------------

seeds = []  # list of (category, text)
with open("/tmp/seed-queries.jsonl") as f:
    for line in f:
        item = json.loads(line)
        seeds.append((item["category"], item["question"]))

# Tokenize: lowercase alphanumeric words, length >= 2.
TOK_RE = re.compile(r"[a-z0-9]{2,}")

def tokenize(text: str) -> list[str]:
    return TOK_RE.findall(text.lower())

# Build IDF over (seeds ∪ test queries).
docs = [tokenize(t) for _, t in seeds] + [tokenize(q) for q in test_qs]
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

seed_vecs = [(cat, tfidf_vec(tokenize(text))) for cat, text in seeds]

def categorise(query: str) -> str:
    qv = tfidf_vec(tokenize(query))
    best_cat, best_sim = None, -1.0
    for cat, sv in seed_vecs:
        sim = cos(qv, sv)
        if sim > best_sim:
            best_sim, best_cat = sim, cat
    return best_cat

cat_of: dict[str, str] = {q: categorise(q) for q in test_qs}
print("Category distribution across the 729-query test set:")
for cat, count in sorted(Counter(cat_of.values()).items(), key=lambda x: -x[1]):
    print(f"  {cat:<40} {count:>4}")


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

per_model_cat = {}  # model -> category -> (correct, total)
for model, fn in EVAL_FILES.items():
    counts = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
    answered = set()
    with (DATA / fn).open() as f:
        for line in f:
            item = json.loads(line)
            q = item["query"]
            if q not in test_qs:
                continue
            answered.add(q)
            cat = cat_of[q]
            counts[cat][1] += 1
            if is_correct(item):
                counts[cat][0] += 1
    # Account for unanswered test queries (Llama has 460 of 729) — count as wrong.
    for q in test_qs - answered:
        counts[cat_of[q]][1] += 1
    per_model_cat[model] = {cat: (c, t) for cat, (c, t) in counts.items()}
    overall_c = sum(c for c, _ in counts.values())
    overall_t = sum(t for _, t in counts.values())
    print(f"{model:<15} overall {100 * overall_c / overall_t:>5.1f}%  "
          + "  ".join(f"{CATEGORY_LABELS[cat].replace(chr(10),' '):<28}: "
                      f"{100 * counts[cat][0] / counts[cat][1]:.1f}% "
                      f"({counts[cat][0]}/{counts[cat][1]})"
                      for cat in CATEGORY_ORDER))


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

# 8 model palette — frontier in cool blues, open-source in warm oranges/greens.
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

cats = CATEGORY_ORDER
labels = [CATEGORY_LABELS[c] for c in cats]
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
    # Point markers
    ax.scatter(angles[:-1], values, color=PALETTE[model], s=20, zorder=4)

# Style the polar plot.
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

ax.legend(loc="center left", bbox_to_anchor=(1.10, 0.5), frameon=False)

(OUT / "fig_synth_categories_radar.png").parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_synth_categories_radar.png")
fig.savefig(OUT / "fig_synth_categories_radar.pdf")
plt.close(fig)
print(f"Saved {OUT / 'fig_synth_categories_radar.png'}")
