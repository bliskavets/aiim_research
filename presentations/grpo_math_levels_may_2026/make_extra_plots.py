"""
make_extra_plots.py — two focused comparison plots for the may 2026 deck.

Reads train.log files from experiments and parses 'reward' / 'reward_answer_exact'
out of the TRL-style JSON lines. No json input needed.
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP  = os.path.join(ROOT, "experiments")

R_REW = re.compile(r"'reward':\s*([-\d.]+)")
R_ANS = re.compile(r"'rewards/reward_answer_exact/mean':\s*([-\d.]+)")


def parse_log(name):
    p = os.path.join(EXP, name, "train.log")
    if not os.path.exists(p):
        return [], []
    with open(p, "r", errors="ignore") as f:
        txt = f.read()
    return ([float(m.group(1)) for m in R_REW.finditer(txt)],
            [float(m.group(1)) for m in R_ANS.finditer(txt)])


def smooth(x, w=20):
    if len(x) < w:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0))
    sm = (c[w:] - c[:-w]) / w
    pad = np.full(w - 1, sm[0])
    return np.concatenate([pad, sm])


def two_panel(curves, title, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    for name, color, (rew, ans) in curves:
        if rew:
            a.plot(smooth(rew), color=color, label=name, linewidth=1.6)
        if ans:
            b.plot(smooth(ans), color=color, label=name, linewidth=1.6)
    a.set_title("Total reward (20-step smoothing)")
    a.set_xlabel("step"); a.set_ylabel("reward")
    a.grid(alpha=0.25); a.legend(frameon=False, fontsize=9)
    b.set_title("Answer exact (20-step smoothing)")
    b.set_xlabel("step"); b.set_ylabel("answer_exact")
    b.grid(alpha=0.25); b.legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    # exp_044 vs exp_041 — activated GTPO-EMA-flipped, still collapses
    two_panel(
        [
            ("exp_041 GRPO baseline",       "#4f46e5", parse_log("exp_041_qwen3_math_levels3to5")),
            ("exp_044 GTPO-EMA-flipped",    "#d97706", parse_log("exp_044_qwen3_math_levels3to5_gtpo_activated")),
        ],
        "MATH levels 3-5 (integer)  —  Qwen3-4B  —  GTPO-EMA-flipped properly activated",
        os.path.join(HERE, "img", "exp_044_vs_041.png"),
    )

    # exp_045/046 vs baseline — sequence-level shaping is stable
    two_panel(
        [
            ("exp_041 GRPO baseline",   "#4f46e5", parse_log("exp_041_qwen3_math_levels3to5")),
            ("exp_045 SCRS-Confidence", "#059669", parse_log("exp_045_qwen3_math_levels3to5_scrs")),
            ("exp_046 SCRS-Entropy",    "#d9770a", parse_log("exp_046_qwen3_math_levels3to5_scrs_entropy")),
            ("exp_048 UCAS Stage1",     "#0284c7", parse_log("exp_048_qwen3_math_levels3to5_ucas_stage1")),
        ],
        "MATH levels 3-5 (integer)  —  sequence-level confidence shaping vs GRPO baseline",
        os.path.join(HERE, "img", "seq_level_overview.png"),
    )


if __name__ == "__main__":
    main()
