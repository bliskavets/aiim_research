"""
plot_progress.py — parse a training log and generate a quick dashboard.

Usage:
  python skills/plot_progress.py experiments/exp_NNN_name/train.log
  python skills/plot_progress.py experiments/exp_NNN_name/train.log --output figures/interim.png
"""
import argparse, re, json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

METRIC_KEYS = [
    "reward", "kl",
    "rewards/reward_format_exact/mean",
    "rewards/reward_answer_numeric/mean",
    "rewards/reward_answer_exact/mean",
    "completions/mean_length",
    "completions/clipped_ratio",
    "grad_norm",
]


def parse_log(log_path):
    records = []
    with open(log_path) as f:
        for line in f:
            if "'reward'" not in line:
                continue
            # extract all key: value pairs
            rec = {}
            for m in re.finditer(r"'([\w/]+)':\s*([\d.\-e]+)", line):
                try:
                    rec[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
            if "reward" in rec:
                records.append(rec)
    return records


def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def plot_dashboard(records, output, title=""):
    if not records:
        print("No records found.")
        return

    steps = [r.get("step", i) for i, r in enumerate(records)]
    present = [k for k in METRIC_KEYS if any(k in r for r in records)]

    n = len(present)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 4 * rows))
    if title:
        fig.suptitle(title, fontsize=12)
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for i, key in enumerate(present):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        vals = [r.get(key) for r in records]
        ax.plot(steps, vals, linewidth=1.5)
        ax.set_title(key.split("/")[-1])
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        # annotate final value
        final = [v for v in vals if v is not None]
        if final:
            ax.annotate(f"{final[-1]:.4f}", xy=(steps[-1], final[-1]),
                        fontsize=8, ha="right", color="red")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()

    # print summary
    if records:
        last = records[-1]
        print(f"\nStep {int(last.get('step', len(records)))}: "
              f"reward={last.get('reward', 'N/A'):.3f}  "
              f"format={last.get('rewards/reward_format_exact/mean', 'N/A')}  "
              f"KL={last.get('kl', 'N/A'):.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to .log or metrics.json file")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"File not found: {p}")
        return

    if p.suffix == ".json":
        records = parse_json(p)
    else:
        records = parse_log(p)

    output = args.output or str(p.parent / "figures" / "interim_progress.png")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    plot_dashboard(records, output, title=p.parent.name)


if __name__ == "__main__":
    main()
