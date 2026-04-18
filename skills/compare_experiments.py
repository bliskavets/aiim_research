"""
compare_experiments.py — plot reward/KL curves from multiple experiments on one figure.

Usage:
  python skills/compare_experiments.py --experiments exp_001 exp_006 exp_009b
  python skills/compare_experiments.py --experiments exp_001 exp_006 --metric reward kl
  python skills/compare_experiments.py --all  # compare all experiments with metrics.json
"""
import argparse, json, re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

ROOT = Path(__file__).parent.parent / "experiments"


def load_metrics_json(exp_dir: Path):
    for fname in ["metrics.json", "metrics_full.json", "metrics_summary.json"]:
        p = exp_dir / fname
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list) and data and "step" in data[0]:
                return data
            if isinstance(data, dict):
                # try to convert dict-of-lists to list-of-dicts
                if "step" in data:
                    n = len(data["step"])
                    return [{k: data[k][i] for k in data} for i in range(n)]
    return None


def load_metrics_log(exp_dir: Path):
    logs = list(exp_dir.glob("*.log"))
    if not logs:
        return None
    records = []
    pattern = re.compile(r"'step':\s*(\d+).*?'reward':\s*([\d.\-]+).*?'kl':\s*([\d.\-e]+)")
    for log_path in logs:
        with open(log_path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    records.append({
                        "step": int(m.group(1)),
                        "reward": float(m.group(2)),
                        "kl": float(m.group(3)),
                    })
    return records if records else None


def load_experiment(exp_name: str):
    # find dir by prefix
    matches = [d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith(exp_name)]
    if not matches:
        print(f"  [warn] not found: {exp_name}")
        return None, None
    exp_dir = sorted(matches)[0]
    data = load_metrics_json(exp_dir)
    if data is None:
        data = load_metrics_log(exp_dir)
    if data is None:
        print(f"  [warn] no metrics found in {exp_dir.name}")
    return exp_dir.name, data


def plot(experiments, metrics, output):
    n = len(metrics)
    colors = cm.tab10(np.linspace(0, 1, max(n, 1)))

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, (name, data) in enumerate(experiments):
            if data is None:
                continue
            steps = [r.get("step", j) for j, r in enumerate(data)]
            values = [r.get(metric) for r in data]
            values = [v for v in values if v is not None]
            steps = steps[:len(values)]
            if values:
                ax.plot(steps, values, label=name, color=colors[i], linewidth=1.5)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--metric", nargs="+", default=["reward", "kl"])
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    exp_names = args.experiments
    if args.all:
        exp_names = [d.name for d in sorted(ROOT.iterdir()) if d.is_dir() and d.name.startswith("exp_")]

    loaded = []
    for name in exp_names:
        label, data = load_experiment(name)
        if label:
            loaded.append((label, data))

    if not loaded:
        print("No experiments loaded.")
        return

    plot(loaded, args.metric, args.output)

    # print summary table
    print(f"\n{'Experiment':<45} {'Steps':>6} {'Peak reward':>12} {'Final reward':>12} {'Final KL':>10}")
    print("-" * 90)
    for name, data in loaded:
        if data is None:
            print(f"{name:<45} {'N/A':>6}")
            continue
        rewards = [r.get("reward", 0) for r in data if "reward" in r]
        kls = [r.get("kl", 0) for r in data if "kl" in r]
        steps = [r.get("step", i) for i, r in enumerate(data)]
        peak = max(rewards) if rewards else 0
        final_r = rewards[-1] if rewards else 0
        final_kl = kls[-1] if kls else 0
        print(f"{name:<45} {len(data):>6} {peak:>12.3f} {final_r:>12.3f} {final_kl:>10.5f}")


if __name__ == "__main__":
    main()
