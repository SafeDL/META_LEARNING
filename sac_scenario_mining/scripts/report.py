"""Create the Stage 1 comparison table and required audit figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"run": path.parent.name, **data}


def _read_ttc(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [float(row["min_ttc"]) for row in csv.DictReader(handle)]


def _learning_curve(monitor_path: Path) -> tuple[list[int], list[float]]:
    steps, returns, total = [], [], 0
    with monitor_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            if line.startswith("r,"):
                continue
            reward, length, _ = line.strip().split(",")
            total += int(length)
            steps.append(total)
            returns.append(float(reward))
    return steps, returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root",
                        default="results/sac_scenario_mining/final_eval")
    parser.add_argument("--training-root",
                        default="results/sac_scenario_mining")
    args = parser.parse_args()

    root, train_root = Path(args.results_root), Path(args.training_root)
    rows = [
        _read_summary(path) for path in sorted(root.glob("*/summary.json"))
    ]
    if not rows:
        raise SystemExit(f"No evaluation summaries in {root}")

    fields = [
        "run", "episodes", "valid_critical_rate", "target_collision_rate",
        "critical_rate", "invalid_rate", "median_min_ttc", "mean_min_ttc",
        "median_min_distance", "mean_episode_return",
        "episodes_to_first_valid_critical"
    ]
    with (root / "comparison.csv").open("w", newline="",
                                        encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    labels = [row["run"] for row in rows]

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, [row["valid_critical_rate"] for row in rows],
             label="valid critical")
    axis.bar(labels, [row["invalid_rate"] for row in rows],
             bottom=[row["valid_critical_rate"] for row in rows],
             label="invalid")
    axis.set_ylim(0, 1)
    axis.set_ylabel("episode rate")
    axis.set_title("Held-out scenario-mining outcomes")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "valid_critical_rate.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for row in rows:
        episodes = root / row["run"] / "episodes.csv"
        axis.hist(_read_ttc(episodes),
                  bins=20,
                  histtype="step",
                  linewidth=1.8,
                  label=row["run"])
    axis.set_xlabel("minimum TTC (s)")
    axis.set_ylabel("episode count")
    axis.set_title("Held-out minimum TTC distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "min_ttc_distribution.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for monitor in sorted(
            train_root.glob("merge_sac_v2_seed*/train_monitor.monitor.csv")):
        steps, returns = _learning_curve(monitor)
        if steps:
            axis.plot(steps, returns, alpha=0.35, label=monitor.parent.name)
    axis.set_xlabel("environment steps")
    axis.set_ylabel("episode return")
    axis.set_title("SAC training learning curves")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {root / 'comparison.csv'} and figures in {plots}")


if __name__ == "__main__":
    main()
