"""Create SAC diagnostics and fair held-out comparison figures."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> list[dict[str, str]]:
    # SB3 Monitor may emit a UTF-8 BOM on Windows; strip it so the first
    # header remains ``r`` rather than ``\ufeffr``.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _summary(path: Path) -> dict:
    return {"run": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))}


def _monitor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    total, steps, rewards = 0, [], []
    for row in rows:
        total += int(float(row["l"]))
        steps.append(total)
        rewards.append(float(row["r"]))
    return np.asarray(steps), np.asarray(rewards)


def _rolling(values: np.ndarray, width: int = 50) -> np.ndarray:
    if not len(values):
        return values
    kernel = np.ones(min(width, len(values))) / min(width, len(values))
    return np.convolve(values, kernel, mode="valid")


def _progress(path: Path) -> list[dict[str, str]]:
    return _rows(path) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/sac_scenario_mining/final_eval")
    parser.add_argument("--training-root", default="results/sac_scenario_mining")
    args = parser.parse_args()
    root, training = Path(args.results_root), Path(args.training_root)
    rows = [_summary(path) for path in sorted(root.glob("*/summary.json"))]
    if not rows:
        raise SystemExit(f"No evaluation summaries in {root}")
    fields = ["run", "episodes", "valid_critical_rate", "target_collision_rate", "critical_rate",
              "invalid_rate", "median_min_ttc", "mean_min_ttc", "median_min_distance",
              "mean_episode_return", "episodes_to_first_valid_critical", "split", "policy_name", "policy_seed"]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = root / "plots"; plots.mkdir(exist_ok=True)
    labels = [r["run"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, [r["valid_critical_rate"] for r in rows], label="valid critical")
    ax.bar(labels, [r["invalid_rate"] for r in rows], bottom=[r["valid_critical_rate"] for r in rows], label="invalid")
    ax.set(ylim=(0, 1), ylabel="case rate", title="Held-out on-ramp-merge outcomes")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "held_out_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for row in rows:
        episodes = _rows(root / row["run"] / "episodes.csv")
        ax.hist([float(e["min_ttc"]) for e in episodes], bins=20, histtype="step", linewidth=1.6, label=row["run"])
    ax.set(xlabel="minimum TTC (s)", ylabel="case count", title="Held-out minimum TTC distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "min_ttc_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for monitor in sorted(training.glob("merge_sac_seed*/train_monitor.monitor.csv")):
        steps, returns = _monitor(monitor)
        if len(returns):
            ax.plot(steps, returns, alpha=.18)
            ax.plot(steps[len(returns) - len(_rolling(returns)):], _rolling(returns), label=monitor.parent.name)
    ax.set(xlabel="environment steps", ylabel="episode return", title="SAC training return (raw and rolling mean)")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "train_episode_return.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    loss_fields = [("train/actor_loss", "actor loss"), ("train/critic_loss", "critic loss"),
                   ("train/ent_coef", "entropy coefficient")]
    for run in sorted(training.glob("merge_sac_seed*")):
        for axis, (field, label) in zip(axes, loss_fields):
            points = [(float(row["time/total_timesteps"]), float(row[field])) for row in _progress(run / "progress.csv")
                      if row.get("time/total_timesteps") and row.get(field)]
            if points:
                xs, ys = zip(*points)
                axis.plot(xs, ys, label=run.name)
            axis.set(xlabel="environment steps", ylabel=label, title=label)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / "sac_losses.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for run in sorted(training.glob("merge_sac_seed*")):
        xs, vcr, invalid = [], [], []
        for summary_path in sorted(run.glob("validation/step_*/summary.json"), key=lambda p: int(p.parent.name.split("_")[1])):
            xs.append(int(summary_path.parent.name.split("_")[1])); s = json.loads(summary_path.read_text(encoding="utf-8"))
            vcr.append(float(s["valid_critical_rate"])); invalid.append(float(s["invalid_rate"]))
        if xs:
            axes[0].plot(xs, vcr, marker="o", label=run.name); axes[1].plot(xs, invalid, marker="o", label=run.name)
    axes[0].set(ylabel="valid critical rate", xlabel="environment steps", title="validation effectiveness", ylim=(0, 1))
    axes[1].set(ylabel="invalid rate", xlabel="environment steps", title="validation validity", ylim=(0, 1))
    axes[0].legend(); fig.tight_layout(); fig.savefig(plots / "validation_metrics.png", dpi=160); plt.close(fig)
    print(f"Wrote comparison and diagnostics to {root}")


if __name__ == "__main__":
    main()
