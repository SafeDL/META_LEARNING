"""Plot the recorded episodic Inner SAC return curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGE_COLORS = {
    "interaction_prior": "#4477aa",
    "context_meta": "#cc6677",
}


def _recorded_stages(manifest_path: str) -> dict[str, dict]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stages = {row["stage"]: row["metrics"] for row in manifest["stages"]}
    for stage in STAGE_COLORS:
        stage_path = path.with_name(f"{stage}.json")
        if stage not in stages and stage_path.exists():
            stages[stage] = json.loads(stage_path.read_text(encoding="utf-8"))["metrics"]
    return stages


def run(manifest_path: str, output: str) -> None:
    stages = _recorded_stages(manifest_path)
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    plotted = False
    for stage, color in STAGE_COLORS.items():
        if stage not in stages:
            continue
        curve = stages[stage]["episode_return_curve"]
        if not curve:
            continue
        values = np.asarray([row["inner_return"] for row in curve], dtype=float)
        episodes = np.arange(1, len(values) + 1)
        window = min(12, len(values))
        trend = np.convolve(values, np.ones(window) / window, mode="same")
        axis.plot(episodes, values, color=color, alpha=0.25, linewidth=1.0)
        axis.plot(episodes, trend, color=color, linewidth=2.0, label=f"{stage} ({window}-episode mean)")
        plotted = True
    if not plotted:
        raise ValueError("manifest contains no recorded Inner SAC return curves")
    axis.set_xlabel("Episode within stage")
    axis.set_ylabel("Inner SAC return")
    axis.set_title("Cut-in Inner SAC training return")
    axis.grid(alpha=0.25)
    axis.legend()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
