"""Plot the recorded episodic Inner SAC return curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run(manifest_path: str, output: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    stages = {row["stage"]: row["metrics"] for row in manifest["stages"]}
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for stage, color in (("interaction_prior", "#4477aa"), ("context_meta", "#cc6677")):
        curve = stages[stage]["episode_return_curve"]
        values = np.asarray([row["inner_return"] for row in curve], dtype=float)
        episodes = np.arange(1, len(values) + 1)
        window = min(12, len(values))
        trend = np.convolve(values, np.ones(window) / window, mode="same")
        axis.plot(episodes, values, color=color, alpha=0.25, linewidth=1.0)
        axis.plot(episodes, trend, color=color, linewidth=2.0, label=f"{stage} ({window}-episode mean)")
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
