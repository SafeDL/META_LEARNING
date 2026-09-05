"""Plot the publication curve for the shared Inner SAC interaction prior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DOMAIN_STYLES = {
    "close_closing_early": ("Close-closing early", "#4477aa"),
    "balanced_interaction": ("Balanced interaction", "#228833"),
    "late_tight_cutin": ("Late-tight cut-in", "#cc6677"),
}


def _recorded_stages(manifest_path: str) -> dict[str, dict]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stages = {row["stage"]: row["metrics"] for row in manifest["stages"]}
    stage_path = path.with_name("interaction_prior.json")
    if "interaction_prior" not in stages and stage_path.exists():
        stages["interaction_prior"] = json.loads(
            stage_path.read_text(encoding="utf-8")
        )["metrics"]
    return stages


def _trailing_mean(values: np.ndarray, window: int = 8) -> np.ndarray:
    """Return a trailing mean with ``min_periods=1`` and no edge padding."""
    if values.ndim != 1 or not len(values):
        raise ValueError("rolling mean requires a non-empty one-dimensional array")
    width = min(int(window), len(values))
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - width)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def run(manifest_path: str, output_prefix: str) -> None:
    stages = _recorded_stages(manifest_path)
    prior = stages.get("interaction_prior")
    if prior is None:
        raise ValueError("manifest contains no interaction-prior return curve")
    grouped: dict[str, list[float]] = {domain: [] for domain in DOMAIN_STYLES}
    for row in prior.get("episode_return_curve", []):
        domain = str(row["logical_domain_id"])
        if domain not in grouped:
            raise ValueError(f"unexpected Logical Domain in curve: {domain!r}")
        grouped[domain].append(float(row["inner_return"]))
    missing = [domain for domain, values in grouped.items() if not values]
    if missing:
        raise ValueError(
            "interaction-prior curve is missing required Logical Domains: "
            + ", ".join(missing)
        )

    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for domain, (label, color) in DOMAIN_STYLES.items():
        values = np.asarray(grouped[domain], dtype=float)
        episodes = np.arange(1, len(values) + 1)
        trend = _trailing_mean(values, window=8)
        axis.plot(episodes, values, color=color, alpha=0.15, linewidth=0.8)
        axis.plot(episodes, trend, color=color, linewidth=2.0, label=label)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", alpha=0.20, linewidth=0.6)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Inner episodic return")
    axis.legend(frameon=False)
    figure.tight_layout()
    output = Path(output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    run(args.manifest, args.output_prefix)


if __name__ == "__main__":
    main()
