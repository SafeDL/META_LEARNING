"""Generate the four compact figures required by the highway-env MVP design."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ..data.response_bank import ResponseBank
from ..envs.cutin_env import CutInEnv, CutInScenario
from ..diva.low_rank_prior import LowRankPrior
from ..sut.idm_profiles import get_profile


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def plot_vulnerability_maps(bank: ResponseBank, output: Path) -> None:
    figure, axes = plt.subplots(
        2, 3, figsize=(12, 7), sharex=True, sharey=True, layout="constrained"
    )
    for axis, sut_name, response in zip(axes.flat, bank.sut_names, bank.vulnerability, strict=True):
        image = axis.scatter(bank.anchors[:, 1], bank.anchors[:, 0], c=response, cmap="magma", vmin=0, vmax=1)
        axis.set_title(sut_name)
        axis.set_xlabel("Relative speed (m/s)")
        axis.set_ylabel("Initial gap (m)")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="Vulnerability")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_svd_variance(bank: ResponseBank, output: Path) -> None:
    evrs = []
    for target_index in range(len(bank.sut_names)):
        prior = LowRankPrior.fit(np.delete(bank.vulnerability, target_index, axis=0), 2)
        evrs.append(prior.explained_variance_ratio)
    mean_evr = np.mean(np.vstack(evrs), axis=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(np.arange(1, len(mean_evr) + 1), mean_evr * 100)
    axis.axhline(80, color="tab:red", linestyle="--", label="80% rank-2 criterion")
    axis.set(xlabel="Singular component", ylabel="Explained variance (%)", xticks=np.arange(1, len(mean_evr) + 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ranking(rows: list[dict], output: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(float(row["ndcg_at_10"]))
    labels, values, errors = [], [], []
    for method, samples in grouped.items():
        labels.append(method.replace(" + Adaptation", ""))
        values.append(np.mean(samples))
        errors.append(np.std(samples))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, yerr=errors, capsize=4)
    axis.set(ylabel="NDCG@10", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mining(rows: list[dict], output: Path) -> None:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(np.fromstring(row["curve"], sep=";"))
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, curves in grouped.items():
        stacked = np.vstack(curves)
        x_axis = np.arange(1, stacked.shape[1] + 1)
        mean, std = stacked.mean(axis=0), stacked.std(axis=0)
        axis.plot(x_axis, mean, label=method)
        if len(curves) > 1:
            axis.fill_between(x_axis, mean - std, mean + std, alpha=0.16)
    axis.set(xlabel="Target test budget (support included)", ylabel="Cumulative critical score")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _annotate_frame(
    frame: np.ndarray, sut_name: str, scenario: CutInScenario, time_seconds: float
) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 40), fill=(20, 20, 20))
    draw.text(
        (8, 7),
        f"Highway-env Cut-in | {sut_name} | t={time_seconds:.1f}s | "
        f"gap={scenario.initial_gap:.1f}m, dv={scenario.relative_speed:.1f}m/s",
        fill=(255, 255, 255),
    )
    return image


def save_cutin_gif(
    sut_name: str, scenario: CutInScenario, output: Path, seed: int = 0
) -> dict[str, object]:
    """Render one representative hazardous Cut-in episode for a named SUT."""
    env = CutInEnv(get_profile(sut_name), scenario, render_mode="rgb_array")
    env.reset(seed=seed)
    frames: list[Image.Image] = []
    terminated = truncated = False
    while not (terminated or truncated):
        frame = env.render()
        frames.append(_annotate_frame(frame, sut_name, scenario, env.time))
        _, _, terminated, truncated, _ = env.step(1)
    if not frames:
        raise RuntimeError("Highway-env Cut-in replay produced no frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        disposal=2,
        optimize=True,
    )
    result = env.episode_result()
    env.close()
    return {
        "sut": sut_name,
        "scenario": {"initial_gap": scenario.initial_gap, "relative_speed": scenario.relative_speed},
        "frames": len(frames),
        "collision": result.collision,
        "near_miss": result.near_miss,
        "vulnerability": result.vulnerability,
        "gif": output.name,
    }


def render_results(
    bank: ResponseBank, ranking_path: Path, mining_path: Path, output_dir: Path
) -> dict[str, object]:
    """Write the four static figures and one replay GIF per SUT."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_vulnerability_maps(bank, output_dir / "vulnerability_maps_highway.png")
    plot_svd_variance(bank, output_dir / "svd_explained_variance_highway.png")
    plot_ranking(_read_rows(ranking_path), output_dir / "ranking_ndcg_highway.png")
    plot_mining(_read_rows(mining_path), output_dir / "mining_curve_highway.png")
    replays = []
    for sut_index, sut_name in enumerate(bank.sut_names):
        anchor_index = int(np.argmax(bank.vulnerability[sut_index]))
        anchor = bank.anchors[anchor_index]
        replays.append(
            save_cutin_gif(
                sut_name,
                CutInScenario(float(anchor[0]), float(anchor[1])),
                output_dir / f"cutin_replay_{sut_name.lower()}.gif",
                seed=anchor_index,
            )
        )
    manifest = {
        "simulator": "highway-env",
        "response_bank": str(bank.anchors.shape),
        "static_figures": [
            "vulnerability_maps_highway.png",
            "svd_explained_variance_highway.png",
            "ranking_ndcg_highway.png",
            "mining_curve_highway.png",
        ],
        "replays": replays,
    }
    (output_dir / "visualization_manifest_highway.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/diva_highway/cutin_mvp/response_bank_highway.npz"))
    parser.add_argument("--ranking", type=Path, default=Path("results/diva_highway/cutin_mvp/loso_ranking_highway.csv"))
    parser.add_argument("--mining", type=Path, default=Path("results/diva_highway/cutin_mvp/loso_mining_highway.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diva_highway/cutin_mvp"))
    args = parser.parse_args()
    bank = ResponseBank.load(args.bank)
    render_results(bank, args.ranking, args.mining, args.output_dir)


if __name__ == "__main__":
    main()
