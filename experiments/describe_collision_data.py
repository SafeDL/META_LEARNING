"""Descriptive figures for audited historical responses on common coordinates."""

import csv
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.highway.catalog import SUTS

from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .scenarios import DOMAINS
from .storage import load_archive

OUTPUT = ROOT / "experiment_outputs/benchmark/collision/data_overview"
LABELS = {
    "idm_mobil": "IDM/MOBIL",
    "fvdm_lane": "FVDM/rules",
    "linear_mpc": "Linear MPC",
    "nonlinear_trajectory": "Nonlinear traj.",
    "uct": "UCT",
    "beam": "Beam",
    "ppo": "PPO",
    "dqn": "DQN",
}
TITLES = {
    "front_brake": "Front-vehicle braking",
    "cut_in": "Neighbor cut-in",
    "cut_out": "Lead cut-out with hidden obstacle",
    "lane_change_rear": "Lane-change rear conflict",
}
COLORS = {"rule": "#376b96", "optimization": "#277e73", "search": "#a56825", "learning": "#8062a2"}


def draw(kind, sources, counts, disagreement, response_distance):
    labels = [LABELS[s.lineage] for s in sources]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10}):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15.5, 5.5),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [0.9, 1.2, 1.2]},
        )
        axis = axes[0]
        bars = axis.barh(labels, counts, color=[COLORS[s.family] for s in sources])
        axis.invert_yaxis()
        axis.set(
            xlabel="Collisions in the 512-point archive",
            xlim=(0, DEFAULT.candidate_count),
            title="Observed collision responses",
        )
        axis.bar_label(bars, padding=3, fontsize=8)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
        for axis, values, title, label, upper, digits in (
            (
                axes[1],
                100 * disagreement,
                "Collision-label disagreement (%)",
                "Percent of differing responses",
                100,
                1,
            ),
            (
                axes[2],
                response_distance,
                "Continuous-response distance",
                "Normalized response RMSE",
                1,
                2,
            ),
        ):
            raster = axis.imshow(values, vmin=0, vmax=upper, cmap="Blues")
            axis.set(
                xticks=range(len(labels)),
                yticks=range(len(labels)),
                xticklabels=labels,
                yticklabels=labels,
                title=title,
            )
            plt.setp(
                axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8
            )
            axis.tick_params(axis="y", labelsize=8)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(
                        j,
                        i,
                        f"{values[i, j]:.{digits}f}",
                        ha="center",
                        va="center",
                        color="white" if values[i, j] > 0.6 * upper else "#172533",
                        fontsize=7,
                    )
            figure.colorbar(raster, ax=axis, fraction=0.045, pad=0.02, label=label)
        figure.suptitle(f"Historical SUT response diversity: {TITLES[kind]}", fontsize=13)
        for extension in ("png", "svg", "pdf"):
            figure.savefig(OUTPUT / f"historical_responses_{kind}.{extension}", dpi=180)
        plt.close(figure)


def main():
    audit = json.loads(
        (ROOT / "experiment_outputs/benchmark/collision/construction_data_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if audit["errors"]:
        raise ValueError("Resolve data audit inconsistencies before descriptive reporting")
    verified = {(r["source_id"], r["scenario"]): r for r in audit["banks"]}
    sources = [s for s in SUTS if s.role == "history"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, scenarios, pending = [], {}, []
    for kind in DOMAINS:
        paths = [TEST_RECORDS_ROOT / "history" / kind / f"{s.sut_id}.npz" for s in sources]
        if any(not p.exists() or (s.sut_id, kind) not in verified for s, p in zip(sources, paths)):
            pending.append(kind)
            continue
        for s, path in zip(sources, paths):
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != verified[(s.sut_id, kind)]["archive_sha256"]
            ):
                raise ValueError("Historical responses changed after their raw-execution audit")
        archives = [load_archive(p) for p in paths]
        if any(
            a.scenarios != archives[0].scenarios or len(a.scenarios) != DEFAULT.candidate_count
            for a in archives
        ):
            raise ValueError(
                "Source disagreement requires the same audited physical coordinate pool"
            )
        labels = np.stack([a.collision for a in archives])
        auxiliary = np.stack([a.auxiliary for a in archives])
        counts = labels.sum(1)
        disagreement = (labels[:, None] != labels[None]).mean(2)
        response_distance = np.sqrt(
            np.mean((auxiliary[:, None] - auxiliary[None]) ** 2, axis=(2, 3))
        )
        for values in (disagreement, response_distance):
            if not (
                np.isfinite(values).all()
                and np.allclose(values, values.T)
                and np.allclose(np.diag(values), 0)
                and values.min() >= 0
                and values.max() <= 1
            ):
                raise ValueError("Historical response comparison is not a valid symmetric distance")
        draw(kind, sources, counts, disagreement, response_distance)
        scenarios[kind] = {
            "candidates": len(archives[0].scenarios),
            "source_ids": [s.sut_id for s in sources],
            "collision_counts": counts.tolist(),
            "collision_disagreement_fraction": disagreement.tolist(),
            "normalized_continuous_response_rmse": response_distance.tolist(),
            "archive_sha256": {
                s.sut_id: verified[(s.sut_id, kind)]["archive_sha256"] for s in sources
            },
        }
        rows.extend(
            {
                "scenario": kind,
                "source_id": s.sut_id,
                "family": s.family,
                "lineage": s.lineage,
                "candidate_count": DEFAULT.candidate_count,
                "collision_count": int(count),
            }
            for s, count in zip(sources, counts)
        )
    report = {
        "status": "partial" if pending else "complete",
        "scope": "Audited historical construction coordinates only",
        "continuous_normalization": "Clearance clipped to [-5,100] m divided by 100; TTC capped at 10 s divided by 10, matching the encoder inputs.",
        "comparison_unit": "Matched coordinates within each functional scenario; no naturalistic frequency weighting.",
        "scenarios": scenarios,
        "pending": pending,
    }
    (OUTPUT / "overview.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "collision_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "scenario",
                "source_id",
                "family",
                "lineage",
                "candidate_count",
                "collision_count",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        report["status"],
        len(scenarios),
        "/",
        len(DOMAINS),
        "audited historical response overviews",
        flush=True,
    )


if __name__ == "__main__":
    main()
