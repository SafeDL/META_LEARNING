"""Analyze normalized RSS margins and response-time sensitivity on recorded trajectories."""

import csv
import gzip
import json
from dataclasses import asdict, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from .render_scenarios import BANKS, CASES, OUTPUT
from .rss import DEFAULT, frame_margin, measurement_contract

RESPONSE_TIMES = (0.3, 0.5, 1.0, 1.5)


def summarize_trajectory(trajectory, parameters):
    raw, normalized = [], []
    for frame in trajectory:
        diagnosis = frame_margin(frame, parameters)
        normalized.append(diagnosis["margin"])
        raw.append(
            min(
                max(
                    value * pair["required_center_separation_m"][direction]
                    for direction, value in pair["directional_margins"].items()
                )
                for pair in diagnosis["pairs"]
            )
        )
    return {
        "raw_minimum_m": min(raw),
        "normalized_minimum": min(normalized),
        "initial_normalized": normalized[0],
        "normalized_minimum_at_initial": int(np.argmin(normalized)) == 0,
        "negative_sample_fraction": float(np.mean(np.asarray(normalized) < 0)),
        "raw_normalized_frame_sign_mismatches": sum(
            (r < 0) != (n < 0) for r, n in zip(raw, normalized)
        ),
    }


def distribution(values):
    quantiles = np.quantile(values, (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1))
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), quantiles))


def collision_auc(labels, risk):
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive * negative:
        return None
    ranks = rankdata(risk)
    return float(
        (sum(r for r, y in zip(ranks, labels) if y) - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def plot_comparison(groups):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    for axis, (kind, title, rows) in zip(axes.flat, groups):
        for collision, color, label in (
            (False, "#3D78A6", "No collision"),
            (True, "#CE5656", "Collision"),
        ):
            values = np.sort(
                [row["normalized_minimum"] for row in rows if row["collision"] == collision]
            )
            if len(values):
                axis.step(
                    values,
                    np.arange(1, len(values) + 1) / len(values),
                    where="post",
                    color=color,
                    label=f"{label} (n={len(values)})",
                )
        axis.axvline(0, color="#53616C", linewidth=1, linestyle="--")
        axis.set(
            title=title,
            xlabel="Minimum normalized RSS margin",
            ylabel="Within-group empirical CDF",
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.15)
        axis.legend(fontsize=8, loc="lower right")
        raw_count = len({round(row["raw_minimum_m"], 8) for row in rows})
        normalized_count = len({round(row["normalized_minimum"], 8) for row in rows})
        axis.set_title(
            f"{title}\nDistinct raw / normalized scores: {raw_count} / {normalized_count}"
        )
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0,
        0.95,
        "Existing trajectory trial\n\n4 DQN banks + 1 linear MPC bank\n512 scenarios per bank; 2,560 total\nADS / other response = 1 / 2 s\n\nNegative margin: distance-envelope violation\nCollision uses the saved simulator label.\n\nFull trajectories include impact.\nThese are descriptive development checks,\nnot pre-impact prediction performance.",
        va="top",
        color="#243746",
        linespacing=1.5,
    )
    figure.suptitle(
        "RSS margin scaling on recorded driving scenarios", fontsize=16, color="#243746"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"rss_trial.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    results, groups, csv_rows = [], [], []
    for kind, target, title, _ in CASES:
        rows, sensitivity = [], {tau: [] for tau in RESPONSE_TIMES}
        for path in sorted((BANKS / target / kind / "env_81001/executions").glob("*.json.gz")):
            execution = json.loads(gzip.decompress(path.read_bytes()))
            for tau in RESPONSE_TIMES:
                summary = summarize_trajectory(
                    execution["trajectory"], replace(DEFAULT, ego_response_s=tau)
                )
                row = {
                    "kind": kind,
                    "target": target,
                    "physical_index": int(path.name.split(".")[0]),
                    "scenario_id": execution["scenario"]["scenario_id"],
                    "collision": execution["feedback"]["collision"],
                    "ego_response_s": tau,
                    "other_response_s": DEFAULT.other_response_s,
                    **summary,
                }
                sensitivity[tau].append(row)
                csv_rows.append(row)
                if tau == DEFAULT.ego_response_s:
                    rows.append(row)
        if len(rows) != 512:
            raise ValueError(f"Expected the complete 512-execution bank: {kind}/{target}")
        labels = [row["collision"] for row in rows]
        raw = np.array([row["raw_minimum_m"] for row in rows])
        normalized = np.array([row["normalized_minimum"] for row in rows])
        positive = sum(labels)
        result = {
            "kind": kind,
            "target": target,
            "executions": len(rows),
            "collisions": positive,
            "collisions_with_negative_margin": int(
                sum(y and m < 0 for y, m in zip(labels, normalized))
            ),
            "noncollisions_with_negative_margin": int(
                sum(not y and m < 0 for y, m in zip(labels, normalized))
            ),
            "initial_negative": sum(row["initial_normalized"] < 0 for row in rows),
            "minimum_at_initial": sum(row["normalized_minimum_at_initial"] for row in rows),
            "distinct_minimum_margins": len({round(m, 8) for m in normalized}),
            "raw_distinct_minimum_margins": len({round(m, 8) for m in raw}),
            "raw_distribution_m": distribution(raw),
            "normalized_distribution": distribution(normalized),
            "collision_normalized_distribution": distribution(normalized[np.array(labels)]),
            "noncollision_normalized_distribution": distribution(normalized[~np.array(labels)]),
            "raw_descriptive_collision_association_auc": collision_auc(labels, -raw),
            "descriptive_collision_association_auc": collision_auc(labels, -normalized),
            "raw_normalized_spearman": float(spearmanr(raw, normalized).statistic)
            if np.ptp(raw) > 1e-8
            else None,
            "raw_normalized_frame_sign_mismatches": sum(
                row["raw_normalized_frame_sign_mismatches"] for row in rows
            ),
            "negative_sample_fraction_distribution": distribution(
                [row["negative_sample_fraction"] for row in rows]
            ),
            "asinh_rank_max_difference": float(
                np.max(np.abs(rankdata(normalized) - rankdata(np.arcsinh(normalized))))
            ),
            "asinh_zero_sign_mismatches": int(
                np.sum((normalized < 0) != (np.arcsinh(normalized) < 0))
            ),
            "zero_clipping_distinct_values": len({round(max(0, -m), 8) for m in normalized}),
            "response_time_sensitivity": [
                {
                    "ego_response_s": tau,
                    "other_response_s": DEFAULT.other_response_s,
                    "negative": sum(row["normalized_minimum"] < 0 for row in tau_rows),
                    "initial_negative": sum(row["initial_normalized"] < 0 for row in tau_rows),
                    "median_minimum_margin": float(
                        np.median([row["normalized_minimum"] for row in tau_rows])
                    ),
                    "rank_correlation_with_default": float(
                        spearmanr(
                            normalized, [row["normalized_minimum"] for row in tau_rows]
                        ).statistic
                    ),
                }
                for tau, tau_rows in sensitivity.items()
            ],
        }
        results.append(result)
        groups.append((kind, title, rows))
        print(json.dumps(result), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "bank_diagnostics.json").write_text(
        json.dumps(
            {
                "scope": "Complete 512-candidate bank for each displayed target/scenario pair.",
                "interpretation": "Metric-development diagnostics on existing target data, not independent validation of a new method. AUC uses complete trajectories including impact, not pre-impact predictions. Negative margin does not imply collision. Raw means max of four directed centre-separation deficits in metres, then min over actors and sampled times. Normalized means each directional deficit divided by its own required centre separation BEFORE max/min. No fitting, new simulator calls, or selector changes.",
                "parameters": asdict(DEFAULT),
                "continuous_response": measurement_contract(),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (OUTPUT / "rss_trial.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_comparison(groups)


if __name__ == "__main__":
    main()
