"""Second-round confirmatory evaluation; no selection from fresh results."""
import json
from functools import lru_cache

import numpy as np

from methods.history_guided_testing.io import write_json
from methods.history_guided_testing.scenarios import parameter_cells

from .config import BUDGET, CONFIRMATION, METHODS


@lru_cache(maxsize=1)
def signs(count):
    combinations = np.arange(2**count, dtype=np.uint32)
    return (2 *
            ((combinations[:, None] >> np.arange(count)) & 1).astype(np.int8) -
            1)


def paired(differences):
    differences = np.asarray(differences, dtype=float)
    null = signs(len(differences)) @ differences / len(differences)
    p = float(np.mean(np.abs(null) >= abs(differences.mean()) - 1e-12))
    rng = np.random.default_rng(20261005)
    resampled = differences[rng.integers(0,
                                         len(differences),
                                         size=(20000,
                                               len(differences)))].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "per_pool_differences": differences.tolist(),
        "bootstrap_95_ci": np.quantile(resampled, [.025, .975]).tolist(),
        "exact_two_sided_sign_flip_p": p
    }


def main():
    protocol = json.loads(
        (CONFIRMATION / "protocol.json").read_text(encoding="utf-8"))
    records = {method: [] for method in METHODS}
    for pool in range(protocol["pool_count"]):
        folder = CONFIRMATION / f"pool_{pool}"
        bank = {
            row["index"]: row
            for row in map(json.loads, (folder /
                                        "measurements.jsonl").read_text(
                                            encoding="utf-8").splitlines())
        }
        scenes = json.loads(
            (folder / "scenarios.json").read_text(encoding="utf-8"))
        x = np.asarray([s["numeric_input"] for s in scenes])
        cells = parameter_cells(x)
        for method in METHODS:
            selected = json.loads((folder / f"{method}.json").read_text(
                encoding="utf-8"))["selected_indices"]
            assert len(selected) == len(set(selected)) == BUDGET
            collision = np.array([bank[i]["collision"] for i in selected])
            risks = np.array([bank[i]["risk"] for i in selected])
            curve = np.cumsum(collision)
            elite = np.zeros(512)
            np.maximum.at(elite, cells[selected], np.maximum(risks - .5, 0))
            row = {
                "pool": pool,
                "mean_cumulative_collisions": float(curve.mean()),
                "curve": curve.tolist(),
                "collision_cells_200": len(set(cells[selected][collision])),
                "risk_qd_200": float(elite.sum()),
                **{
                    f"F{k}": int(curve[k - 1])
                    for k in (10, 50, 100, 200)
                }
            }
            records[method].append(row)
    aggregate = {
        method: {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "std": float(np.std([r[key] for r in rows], ddof=1))
            }
            for key in ("F10", "F50", "F100", "F200",
                        "mean_cumulative_collisions", "collision_cells_200",
                        "risk_qd_200")
        }
        for method, rows in records.items()
    }
    comparisons = {
        method: {
            metric:
            paired([
                a[metric] - b[metric]
                for a, b in zip(records["candidate"], records[method])
            ])
            for metric in ("mean_cumulative_collisions", "F50", "F100", "F200")
        }
        for method in ("main", "ras_frt_uq")
    }
    ordered = sorted(
        comparisons,
        key=lambda name: comparisons[name]["mean_cumulative_collisions"][
            "exact_two_sided_sign_flip_p"])
    corrected = 0.
    for rank, method in enumerate(ordered):
        value = comparisons[method]["mean_cumulative_collisions"]
        corrected = max(
            corrected,
            min(1., (2 - rank) * value["exact_two_sided_sign_flip_p"]))
        value["holm_adjusted_p"] = corrected
        value["round_adjusted_p"] = min(1., 6 * corrected)
    success = all(
        comparisons[m]["mean_cumulative_collisions"]["holm_adjusted_p"] <
        protocol["round_alpha"]
        and comparisons[m]["mean_cumulative_collisions"]["mean_difference"] > 0
        for m in comparisons)
    success = success and comparisons["ras_frt_uq"][
        "mean_cumulative_collisions"]["mean_difference"] >= 1
    success = success and all(
        comparisons[m]["F200"]["bootstrap_95_ci"][0] >= -1
        for m in comparisons)
    components = {
        method:
        paired([
            a["mean_cumulative_collisions"] - b["mean_cumulative_collisions"]
            for a, b in zip(records["candidate"], records[method])
        ])
        for method in ("class_rank", "risk_only", "calibrated_only")
    }
    value = {
        "protocol": protocol,
        "records": records,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "components": components,
        "success": success,
        "statistical_unit": "20 independent Sobol pools",
        "scope":
        "fixed target FVDM, simulator and parameter ranges; no unseen controller-family claim",
        "full_pool_recall_evaluated": False,
        "secondary_p_values_are_exploratory": True
    }
    write_json(CONFIRMATION / "summary.json", value)
    for method in METHODS:
        print(method, {
            k: round(v["mean"], 3)
            for k, v in aggregate[method].items()
        },
              flush=True)
    print(
        "PRIMARY",
        {m: comparisons[m]["mean_cumulative_collisions"]
         for m in comparisons},
        flush=True)
    print("PREDECLARED SUCCESS", success, flush=True)
    draw(value)
    return value


def draw(value):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {
        "candidate": "Budget-dependent dual readouts",
        "main": "Frozen history GP",
        "ras_frt_uq": "RAS-FRT-UQ",
        "class_rank": "Historical collision ranking",
        "risk_only": "Continuous risk only",
        "calibrated_only": "Calibrated score only"
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method, rows in value["records"].items():
        curves = np.array([r["curve"] for r in rows])
        average, sd = curves.mean(0), curves.std(0, ddof=1) / np.sqrt(
            len(rows))
        line, = ax.plot(np.arange(1, BUDGET + 1),
                        average,
                        label=labels[method],
                        linewidth=2 if method == "candidate" else 1.2)
        ax.fill_between(np.arange(1, BUDGET + 1),
                        average - 1.96 * sd,
                        average + 1.96 * sd,
                        color=line.get_color(),
                        alpha=.08)
    ax.set(xlabel="Target scenario calls", ylabel="Discovered collisions")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(CONFIRMATION / "discovery.png", dpi=200)
    fig.savefig(CONFIRMATION / "discovery.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
