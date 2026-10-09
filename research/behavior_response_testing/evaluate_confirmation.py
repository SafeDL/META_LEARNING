"""Rebuild complete-pool metrics and apply the frozen dual-metric gate."""
import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from research.response_adaptive_testing.develop import metrics

from .confirmation import CONFIRMATION, verify_lock
from .statistics import bootstrap_indices, exact_sign_flip, holm_adjust


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    methods = protocol["methods"]
    records, per_profile = [], {method: [] for method in methods}
    for profile in protocol["profiles"]:
        grouped = {method: [] for method in methods}
        for replicate in range(protocol["replicates_per_profile"]):
            folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
            with np.load(folder / "responses.npz") as bank:
                for method in methods:
                    for seed in protocol["seeds"]:
                        result = read_json(folder / "selection" /
                                           f"{method}_{seed}.json")
                        selected = result["selected_indices"]
                        assert len(selected) == len(
                            set(selected)) == protocol["budget"]
                        expected = [{
                            "index": i,
                            "collision": bool(bank["collision"][i])
                        } if method == "ras_frt_uq" else {
                            "index": i,
                            "continuous_risk": float(bank["risk"][i])
                        } for i in selected]
                        assert result["observations"] == expected
                        measured = metrics(bank["collision"], selected)
                        measured.update({
                            f"F{n}": measured["curve"][n - 1]
                            for n in (10, 30, 150)
                        })
                        for key, value in measured.items():
                            assert result[key] == value
                        total = measured["pool_collisions"]
                        complete = measured["all_failures_found"]
                        row = {
                            "profile":
                            profile["name"],
                            "controller":
                            profile["controller"],
                            "replicate":
                            replicate,
                            "method":
                            method,
                            "seed":
                            seed,
                            **measured, "normalized_area":
                            measured["mean_cumulative_collisions"] /
                            total if total else 0,
                            "missed_failures":
                            total - measured["F200"],
                            "budget_upper_bound_attainment":
                            measured["F200"] /
                            min(protocol["budget"], total) if total else 1,
                            "queries_to_all_failures":
                            int(
                                np.flatnonzero(
                                    np.array(measured["curve"]) == total)[0]) +
                            1 if total and complete else
                            0 if not total else None,
                            "selector_elapsed_s":
                            result["selector_elapsed_s"]
                        }
                        records.append(row)
                        grouped[method].append(row)
        for method, rows in grouped.items():
            per_profile[method].append({
                "profile": profile["name"],
                "controller": profile["controller"],
                **{
                    key: float(np.mean([r[key] for r in rows]))
                    for key in ("normalized_area", "recall", "F10", "F30", "F50", "F100", "F150", "F200", "mean_cumulative_collisions", "missed_failures", "all_failures_found", "budget_upper_bound_attainment", "selector_elapsed_s")
                }
            })
    keys = [(control, metric) for control in protocol["primary_controls"]
            for metric in ("normalized_area", "recall")]
    differences = np.column_stack([[
        a[metric] - b[metric]
        for a, b in zip(per_profile["candidate"], per_profile[control])
    ] for control, metric in keys])
    probabilities = exact_sign_flip(differences)
    adjusted = holm_adjust(probabilities)
    indices = bootstrap_indices(protocol["profiles"])
    intervals = np.quantile(differences[indices].mean(1), [0.025, 0.975],
                            axis=0).T
    comparisons = {}
    for column, (control, metric) in enumerate(keys):
        comparisons.setdefault(control, {})[metric] = {
            "mean_difference":
            float(differences[:, column].mean()),
            "per_profile_differences":
            differences[:, column].tolist(),
            "bootstrap_95_ci":
            intervals[column].tolist(),
            "exact_two_sided_p":
            float(probabilities[column]),
            "holm_p":
            float(adjusted[column]),
            "round_spending_adjusted_p":
            min(1.,
                float(adjusted[column]) * 2**protocol["round"])
        }
    success = bool(
        np.all(differences.mean(0) > 0)
        and np.all(adjusted < protocol["round_alpha"]))
    success = success and all(
        value["bootstrap_95_ci"][0] > 0
        for control in ("previous_best", "matched_collision_gp")
        for value in comparisons[control].values())
    aggregate = {
        method: {
            key: {
                "mean": float(np.mean([r[key] for r in rows])),
                "profile_std": float(np.std([r[key] for r in rows], ddof=1))
            }
            for key in rows[0] if key not in ("profile", "controller")
        }
        for method, rows in per_profile.items()
    }
    summary = {
        "protocol": protocol,
        "records": records,
        "per_profile": per_profile,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "success": success,
        "statistical_units": len(protocol["profiles"]),
        "selector_disclosures_audited": len(records) * protocol["budget"],
        "scope": protocol["scope"],
        "full_pools_measured": True
    }
    write_json(CONFIRMATION / "summary.json", summary)
    print("BEHAVIOR CO-PRIMARY SUCCESS", success, comparisons, flush=True)
    return summary


if __name__ == "__main__":
    main()
