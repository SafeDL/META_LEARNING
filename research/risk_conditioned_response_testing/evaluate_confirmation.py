"""Rebuild all pool metrics and apply the registered two-metric gate."""
import numpy as np

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.statistics import (
    bootstrap_indices, exact_sign_flip, holm_adjust)
from research.response_adaptive_testing.develop import metrics

from .confirmation import CONFIRMATION, verify_lock


def measure_row(collision, selected, result, profile, method, replicate, seed,
                budget):
    actual = metrics(collision, selected)
    actual.update({
        f"F{count}": actual["curve"][count - 1]
        for count in (10, 30, 50, 100, 150, 200)
    })
    for key, value in actual.items():
        if result.get(key) != value:
            raise ValueError(f"Cached metric {key} differs from rebuilt count")
    total = actual["pool_collisions"]
    return {
        "profile": profile["name"],
        "controller": profile["controller"],
        "replicate": replicate,
        "method": method,
        "seed": seed,
        **actual,
        "normalized_area": actual["mean_cumulative_collisions"] / total
        if total else 0.0,
        "missed_failures": total - actual["F200"],
        "budget_upper_bound_attainment": actual["F200"] / min(budget, total)
        if total else 1.0,
        "queries_to_all_failures": int(
            np.flatnonzero(np.asarray(actual["curve"]) == total)[0]) + 1
        if total and actual["all_failures_found"] else 0 if not total else None,
        "selector_elapsed_s": result["selector_elapsed_s"],
    }


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    methods = protocol["methods"]
    rows = []
    per_profile = {method: [] for method in methods}
    for profile in protocol["profiles"]:
        grouped = {method: [] for method in methods}
        for replicate in range(protocol["replicates_per_profile"]):
            folder = (CONFIRMATION / profile["name"] /
                      f"pool_{replicate}")
            with np.load(folder / "responses.npz") as bank:
                collision = bank["collision"]
                for method in methods:
                    for seed in protocol["seeds"]:
                        result = read_json(folder / "selection" /
                                           f"{method}_{seed}.json")
                        selected = result["selected_indices"]
                        if (len(selected) != protocol["budget"] or
                                len(selected) != len(set(selected))):
                            raise ValueError("Invalid query count or duplicate")
                        expected = ([{"index": index,
                                      "collision": bool(collision[index])}
                                     for index in selected]
                                    if method == "ras_frt_uq" else
                                    [{"index": index,
                                      "continuous_risk": float(bank["risk"][index])}
                                     for index in selected])
                        if result["observations"] != expected:
                            raise ValueError(
                                "Logged feedback differs from one-step disclosure")
                        row = measure_row(collision, selected, result, profile,
                                          method, replicate, seed,
                                          protocol["budget"])
                        rows.append(row)
                        grouped[method].append(row)
        for method, local_rows in grouped.items():
            keys = ("normalized_area", "recall", "F10", "F30", "F50",
                    "F100", "F150", "F200", "mean_cumulative_collisions",
                    "missed_failures", "all_failures_found",
                    "budget_upper_bound_attainment", "selector_elapsed_s")
            per_profile[method].append({
                "profile": profile["name"],
                "controller": profile["controller"],
                **{key: float(np.mean([r[key] for r in local_rows]))
                   for key in keys},
            })
    controls = protocol["primary_controls"]
    metric_names = ("normalized_area", "recall")
    differences = np.column_stack([
        np.asarray([candidate[metric] - control_row[metric]
                    for candidate, control_row in zip(
                        per_profile["risk_conditioned"],
                        per_profile[control])])
        for control in controls for metric in metric_names
    ])
    probabilities = exact_sign_flip(differences)
    adjusted = holm_adjust(probabilities)
    bootstrap = bootstrap_indices(protocol["profiles"])
    intervals = np.quantile(differences[bootstrap].mean(1),
                            [0.025, 0.975], axis=0).T
    comparisons = {}
    for column, (control, metric) in enumerate(
            (control, metric) for control in controls for metric in metric_names):
        comparisons.setdefault(control, {})[metric] = {
            "mean_difference": float(differences[:, column].mean()),
            "per_profile_differences": differences[:, column].tolist(),
            "bootstrap_95_ci": intervals[column].tolist(),
            "exact_two_sided_p": float(probabilities[column]),
            "holm_p": float(adjusted[column]),
        }
    success = bool(np.all(differences.mean(0) > 0) and
                   np.all(adjusted < protocol["round_alpha"]))
    success = success and all(
        comparisons[control][metric]["bootstrap_95_ci"][0] > 0
        for control in ("previous_best", "matched_collision_gp")
        for metric in metric_names)

    secondary = {}
    for control in ("behavior_posterior", "collision_only_ablation"):
        secondary[control] = {}
        for metric in metric_names:
            delta = np.asarray([
                candidate[metric] - reference[metric]
                for candidate, reference in zip(per_profile["risk_conditioned"],
                                                per_profile[control])])
            ci = np.quantile(delta[bootstrap].mean(1), [0.025, 0.975])
            secondary[control][metric] = {
                "mean_difference": float(delta.mean()),
                "bootstrap_95_ci_descriptive_only": ci.tolist(),
            }

    aggregate = {
        method: {
            metric: {
                "mean": float(np.mean([profile[metric]
                                       for profile in per_profile[method]])),
                "profile_std": float(np.std(
                    [profile[metric] for profile in per_profile[method]],
                    ddof=1)),
            }
            for metric in per_profile[method][0]
            if metric not in ("profile", "controller")
        }
        for method in methods
    }
    summary = {
        "protocol": protocol,
        "records": rows,
        "per_profile": per_profile,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "mechanism_comparisons_descriptive": secondary,
        "success": success,
        "statistical_units": len(protocol["profiles"]),
        "selector_disclosures_audited": len(rows) * protocol["budget"],
        "full_pools_measured": True,
        "scope": protocol["scope"],
    }
    write_json(CONFIRMATION / "summary.json", summary)
    print("RISK-CONDITIONED CO-PRIMARY SUCCESS", success,
          "AREA VS BEST", comparisons["previous_best"]["normalized_area"],
          "RECALL VS BEST", comparisons["previous_best"]["recall"],
          flush=True)
    return summary


if __name__ == "__main__":
    main()
