"""Audit disclosure records and evaluate the locked co-primary experiment."""
import numpy as np
from threadpoolctl import threadpool_limits

from methods.history_guided_testing.io import read_json, write_json

from .config import BUDGET, SEEDS
from .confirmation import CONFIRMATION, METHODS, PRIMARY_CONTROLS, REPLICATES, verify_lock
from .develop import metrics


def sign_flip_p(differences):
    count = len(differences)
    observed = np.abs(differences.mean(0))
    extreme = np.zeros(differences.shape[1], dtype=np.int64)
    with threadpool_limits(limits=1):
        for start in range(0, 2**count, 65536):
            combinations = np.arange(start,
                                     min(start + 65536, 2**count),
                                     dtype=np.uint32)
            signs = 2 * ((combinations[:, None] >> np.arange(count))
                         & 1).astype(np.int8) - 1
            null = signs.astype(float) @ differences / count
            extreme += (np.abs(null) >= observed - 1e-12).sum(0)
    return extreme / 2**count


def bootstrap_indices(controllers, count=20000):
    rng = np.random.default_rng(20261007)
    return np.concatenate([
        rng.choice(np.flatnonzero(controllers == controller),
                   size=(count, int((controllers == controller).sum())),
                   replace=True) for controller in ("IDM", "FVDM")
    ],
                          axis=1)


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    records, per_profile = [], {method: [] for method in METHODS}
    for profile in protocol["profiles"]:
        rows = {method: [] for method in METHODS}
        for replicate in range(REPLICATES):
            folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
            bank = np.load(folder / "responses.npz")
            scenes = read_json(folder / "scenarios.json")
            if not np.array_equal(
                    bank["x"],
                    np.asarray([scene["numeric_input"] for scene in scenes])):
                raise ValueError(
                    "Measured confirmation coordinates differ from declared scenarios"
                )
            for method in METHODS:
                for seed in SEEDS:
                    result = read_json(folder / "selection" /
                                       f"{method}_{seed}.json")
                    selected = result["selected_indices"]
                    if len(selected) != BUDGET or len(set(selected)) != BUDGET:
                        raise ValueError(
                            "Invalid confirmation selection budget")
                    observations = result["observations"]
                    if [row["index"] for row in observations] != selected:
                        raise ValueError(
                            "Confirmation observations differ from selection sequence"
                        )
                    for row in observations:
                        index = row["index"]
                        if method == "ras_frt_uq":
                            if set(row) != {"index", "collision"
                                            } or row["collision"] != bool(
                                                bank["collision"][index]):
                                raise ValueError(
                                    "Incorrect or extra RAS feedback")
                        elif set(row) != {"index", "continuous_risk"
                                          } or row["continuous_risk"] != float(
                                              bank["risk"][index]):
                            raise ValueError(
                                "Incorrect or extra continuous-risk feedback")
                    measured = metrics(bank["collision"], selected)
                    for key in ("F200", "pool_collisions",
                                "mean_cumulative_collisions", "recall",
                                "curve"):
                        if result[key] != measured[key]:
                            raise ValueError(
                                f"Saved metric {key} differs from full-pool truth"
                            )
                    total = measured["pool_collisions"]
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
                        measured["F200"] / min(BUDGET, total) if total else 1,
                        "queries_to_all_failures":
                        (int(
                            np.flatnonzero(
                                np.asarray(measured["curve"]) == total)[0]) +
                         1 if total and measured["all_failures_found"] else
                         0 if not total else None),
                        "selector_elapsed_s":
                        result["selector_elapsed_s"]
                    }
                    records.append(row)
                    rows[method].append(row)
        for method in METHODS:
            per_profile[method].append({
                "profile": profile["name"],
                "controller": profile["controller"],
                **{
                    key: float(np.mean([row[key] for row in rows[method]]))
                    for key in ("normalized_area", "recall", "F50", "F100", "F200", "mean_cumulative_collisions", "missed_failures", "all_failures_found", "budget_upper_bound_attainment", "selector_elapsed_s")
                }
            })
    keys = [(control, metric) for control in PRIMARY_CONTROLS
            for metric in ("normalized_area", "recall")]
    differences = np.column_stack([[
        a[metric] - b[metric]
        for a, b in zip(per_profile["candidate"], per_profile[control])
    ] for control, metric in keys])
    p = sign_flip_p(differences)
    order = np.argsort(p)
    holm = np.empty_like(p)
    holm[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    controllers = np.array(
        [profile["controller"] for profile in protocol["profiles"]])
    indices = bootstrap_indices(controllers)
    bootstrap = differences[indices].mean(1)
    comparisons = {}
    for column, (control, metric) in enumerate(keys):
        comparisons.setdefault(control, {})[metric] = {
            "mean_difference":
            float(differences[:, column].mean()),
            "per_profile_differences":
            differences[:, column].tolist(),
            "bootstrap_95_ci":
            np.quantile(bootstrap[:, column], [0.025, 0.975]).tolist(),
            "exact_two_sided_p":
            float(p[column]),
            "holm_p":
            float(holm[column]),
            "round_spending_adjusted_p":
            min(1,
                float(holm[column]) * 2**protocol["round"])
        }
    success = all(value["mean_difference"] > 0
                  and value["holm_p"] < protocol["round_alpha"]
                  for values in comparisons.values()
                  for value in values.values())
    success = success and all(
        value["bootstrap_95_ci"][0] > 0
        for value in comparisons["previous_best"].values())
    aggregate = {
        method: {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "profile_std": float(np.std([row[key]
                                             for row in rows], ddof=1))
            }
            for key in rows[0] if key not in ("profile", "controller")
        }
        for method, rows in per_profile.items()
    }
    static_differences = np.column_stack([[
        a[metric] - b[metric]
        for a, b in zip(per_profile["candidate"], per_profile["class_rank"])
    ] for metric in ("normalized_area", "recall")])
    static = {
        metric: {
            "mean_difference":
            float(static_differences[:, column].mean()),
            "bootstrap_95_ci":
            np.quantile(static_differences[indices].mean(1)[:, column],
                        [0.025, 0.975]).tolist()
        }
        for column, metric in enumerate(("normalized_area", "recall"))
    }
    value = {
        "protocol": protocol,
        "success": bool(success),
        "records": records,
        "per_profile": per_profile,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "class_rank_exploratory_comparison": static,
        "scope": protocol["scope"],
        "full_pools_measured": True,
        "selector_disclosures_audited": len(records) * BUDGET,
        "statistical_units": len(protocol["profiles"])
    }
    write_json(CONFIRMATION / "summary.json", value)
    for method, row in aggregate.items():
        print("CONFIRMATION RESULT",
              method, {
                  key: round(row[key]["mean"], 5)
                  for key in ("normalized_area", "recall", "F200",
                              "mean_cumulative_collisions")
              },
              flush=True)
    print("CO-PRIMARY SUCCESS", success, comparisons, flush=True)
    return value


if __name__ == "__main__":
    main()
