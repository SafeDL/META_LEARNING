"""Audit all E02 tasks and summarize the fixed ADS cohort and response regions."""

import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from controllers.highway.catalog import FORMAL_TARGET_SUTS

from experiments.budget import protocol
from experiments.budget.audit import audit_target, result_directory, verify_historical_baselines
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.storage import read_json, write_json
from experiments.transfer_batches import prepare_context


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(values):
    """ADS x scene x method x metric; resample the same eight ADS for all contrasts."""
    expected = (8, 4, 7, 2)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("E02 statistics require the complete, finite registered matrix")
    draws = np.random.default_rng(1301).integers(0, 8, size=(10000, 8))
    result = {}
    for label, ads in [(kind, values[:, i]) for i, kind in enumerate(protocol.RUN_SCENARIOS)] + [("macro", values.mean(axis=1))]:
        bootstrap = ads[draws].mean(axis=1)
        methods = {}
        for j, method in enumerate(protocol.METHODS):
            low, high = np.quantile(bootstrap[:, j], (0.025, 0.975), axis=0)
            methods[method] = {"C200": float(ads[:, j, 0].mean()), "AUC": float(ads[:, j, 1].mean()),
                               "ci_low": low.tolist(), "ci_high": high.tolist()}
        comparisons = {}
        for j, method in enumerate(protocol.METHODS[1:], 1):
            gains = ads[:, 0] - ads[:, j]
            distribution = bootstrap[:, 0] - bootstrap[:, j]
            low, high = np.quantile(distribution, (0.025, 0.975), axis=0)
            simultaneous = {}
            for family in (10, 12):
                if family == 10 and method == "static_prior":
                    continue
                tail = 0.05 / (2 * family)
                a, b = np.quantile(distribution, (tail, 1 - tail), axis=0)
                simultaneous[str(family)] = {"low": a.tolist(), "high": b.tolist(),
                                             "interval_coverage": 1 - 0.05 / family}
            comparisons[method] = {"gain": gains.mean(axis=0).tolist(), "ci_low": low.tolist(),
                                    "ci_high": high.tolist(), "ads_gains": gains.tolist(),
                                    "wins": (gains > 1e-10).sum(axis=0).tolist(),
                                    "ties": (np.abs(gains) <= 1e-10).sum(axis=0).tolist(),
                                    "losses": (gains < -1e-10).sum(axis=0).tolist(),
                                    "simultaneous": simultaneous,
                                    "both_metrics_supported": bool((low > 0).all() and (gains.mean(axis=0) > 0).all())}
        result[label] = {"methods": methods, "ours_minus": comparisons, "valid_ads": 8}
    return {
        "metric_order": ["C200", "AUC"], "bootstrap_replicates": 10000, "bootstrap_seed": 1301,
        "unit": "ADS", "scene_aggregation": "equal scenes within each ADS, then equal ADS",
        "confidence_level": 0.95, "interval": "paired percentile, fixed ADS cohort",
        "scopes": result,
    }


def summarize(audits, baseline_checks, *, result_path=result_directory, destination=None):
    tasks, costs, curves, regions, populations = [], [], [], [], []
    values = np.empty((8, 4, 7, 2))
    audit_map = {(a["scene"], a["target"]): a for a in audits}
    for scene_index, kind in enumerate(protocol.RUN_SCENARIOS):
        labels = np.asarray([audit_map[kind, s.sut_id]["labels"] for s in FORMAL_TARGET_SUTS], dtype=bool)
        counts = labels.sum(axis=0)
        masks = {"shared_collision": counts == 8, "different_response": (counts > 0) & (counts < 8),
                 "shared_safe": counts == 0}
        populations.append({"scenario": kind, **{name: int(mask.sum()) for name, mask in masks.items()}})
        for ads_index, sut in enumerate(FORMAL_TARGET_SUTS):
            for method_index, method in enumerate(protocol.METHODS):
                path = result_path(sut.sut_id, kind, method) / "result.json.gz"
                saved = read_json(path)
                metrics, result = saved["metrics"], saved["result"]
                indices = result["selected_indices"]
                # Direct formulas independently checked above are used for every displayed curve.
                cumulative = np.cumsum(labels[ads_index, indices])
                if cumulative.tolist() != metrics["discovery_curve"]:
                    raise ValueError("Report curve differs from the audited native collision sequence")
                values[ads_index, scene_index, method_index] = (cumulative[-1], cumulative.mean())
                row = {
                    "scenario": kind, "sut_id": sut.sut_id, "lineage": sut.lineage, "method": method,
                    "seed": protocol.SEED, "queries": 200, "C20": int(cumulative[19]),
                    "C50": int(cumulative[49]), "C100": int(cumulative[99]), "C200": int(cumulative[-1]),
                    "AUC": float(cumulative.mean()), "post_initial_collisions": int(cumulative[-1] - cumulative[19]),
                    "first_collision_query": metrics["first_collision_query"],
                    "first_collision_right_censored": metrics["no_collision_within_budget"],
                    "pool_collisions": int(labels[ads_index].sum()),
                    "collision_recall": metrics["collision_recall"],
                    "new_paid_reveals": 0 if method == "random" else 200,
                    "reused_paid_reveals": 200 if method == "random" else 0,
                    "new_physical_executions": 0,
                }
                tasks.append(row)
                resource = saved["extraction_resources"] if method == "random" else result["resources"]
                costs.append({
                    "scenario": kind, "sut_id": sut.sut_id, "method": method,
                    "scope": "curve_extraction" if method == "random" else "selector_including_query_waits",
                    "device": resource["device"], "wall_s": resource["wall_s"],
                    "peak_cpu_bytes": resource["sampled_peak_cpu_bytes"],
                    "cpu_increase_bytes": resource["sampled_cpu_increase_bytes"],
                    "peak_cuda_allocated_bytes": resource["peak_cuda_allocated_bytes"],
                    "selection_s": metrics["selection_s"], "adaptation_s": metrics["adaptation_s"],
                    "rss_compute_s": metrics["rss_compute_s"], "replay_wall_s": metrics["replay_wall_s"],
                    "equivalent_execution_s": metrics["equivalent_execution_s"],
                    "selector_cost_measured": method != "random", "new_physical_executions": 0,
                })
                contributions = {}
                for region in ("shared_collision", "different_response"):
                    part = np.cumsum(labels[ads_index, indices] & masks[region][indices])
                    contributions[region] = part
                    regions.append({
                        "scenario": kind, "sut_id": sut.sut_id, "method": method, "region": region,
                        "coordinates": int(masks[region].sum()),
                        "available_collisions": int((labels[ads_index] & masks[region]).sum()),
                        "C200": int(part[-1]), "AUC": float(part.mean()),
                        "evidence_available": bool(masks[region].any()),
                    })
                if not np.array_equal(sum(contributions.values()), cumulative):
                    raise ValueError("Response region contributions do not close to total discoveries")
                for step, count in enumerate(cumulative, 1):
                    curves.append({"scenario": kind, "sut_id": sut.sut_id, "method": method,
                                   "query": step, "collisions": int(count),
                                   "shared_collision": int(contributions["shared_collision"][step - 1]),
                                   "different_response": int(contributions["different_response"][step - 1])})
    if (len(tasks) != 224 or len(curves) != 44800
            or sum(r["new_paid_reveals"] for r in tasks) != 38400
            or sum(r["reused_paid_reveals"] for r in tasks) != 6400):
        raise ValueError("E02 matrix or paid-feedback accounting does not close")
    statistics = paired_statistics(values)
    # Use the same ADS resampling to describe each response region's contribution.
    regional_statistics = {}
    for region in ("shared_collision", "different_response"):
        regional_values = np.zeros_like(values)
        for row in regions:
            if row["region"] == region:
                a = [s.sut_id for s in FORMAL_TARGET_SUTS].index(row["sut_id"])
                s = list(protocol.RUN_SCENARIOS).index(row["scenario"])
                m = protocol.METHODS.index(row["method"])
                regional_values[a, s, m] = row["C200"], row["AUC"]
        regional_statistics[region] = paired_statistics(regional_values)
    directory = protocol.OUTPUT / "reports" if destination is None else destination
    for name, rows in (("task_results", tasks), ("costs", costs), ("discovery_curves", curves),
                       ("response_regions", regions), ("region_populations", populations)):
        write_csv(directory / f"{name}.csv", rows)
    write_json(directory / "paired_summary.json", statistics)
    write_json(directory / "regional_summary.json", regional_statistics)
    if ({row["scenario"] for row in baseline_checks} != set(protocol.RUN_SCENARIOS)
            or len(baseline_checks) != 4):
        raise ValueError("Historical baseline reconstruction omits a registered scenario")
    write_json(directory / "historical_baseline_audit.json", baseline_checks)
    forecast_errors = [a["forecasts"]["maximum_probability_error"]
                       for t in audits for a in t["audits"] if "forecasts" in a]
    graph_errors = [a["maximum_graph_probability_error"]
                    for t in audits for a in t["audits"] if "maximum_graph_probability_error" in a]
    completion = {
        "status": "complete", "cases": len(tasks), "curve_rows": len(curves),
        "logical_budget_queries": 44800, "new_paid_reveals": 38400, "reused_paid_reveals": 6400,
        "new_physical_executions": 0, "native_truth_records": 64000,
        "maximum_reconstructed_gp_probability_error": max(forecast_errors),
        "maximum_reconstructed_graph_probability_error": max(graph_errors),
        "maximum_reconstructed_proxy_error": max(row["proxy_maximum_error"] for row in baseline_checks),
        "historical_baseline_scenarios_verified": len(baseline_checks),
        "all_common_initializations_verified": True, "region_curves_close": True,
        "primary_comparison_supported": statistics["scopes"]["macro"]["ours_minus"]["target_only"]["both_metrics_supported"],
        "static_adaptation_increment_supported": statistics["scopes"]["macro"]["ours_minus"]["static_prior"]["both_metrics_supported"],
        "files": {p.name: protocol.sha(p) for p in (
            *[directory / f"{name}.csv" for name in
              ("task_results", "costs", "discovery_curves", "response_regions", "region_populations")],
            directory / "paired_summary.json", directory / "regional_summary.json",
            directory / "historical_baseline_audit.json",
        )},
    }
    write_json(directory / "completion.json", completion)
    return completion


def report():
    initialize_worker()
    missing = [str(result_directory(s.sut_id, kind, method).relative_to(protocol.OUTPUT))
               for kind in protocol.RUN_SCENARIOS for s in FORMAL_TARGET_SUTS
               for method in protocol.METHODS if method != "random"
               if not (result_directory(s.sut_id, kind, method) / "result.json.gz").exists()]
    if missing:
        return {"status": "incomplete", "missing": missing, "expected_cases": 224}
    prepare_processes()
    audits, baseline_checks = [], []
    for kind in protocol.RUN_SCENARIOS:
        context, banks = prepare_context(protocol.INPUT, kind, "prediction")
        check, probability = verify_historical_baselines(banks[FORMAL_TARGET_SUTS[0].sut_id], context)
        baseline_checks.append(check)
        context["scenario_fuzz_probabilities"] = probability
        with ProcessPoolExecutor(max_workers=METHOD_WORKERS, mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker) as pool:
            futures = [pool.submit(audit_target, kind, s.sut_id, context) for s in FORMAL_TARGET_SUTS]
            for future in as_completed(futures):
                audits.append(future.result())
    return summarize(audits, baseline_checks)


if __name__ == "__main__":
    print(report(), flush=True)
