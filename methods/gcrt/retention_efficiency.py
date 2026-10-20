"""Single-seed matched stopping replay with measured execution-cost accounting."""

import csv
import platform
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from benchmark.data import measured_seconds
from benchmark.initialization import initial_indices
from benchmark.timing import now
from methods.baselines.hgt.io import read_json, write_json
from methods.gcrt.config import BUDGET, INITIAL, ROOT, ROLES, SEED, SEGMENT
from methods.gcrt.controller import StopController, block_scores, threshold
from methods.gcrt.data import forecast_task
from methods.gcrt.evaluation import stopping_row
from methods.gcrt.efficiency_report import main as report
from methods.gcrt.report import construction
from methods.gcrt.train import load_stopper

OUTPUT = ROOT / "retention_efficiency"
FAMILIES = {"cutin": "车辆切入", "brake": "前车紧急制动"}
METHODS = ("full_budget", "gcrt", "without_correction")
STOPPING_METHODS = ("gcrt", "without_correction")


def load_task(family, pool, sut):
    folder = ROOT / family
    path = folder / "trajectories" / f"pool_{pool}" / f"{sut}.json"
    trace = read_json(path)
    task = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    assert trace["seed"] == SEED and trace["policy"] == "segment"
    assert task["budget"] == BUDGET and task["initial"] == INITIAL
    assert len(task["labels"]) == len(task["records"]) == BUDGET
    selected = trace["selected_indices"]
    assert len(selected) == len(set(selected)) == BUDGET
    assert selected[:INITIAL] == initial_indices(1024, INITIAL, SEED)
    labels = np.asarray([q["risk"] > 0.5 for q in trace["queries"]])
    np.testing.assert_array_equal(labels, task["labels"])
    np.testing.assert_array_equal(np.cumsum(labels), trace["found"])
    np.testing.assert_allclose(
        measured_seconds(pool, sut, trace["scenario_ids"]), trace["execution_s"],
        rtol=0, atol=1e-12,
    )
    for step, state in enumerate(task["records"]):
        assert state["step"] == step and state["found"] == int(labels[:step].sum())
        assert state["pool_size"] == 1024
    return trace, task


def task_row(family, method, trace, prediction, gamma, preparation_s):
    """Charge only the used prefix, including the terminal stopping state."""
    if method == "full_budget":
        prediction = {
            "forecast": [], "prequery_s": [0.0] * BUDGET,
            "correction_s": [0.0] * BUDGET,
        }
    result = stopping_row(trace, prediction, gamma)
    used = result["queries"]
    timing = trace["timing"]
    common_s = preparation_s + timing["task_initialization_s"] + sum(
        timing["query_compute_s"][:used]
    )
    state_s = prediction_s = correction_s = decision_s = 0.0
    if method != "full_budget":
        state_s = sum(trace["state_recording_s"][:used:SEGMENT])
        if used < BUDGET:
            state_s += trace["state_recording_s"][used]
        prediction_s = sum(prediction["prequery_s"][:used])
        correction_s = sum(prediction["correction_s"][:used])
        decision_s = sum(
            r["compute_s"] for r in prediction["forecast"] if r["query"] <= used
        )
    overhead_s = state_s + prediction_s + correction_s + decision_s
    algorithm_s = common_s + overhead_s
    simulation_s = result["execution_s"]
    full_found = result["full_found"]
    full_common_s = preparation_s + timing["task_initialization_s"] + sum(
        timing["query_compute_s"]
    )
    full_simulation_s = sum(trace["execution_s"])
    full_total_s = full_common_s + full_simulation_s
    return {
        "task_id": f"{family}/pool_{trace['pool']}/{trace['sut']}",
        "family": family, "sut": trace["sut"], "pool": trace["pool"],
        "method": method, "threshold": gamma if method != "full_budget" else None,
        "queries": used, "query_reduction": 1 - used / BUDGET,
        "found": result["found"], "full_found": full_found,
        "retention": result["retention"],
        "target_met": 20 * result["found"] >= 19 * full_found if full_found else None,
        "missed": result["missed"], "stop_reason": result["stop_reason"],
        "preparation_s": preparation_s,
        "common_algorithm_s": common_s, "state_construction_s": state_s,
        "segment_prediction_s": prediction_s, "feedback_correction_s": correction_s,
        "stopping_decision_s": decision_s, "stopping_overhead_s": overhead_s,
        "terminal_decision_s": result["terminal_decision_s"],
        "simulation_s": simulation_s, "algorithm_s": algorithm_s,
        "total_s": simulation_s + algorithm_s,
        "simulation_saved_s": full_simulation_s - simulation_s,
        "common_compute_saved_s": full_common_s - common_s,
        "total_saved_s": full_total_s - simulation_s - algorithm_s,
        "total_reduction": 1 - (simulation_s + algorithm_s) / full_total_s,
        "selected_indices": result["selected_indices"],
    }


def summarize(rows):
    valid = [r for r in rows if r["retention"] is not None]
    total_found = sum(r["found"] for r in rows)
    full_found = sum(r["full_found"] for r in rows)

    def average(key):
        return float(np.mean([r[key] for r in rows]))

    def spread(key):
        return float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else 0.0

    total_s = sum(r["total_s"] for r in rows)
    saved_s = sum(r["total_saved_s"] for r in rows)
    return {
        "family": rows[0]["family"] if len({r["family"] for r in rows}) == 1 else "all",
        "method": rows[0]["method"], "tasks": len(rows),
        "valid_tasks": len(valid), "zero_discovery_tasks": len(rows) - len(valid),
        "queries_mean": average("queries"), "queries_sd": spread("queries"),
        "query_reduction": average("query_reduction"),
        "target_met_tasks": sum(r["target_met"] for r in valid),
        "target_rate": sum(r["target_met"] for r in valid) / len(valid) if valid else None,
        "minimum_retention": min(r["retention"] for r in valid) if valid else None,
        "mean_retention": float(np.mean([r["retention"] for r in valid])) if valid else None,
        "pooled_retention": total_found / full_found if full_found else None,
        "found_total": total_found, "full_found_total": full_found,
        "simulation_s": sum(r["simulation_s"] for r in rows),
        "algorithm_s": sum(r["algorithm_s"] for r in rows), "total_s": total_s,
        "task_total_mean_s": average("total_s"), "task_total_sd_s": spread("total_s"),
        "simulation_saved_s": sum(r["simulation_saved_s"] for r in rows),
        "stopping_overhead_s": sum(r["stopping_overhead_s"] for r in rows),
        "total_saved_s": saved_s,
        "total_reduction": saved_s / (total_s + saved_s),
        "simulation_covers_overhead": sum(r["simulation_saved_s"] for r in rows)
        > sum(r["stopping_overhead_s"] for r in rows),
        "all_targets_met": all(r["target_met"] for r in valid) if valid else None,
    }


def write_csv(path, rows):
    fields = [key for key in rows[0] if key != "selected_indices"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_results(rows, replays, calibrations):
    assert set(ROLES["calibration"]).isdisjoint(ROLES["evaluation"])
    fixed = {r["task_id"]: r for r in rows if r["method"] == "full_budget"}
    assert len(fixed) == 12 and len(rows) == 36 and len(replays) == 48
    for row in rows:
        reference = fixed[row["task_id"]]
        assert row["selected_indices"] == reference["selected_indices"][:row["queries"]]
        assert row["full_found"] == reference["full_found"]
        assert INITIAL <= row["queries"] <= BUDGET and row["queries"] % SEGMENT == 0
        assert np.isclose(row["total_s"], row["simulation_s"] + row["algorithm_s"])
        assert np.isclose(row["total_saved_s"], reference["total_s"] - row["total_s"])
        assert np.isclose(
            row["total_saved_s"], row["simulation_saved_s"]
            + row["common_compute_saved_s"] - row["stopping_overhead_s"],
        )
    for replay in replays:
        if replay["method"] == "without_correction":
            assert all(check["tilt"] == 0.0 for check in replay["forecast"])
    for family in FAMILIES:
        bindings = [calibrations[family][m]["weights_sha256"] for m in STOPPING_METHODS]
        assert len(set(bindings)) == 1
        for method in STOPPING_METHODS:
            gamma = calibrations[family][method]["threshold"]
            held = [r for r in replays if r["family"] == family
                    and r["method"] == method and r["role"] == "calibration"]
            assert len(held) == 6
            for replay in held:
                for check in replay["forecast"]:
                    if 20 * check["found"] < 19 * replay["full_found"]:
                        assert check["score"] <= gamma and not check["deterministic"]
    return {
        "passed": True, "evaluation_tasks": 12, "calibration_tasks": 12,
        "method_rows": len(rows), "stopping_replays": len(replays),
        "new_physical_calls": 0, "new_training_steps": 0,
        "same_predictor_weights": True, "calibration_tasks_disjoint": True,
        "prefixes_and_cost_sums_verified": True,
        "uncorrected_coefficients_zero": True,
        "unsafe_calibration_checkpoints_do_not_stop": True,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows, replays, calibrations, offline = [], [], {}, {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            folder = ROOT / family
            published = load_stopper(family, updated=True)
            network = published.network
            penalty = read_json(folder / "training.json")["penalty"]
            calibrations[family] = {}
            cal_wall_s = 0.0
            for role in ("calibration", "evaluation"):
                predictions = {method: [] for method in STOPPING_METHODS}
                for pool in (0, 1, 2):
                    preparation_s = sum(read_json(folder / "pool_cache" / f"pool_{pool}.json").values())
                    for sut in ROLES[role]:
                        trace, task = load_task(family, pool, sut)
                        if role == "calibration" and pool == 0 and sut == ROLES[role][0]:
                            warm = StopController(network, 1.0, penalty)
                            for state, label in zip(task["records"][:INITIAL], task["labels"]):
                                if state["step"] % SEGMENT == 0:
                                    warm.before_state(state)
                                warm.observe(float(label))
                            warm.evaluate_state(task["records"][INITIAL])
                            now()
                        if role == "evaluation":
                            rows.append(task_row(family, "full_budget", trace, None, 1.0, preparation_s))
                        for method in STOPPING_METHODS:
                            started = now()
                            prediction = forecast_task(
                                network, task, penalty,
                                correction=method == "gcrt", clock=now,
                            )
                            if role == "calibration":
                                cal_wall_s += now() - started
                            predictions[method].append(prediction)
                            gamma = calibrations[family][method]["threshold"] if role == "evaluation" else 1.0
                            if role == "evaluation":
                                rows.append(task_row(family, method, trace, prediction, gamma, preparation_s))
                            for check in prediction["forecast"]:
                                check.pop("remaining_log_probability")
                                check["threshold"] = gamma if role == "evaluation" else None
                                check["stop"] = bool(check["deterministic"] or check["score"] > gamma) if role == "evaluation" else None
                            replays.append({"family": family, "method": method, "role": role, **prediction})
                            print(f"REPLAY {family} {role} pool_{pool} {sut} {method}", flush=True)
                if role == "calibration":
                    for method in STOPPING_METHODS:
                        scores = block_scores(predictions[method])
                        calibrations[family][method] = {
                            "threshold": threshold(scores.values()), "controller_scores": scores,
                            "weights_sha256": published.contract["weights_sha256"],
                            "weights": f"results/gcrt/{family}/updated.pt",
                            "correction": method == "gcrt", "calibration_tasks": 6,
                            "rule": "max({0} union unsafe checkpoint scores); stop iff score > threshold",
                        }
            offline[family] = {
                "construction_s": sum(construction(family).values()),
                "published_update_s": read_json(folder / "update.json")["elapsed_s"],
                "calibration_replay_wall_s": cal_wall_s,
            }
    summaries = []
    for family in (*FAMILIES, "all"):
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method
                        and (family == "all" or r["family"] == family)]
            summaries.append(summarize(selected))
    config = {
        "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "seed": SEED, "budget": BUDGET, "initial": INITIAL, "segment": SEGMENT,
        "retention_target": 0.95, "pool_size": 1024, "pools": [0, 1, 2],
        "roles": {role: list(ROLES[role]) for role in ("calibration", "evaluation")},
        "predictor": "published updated.pt for both stopping methods; fixed during evaluation",
        "timing": {
            "metric": "reconstructed serial model-ready task execution time",
            "simulation": "archived measured per-scenario execution_s on selected prefix",
            "common_algorithm": "archived pool preparation + initialization + selected query_compute_s prefix",
            "stopping": "archived necessary snapshots + newly measured segment prediction, correction and decision prefix",
            "measurement_repetitions": 1, "gpu_synchronized": True,
            "offline_training_and_calibration_in_primary_total": False,
            "uncertainty": "sample standard deviation across tasks; no cross-seed repetition estimate",
        },
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "online_workers": 1, "torch_threads": torch.get_num_threads()},
        "archived_hardware": read_json(ROOT / "protocol.json")["hardware"],
        "new_physical_calls": 0, "new_training_steps": 0, "stopping_replays": len(replays),
        "offline_costs": offline,
    }
    audit = verify_results(rows, replays, calibrations)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "calibration.json", calibrations)
    write_json(OUTPUT / "replays.json", replays)
    write_json(OUTPUT / "task_metrics.json", rows)
    write_json(OUTPUT / "summary.json", summaries)
    write_json(OUTPUT / "validation.json", audit)
    write_csv(OUTPUT / "task_metrics.csv", rows)
    write_csv(OUTPUT / "summary.csv", summaries)
    report()


if __name__ == "__main__":
    main()
