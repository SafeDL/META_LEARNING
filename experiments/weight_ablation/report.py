"""Recompute E03 from native evidence, then summarize the registered ADS pairs."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from controllers.highway.catalog import FORMAL_TARGET_SUTS
from testing_methods.history_transfer.adaptation import adapted_online

from experiments.weight_ablation import run as study
from experiments import transfer_execution as execution
from experiments.budget.selectors import greedy_index
from experiments.collect_transfer_records import ApprovedCase, _check_checkpoint, _cost_row
from experiments.executions import ExecutionBank
from experiments.paired_statistics import json_ready, paired_summary
from experiments.parallel import METHOD_WORKERS, initialize_worker, prepare_processes
from experiments.record_audit import BankTruth, budget_record, prediction_record
from experiments.budget.report import write_csv
from experiments.scenarios import features
from experiments.storage import read_json, write_json
from experiments.transfer_prediction_audit import verify_forecasts, verify_weight_updates

METHODS = ("full", "frozen")
ESTIMANDS = ("nll", "recall_100", "unique_collisions", "discovery_auc")
FEW_SHOT = (5, 10, 20, 50)
COSTS = ("selector_wall_s", "selector_peak_cpu_bytes", "selector_peak_cuda_allocated_bytes",
         "paid_feedback_wall_s", "equivalent_execution_s", "adaptation_s", "selection_s")
_CONTEXT = None


def _initialize(context):
    global _CONTEXT
    initialize_worker()
    _CONTEXT = context


def _require_close(value, reference, name):
    actual, expected = np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    if (
        actual.shape != expected.shape or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError(f"{name} is incomplete or nonfinite")
    error = float(np.max(np.abs(actual - expected)))
    if error > study.TOLERANCE:
        raise ValueError(f"{name} differs by {error:.6g}")
    return error


def _audit_fits(truth, snapshots, records, margins, context):
    base = context["models"][0]
    raw = type(base)(base.dimension, deep=False).double().eval()
    with torch.no_grad():
        raw.mean.fill_(study.SETTINGS.online.raw_initial_mean)
    x = features(truth.candidates)
    errors = {}
    for branch, model, settings, parameter_key, fit_key in (
        ("learned", base, study.SETTINGS.online.learned, "target_parameters", "target_fit"),
        ("raw", raw, study.SETTINGS.online.raw, "physical_parameters", "physical_fit"),
    ):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        encoded = model.features(x).detach()
        errors[branch] = 0.0
        for snapshot in snapshots:
            n = snapshot["paid_queries"]
            if n == 0:
                if snapshot[fit_key] is not None:
                    raise ValueError("A zero-feedback snapshot records target adaptation")
                fitted = model
            else:
                indices = [r["index"] for r in records[:n]]
                labels = truth.labels[indices].tolist()
                fitted = adapted_online(
                    model, x[indices], labels, margins[:n], settings, encoded=encoded[indices],
                    implicit_evidence=False,
                )
                recorded_fit = snapshot[fit_key]
                if recorded_fit["paid_observations"] != n or recorded_fit["settings"] != settings.to_dict():
                    raise ValueError("Target adaptation used a different paid prefix or fit settings")
                _require_close(
                    [recorded_fit["objective"], recorded_fit["gradient_max"]],
                    [fitted.adaptation_diagnostics[p] for p in ("objective", "gradient_max")],
                    f"{branch} target fitting diagnostics",
                )
            errors[branch] = max(errors[branch], _require_close(
                [snapshot[parameter_key][p] for p in settings.parameters],
                [float(getattr(fitted, p).detach()) for p in settings.parameters],
                f"{branch} target parameters",
            ))
    return errors


def _read_checked(root, case, truth):
    path, receipt = execution._paths(root, case)
    saved = read_json(path)
    _check_checkpoint(receipt, saved, case.specification, truth, complete=True)
    if case.experiment == "prediction":
        rows = prediction_record(saved, case.specification, truth, method="full", seed=study.SEED)
    else:
        row = budget_record(saved, case.specification, truth, method="full", seed=study.SEED)
        rows = row["metrics"]
    return saved, rows, path, receipt


def _audit_mode(truth, mode, approved, context):
    bank = truth.bank
    original = study.sealed_specification(mode, bank, approved)
    source = study.INPUT if mode == "prediction" else study.protocol.OUTPUT
    original_result, original_rows, original_path, original_paid = _read_checked(source, original, truth)
    case = study.frozen_case(original, original_path, original_paid, context)
    approval = execution.approval_path(study.OUTPUT, case)
    if read_json(approval) != {"key": list(case.key), "specification": case.specification, "scope": "formal"}:
        raise ValueError("Frozen result requires its independent approval")
    location = Path(case.scenario) / case.sut_id / f"{mode}.json"
    status = read_json(study.OUTPUT / "status" / location)
    if status["status"] != "complete":
        raise ValueError(f"The frozen task is incomplete: {status}")
    bridge = study.OUTPUT / "compatibility" / location
    compatibility = read_json(bridge)
    if (
        compatibility["identity"] != {"control_sha256": case.specification["control_sha256"],
                                      "case_sha256": study.digest(case.specification)}
        or compatibility["paid_queries_replayed"] != study.protocol.BUDGET
        or compatibility["new_paid_queries"] != 0
        or not compatibility["selected_order_equal"]
        or set(compatibility["maximum_errors"]) != {
            "prequery_probability", "prequery_component_probabilities", "prequery_component_weights",
            "component_weights_after_query", "prequery_log_likelihood", "probabilities",
            "component_weights", "target_parameters", "physical_parameters",
        }
        or any(not 0 <= error <= study.TOLERANCE for error in compatibility["maximum_errors"].values())
    ):
        raise ValueError("The sealed control has no matching successful compatibility replay")
    return audit_paired_mode(truth, mode, context, _read_checked(study.OUTPUT, case, truth),
                             (original_result, original_rows, original_path, original_paid),
                             compatibility, bridge, approval)


def audit_paired_mode(truth, mode, context, frozen_run, control_run, compatibility, bridge, approval):
    bank = truth.bank
    frozen_result, frozen_rows, frozen_path, frozen_paid = frozen_run
    original_result, original_rows, original_path, original_paid = control_run
    initial_count = 200 if mode == "prediction" else study.protocol.INITIAL
    records = frozen_result["query_ledger"]
    if any(
        (a["index"], a["feedback"]) != (b["index"], b["feedback"])
        for a, b in zip(records[:initial_count], original_result["query_ledger"][:initial_count], strict=True)
    ):
        raise ValueError("The paired methods did not receive identical common feedback")
    frozen = frozen_result["result"]
    dynamic = original_result["result"]
    for result, update in ((dynamic, True), (frozen, False)):
        if (
            result["source_weight_updates"] is not update or result["target_feedback_updates"] is not True
            or result["feedback_used"] != "joint" or not result["explicit_source_components"]
            or result["conditional_inference_device"] != "cpu"
            or result["online_feature_precision"] != "float64"
            or any(q["conditioning_queries"] != i for i, q in enumerate(result["queries"]))
        ):
            raise ValueError("Task changed joint feedback, conditioning, precision or weight switches")
    weights = verify_weight_updates(frozen, records, context["sources"])
    snapshots = frozen["prediction_snapshots"]
    initial_error = _require_close(snapshots[0]["probabilities"],
                                   dynamic["prediction_snapshots"][0]["probabilities"], "K=0 forecasts")
    if mode == "prediction":
        for a, b in zip(snapshots, dynamic["prediction_snapshots"], strict=True):
            for key in ("target_parameters", "physical_parameters"):
                _require_close(list(a[key].values()), list(b[key].values()), "common-feedback target fit")
    else:
        if [s["paid_queries"] for s in snapshots] != [0, 20, 50, 100, 200]:
            raise ValueError("Missing active-query checkpoints")
        selected = frozen["selected_indices"]
        for snapshot in snapshots[1:-1]:
            count = snapshot["paid_queries"]
            remaining = set(range(len(bank.candidates))) - set(selected[:count])
            if greedy_index(bank.candidates, snapshot["probabilities"], remaining) != selected[count]:
                raise ValueError("The frozen acquisition disagrees with greedy collision ranking")
    assert weights["queries"] == study.protocol.BUDGET
    margins = truth.margins[frozen["selected_indices"]]
    forecasts = verify_forecasts(truth.candidates, snapshots, records, margins, method="full",
                                 models=context["models"], pooled=None, sources=context["sources"],
                                 settings=study.SETTINGS)
    fits = _audit_fits(truth, snapshots, records, margins, context)
    return (original_result, frozen_result), (original_rows, frozen_rows), {
        "status": "complete", "shared_feedback_verified": initial_count,
        "k0_probability_maximum_error": initial_error, "weights": weights, "forecasts": forecasts,
        "maximum_refitted_parameter_errors": fits, "compatibility": compatibility,
        "files": {study.relative(p): study.sha(p) for p in (
            bridge, approval, original_path, original_paid, frozen_path, frozen_paid,
        )},
    }


def audit_target(kind, sut, approved):
    bank = ExecutionBank(study.INPUT / "physical/target" / kind / sut)
    truth = BankTruth(bank)
    prediction, prediction_rows, pred_audit = _audit_mode(truth, "prediction", approved, _CONTEXT)
    budget, budget_rows, budget_audit = _audit_mode(truth, "budget", approved, _CONTEXT)
    return summarize_target(truth, prediction, prediction_rows, pred_audit, budget, budget_rows, budget_audit, study.OUTPUT)


def summarize_target(truth, prediction, prediction_rows, pred_audit, budget, budget_rows, budget_audit, root):
    bank = truth.bank
    kind, sut = bank.candidates[0].kind, bank.manifest["sut_id"]
    base = {"scene": kind, "target": sut, "seed": study.SEED}
    prediction_table, curve_table, cost_table = [], [], []
    paired = dict(base)
    for index, method in enumerate(METHODS):
        for row in prediction_rows[index]:
            prediction_table.append({**base, "method": method, "K": row["K"], "nll": row["nll"],
                                     "recall_100": row["recall_100"], "recall_undefined": row["recall_undefined"],
                                     "evaluation_collisions": row["evaluation_collisions"]})
        budget_row = budget_rows[index]
        curve_table.extend(
            {**base, "method": method, "query": q, "cumulative_collisions": value}
            for q, value in enumerate(budget_row["discovery_curve"], 1)
        )
        for metric in ESTIMANDS[:2]:
            values = [row[metric] for row in prediction_rows[index] if row["K"] in FEW_SHOT]
            paired[f"{method}_{metric}"] = float(np.mean(values)) if None not in values else None
        for metric in ESTIMANDS[2:]:
            paired[f"{method}_{metric}"] = budget_row[metric]
        paired[f"{method}_collision_recall"] = budget_row["collision_recall"]
        paired[f"{method}_collision_recall_undefined"] = (
            "no_pool_collisions" if budget_row["collision_recall"] is None else None
        )
        paired[f"{method}_first_collision_query"] = budget_row["first_collision_query"]
        paired[f"{method}_first_collision_right_censored"] = budget_row["no_collision_within_budget"]
        for mode, runs in zip(study.MODES, (prediction, budget), strict=True):
            saved = runs[index]
            result = saved["result"]
            cost = _cost_row(ApprovedCase(mode, sut, kind, study.SEED, method, None), saved)
            cost.update(adaptation_s=sum(q["adaptation_s"] for q in result["queries"]),
                        selection_s=sum(q["selection_s"] for q in result["queries"]))
            cost_table.append(cost)
            for column in COSTS:
                paired[f"{mode}_{method}_{column}"] = cost[column]
    for metric in ESTIMANDS:
        a, b = paired[f"full_{metric}"], paired[f"frozen_{metric}"]
        paired[f"delta_{metric}"] = None if a is None or b is None else a - b
    for mode in study.MODES:
        for column in COSTS:
            paired[f"{mode}_delta_{column}"] = paired[f"{mode}_full_{column}"] - paired[f"{mode}_frozen_{column}"]
    audit = {**base, "bank_snapshot": bank.snapshot_fingerprint, "native_records": len(truth.candidates),
             "prediction": pred_audit, "budget": budget_audit}
    write_json(root / "audits" / kind / f"{sut}.json", audit)
    print({**base, "status": "audited"}, flush=True)
    return paired, prediction_table, curve_table, cost_table, audit


def _summary(pairs, columns):
    values = np.full((len(study.TARGETS), len(study.SCENES), 1, 2, len(columns)), np.nan)
    for row in pairs:
        i, j = study.TARGETS.index(row["target"]), study.SCENES.index(row["scene"])
        for method_index, method in enumerate(METHODS):
            values[i, j, 0, method_index, :] = [row[f"{method}_{metric}"] for metric in columns]
    metadata = dict(ads_ids=study.TARGETS, ads_families=[s.family for s in FORMAL_TARGET_SUTS],
                    methods=METHODS, baseline="frozen", lower_better=[m == "nll" for m in columns],
                    replicates=10000, seed=1301)
    raw = {"overall": paired_summary(values, **metadata)}
    raw.update({scene: paired_summary(values[:, j:j + 1], **metadata)
                for j, scene in enumerate(study.SCENES)})
    return json_ready({
        "metrics": list(columns), "positive_gain": "frozen minus dynamic for NLL; dynamic minus frozen otherwise",
        "scopes": raw,
    })


def report():
    initialize_worker()
    if read_json(study.OUTPUT / "protocol.json") != study.declaration():
        raise ValueError("E03 driver or numerical implementation changed")
    proof, approved = study.input_receipts()
    prepare_processes()
    data = {}
    for kind in study.SCENES:
        bank = ExecutionBank(study.INPUT / "physical/target" / kind / study.TARGETS[0])
        context = study.load_context(kind, proof, study.sealed_specification("prediction", bank, approved))
        with ProcessPoolExecutor(max_workers=METHOD_WORKERS, mp_context=mp.get_context("spawn"),
                                 initializer=_initialize, initargs=(context,)) as pool:
            tasks = {pool.submit(audit_target, kind, sut, approved): sut for sut in study.TARGETS}
            for future in as_completed(tasks):
                data[kind, tasks[future]] = future.result()
    return summarize_experiment(data, study.OUTPUT / "reports",
                                {key: study.OUTPUT for key in data})


def summarize_experiment(data, directory, audit_roots):
    keys = [(kind, sut) for kind in study.SCENES for sut in study.TARGETS]
    pairs = [data[key][0] for key in keys]
    write_csv(directory / "paired_tasks.csv", pairs)
    for index, name in enumerate(("prediction_metrics", "discovery_curves", "costs"), 1):
        write_csv(directory / f"{name}.csv", [row for key in keys for row in data[key][index]])
    summary = _summary(pairs, ESTIMANDS)
    summary["cost_difference"] = "dynamic minus frozen; positive values are higher observed costs"
    summary["cost_scopes"] = {
        mode: _summary(
            [{**row, **{f"{method}_{cost}": row[f"{mode}_{method}_{cost}"]
                       for method in METHODS for cost in COSTS}} for row in pairs],
            COSTS,
        )["scopes"]
        for mode in study.MODES
    }
    write_json(directory / "paired_summary.json", summary)
    per_k = {}
    for k in study.CHECKPOINTS:
        k_pairs = [{**data[key][0], **{
            f'{row["method"]}_{metric}': row[metric]
            for row in data[key][1] if row["K"] == k for metric in ESTIMANDS[:2]
        }} for key in keys]
        per_k[str(k)] = _summary(k_pairs, ESTIMANDS[:2])
    write_json(directory / "prediction_by_k.json", per_k)
    files = ("paired_tasks.csv", "prediction_metrics.csv", "discovery_curves.csv", "costs.csv",
             "paired_summary.json", "prediction_by_k.json")
    audits = [data[key][4] for key in keys]
    complete = {
        "status": "complete", "frozen_trajectories": len(keys) * 2,
        "paired_targets": len(keys), "paired_prediction_checkpoints": len(keys) * len(study.CHECKPOINTS),
        "frozen_logical_queries": len(keys) * 2 * study.protocol.BUDGET,
        "compatibility_replayed_queries": len(keys) * 2 * study.protocol.BUDGET,
        "compatibility_new_paid_queries": 0, "new_physical_executions": 0,
        "native_truth_records": sum(a["native_records"] for a in audits),
        "max_k0_probability_error": max(a[m]["k0_probability_maximum_error"] for a in audits for m in study.MODES),
        "max_compatibility_error": max(max(a[m]["compatibility"]["maximum_errors"].values()) for a in audits for m in study.MODES),
        "max_frozen_weight_error": max(a[m]["weights"]["maximum_weight_error"] for a in audits for m in study.MODES),
        "max_frozen_forecast_error": max(a[m]["forecasts"]["maximum_probability_error"] for a in audits for m in study.MODES),
        "max_frozen_fit_error": max(max(a[m]["maximum_refitted_parameter_errors"].values()) for a in audits for m in study.MODES),
        "files": {name: study.sha(directory / name) for name in files},
        "audit_files": {f"{kind}/{sut}.json": study.sha(audit_roots[kind, sut] / "audits" / kind / f"{sut}.json")
                        for kind, sut in keys},
        "report_source_sha256": study.sha(Path(__file__)),
    }
    write_json(directory / "completion.json", complete)
    print({k: v for k, v in complete.items() if k not in ("files", "audit_files")}, flush=True)
    return complete


if __name__ == "__main__":
    report()
