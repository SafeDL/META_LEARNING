"""Independently audit and summarize the public E02/E03 task matrices."""

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from experiments import transfer_execution as execution
from experiments.budget import audit as budget_audit
from experiments.budget import protocol
from experiments.budget.report import write_csv
from experiments.collect_transfer_records import ApprovedCase
from experiments.executions import ExecutionBank
from experiments.k_shot import validate_result
from experiments.parallel import initialize_worker, prepare_processes
from experiments.public_ads import extend
from experiments.public_ads.evaluate_e01 import ORIGINAL, OUTPUT, TARGETS
from experiments.record_audit import BankTruth, verify_metrics, _canonical_feedback
from experiments.resume import paid_feedback_hash
from experiments.scenarios import FORMAL_SCENARIOS
from experiments.storage import read_json, write_json
from experiments.transfer_prediction_audit import verify_forecasts, verify_weight_updates
from experiments.transfer_settings import native_settings
from experiments.weight_ablation.effectiveness import probability_scores


def audit_frozen(bank, truth, mode, context):
    target, kind = bank.manifest["sut_id"], bank.candidates[0].kind
    case = ApprovedCase(mode, target, kind, 11, "full", None)
    path, paid_path = execution._paths(extend.FROZEN, case)
    saved, paid = read_json(path), read_json(paid_path)
    case = ApprovedCase(*case.key, saved["specification"])
    records, result = saved["query_ledger"], saved["result"]
    indices = [r["index"] for r in records]
    if (len(indices) != 200 or len(set(indices)) != 200 or paid["status"] != "complete"
            or paid["query_ledger"] != records or paid["paid_feedback_hash"] != paid_feedback_hash(records)
            or case.specification["update_source_weights"] is not False
            or result["target_feedback_updates"] is not True
            or result["source_weight_updates"] is not False
            or paid["specification"]["task"] != case.specification
            or (mode == "budget" and saved["evaluation"]["candidate_collisions"] != truth.labels.tolist())):
        raise ValueError("Frozen control changed its adaptation, feedback or evaluator labels")
    execution._validate_output(case, result, records)
    for record in records:
        if _canonical_feedback(record["feedback"]) != truth.paid_feedback(record["index"], native=(mode == "budget")):
            raise ValueError("Frozen feedback differs from native trajectory")
    forecasts = verify_forecasts(bank.candidates, result["prediction_snapshots"], records,
        truth.margins[indices], method="full", models=context["models"],
        sources=context["sources"], pooled=None, settings=native_settings())
    weights = verify_weight_updates(result, records, context["sources"])
    if mode == "budget":
        verify_metrics(bank.candidates, truth.labels.tolist(), indices, result["queries"], records,
                       saved["metrics"], 20, domain=bank.domain)
    return saved, {"forecasts": forecasts, "weights": weights,
                   "result_sha256": protocol.sha(path), "paid_sha256": protocol.sha(paid_path)}


def audit_prediction(bank, truth, frozen, context):
    target, kind = bank.manifest["sut_id"], bank.candidates[0].kind
    path = OUTPUT / "k_shot" / target / kind / "11/full.json.gz"
    dynamic = read_json(path)
    split = dynamic["specification"]["split"]
    if (split != frozen["specification"]["split"]
            or [(r["index"], r["feedback"]) for r in dynamic["query_ledger"]]
            != [(r["index"], r["feedback"]) for r in frozen["query_ledger"]]
            or dynamic["specification"]["bank_snapshot"] != bank.snapshot_fingerprint):
        raise ValueError("Dynamic and frozen E01 use different paid feedback or heldout coordinates")
    indices = [r["index"] for r in dynamic["query_ledger"]]
    audits = {"forecasts": verify_forecasts(bank.candidates,
        dynamic["result"]["prediction_snapshots"], dynamic["query_ledger"], truth.margins[indices],
        method="full", models=context["models"], sources=context["sources"], pooled=None,
        settings=native_settings()),
        "weights": verify_weight_updates(dynamic["result"], dynamic["query_ledger"], context["sources"]),
        "result_sha256": protocol.sha(path)}
    scores, weight_rows = {}, []
    components = None
    for name, saved in (("dynamic", dynamic), ("frozen", frozen)):
        result = saved["result"]
        validate_result(result, split)
        actual = []
        for snapshot in result["prediction_snapshots"]:
            heldout_indices = split["evaluation_indices"]
            nll, brier = probability_scores(np.asarray(snapshot["probabilities"])[heldout_indices],
                                            truth.labels[heldout_indices])
            actual.append({"K": snapshot["paid_queries"], "nll": float(nll), "brier": float(brier)})
        for a, b in zip(actual, saved["metrics"], strict=True):
            np.testing.assert_allclose([a["nll"], a["brier"]], [b["nll"], b["brier"]], atol=1e-12, rtol=0)
        values = np.array([q["prequery_component_probabilities"] for q in result["queries"]])
        if components is None:
            components = values
        else:
            np.testing.assert_allclose(values, components, atol=1e-10, rtol=0)
        heldout = np.mean([[r["nll"], r["brier"]] for r in actual if r["K"] in (5, 10, 20, 50)], axis=0)
        sequential = probability_scores(np.array([q["prequery_probability"] for q in result["queries"]]), truth.labels[indices])
        scores[name] = dict(zip(("heldout_nll", "heldout_brier", "sequential_nll", "sequential_brier"),
                               map(float, (*heldout, *sequential))))
        for step, q in enumerate(result["queries"], 1):
            w = q["component_weights_after_query"]
            weight_rows.append(dict(target=target, scenario=kind, method=name, query=step,
                history_mass=sum(w[:-2]), learned_target_mass=w[-2], raw_target_mass=w[-1],
                **{component: weight for component, weight in zip(result["component_names"], w)}))
    return scores, weight_rows, audits


def audit_target(kind, target, stage):
    bank = extend.bank_for(kind, target)
    truth = BankTruth(bank)
    context = extend._CONTEXT
    if stage == "budget":
        from testing_methods.baselines.scenario_fuzz.graphs import graph
        from testing_methods.baselines.scenario_fuzz.model import batch_graphs
        from testing_methods.baselines.paired_system_active_evaluation.method import build_proxy
        from experiments.history_transfer_experiment import scenario_history
        import torch

        _, public = extend.baseline_inputs(bank, "scenario_fuzz")
        with torch.no_grad():
            probabilities = np.concatenate([public["classifier"].eval()(batch_graphs([
                graph(s) for s in bank.candidates[start:start + 73]])).sigmoid().numpy()
                for start in range(0, 2000, 73)])
        _, paired = extend.baseline_inputs(bank, "paired_system_active_evaluation")
        fresh, _ = build_proxy(bank.candidates, scenario_history(protocol.INPUT, kind))
        np.testing.assert_allclose(fresh, paired["proxy"], atol=1e-12, rtol=0)
        baseline = dict(status="complete", proxy_maximum_error=float(np.max(np.abs(fresh - paired["proxy"]))),
                        target_labels_used=False)
        context = {**context, "scenario_fuzz_probabilities": probabilities}
        saved, audits = {}, []
        for method in protocol.METHODS:
            if method == "random":
                with extend.public_paths():
                    saved[method] = budget_audit.derive_random(bank, truth)
                audits.append({"method": method, "new_paid_queries": 0, "reused_paid_queries": 200})
            else:
                inputs = extend.budget_inputs(bank, method)
                saved[method], audit = budget_audit.check_case(bank, truth, method, context,
                                                           root=extend.BUDGET, inputs=inputs)
                audits.append(audit)
        outcome = budget_audit.check_common_feedback(bank, truth, saved, audits, extend.BUDGET)
        outcome["result_sha256"] = {m: protocol.sha(extend.BUDGET / "budget" / target / kind / "11" / m / "result.json.gz")
                                     for m in protocol.METHODS}
        write_json(extend.BUDGET / "audits" / kind / f"{target}.json", outcome)
        write_json(extend.BUDGET / "audits" / kind / f"{target}_history.json", baseline)
    else:
        frozen, audit_p = audit_frozen(bank, truth, "prediction", context)
        scores, weights, audit_d = audit_prediction(bank, truth, frozen, context)
        frozen_budget, audit_b = audit_frozen(bank, truth, "budget", context)
        dynamic_budget = read_json(extend.BUDGET / "budget" / target / kind / "11/full/result.json.gz")
        if frozen_budget["result"]["selected_indices"][:20] != dynamic_budget["result"]["selected_indices"][:20]:
            raise ValueError("Budget ablation changed the common initialization")
        receipt = dict(status="complete", scenario=kind, target=target, scores=scores,
                       frozen_prediction=audit_p, dynamic_prediction=audit_d, frozen_budget=audit_b)
        write_json(extend.FROZEN / "audits" / kind / f"{target}.json", receipt)
        write_csv(extend.FROZEN / "audits" / kind / f"{target}_weights.csv", weights)
    print("audit complete", stage, kind, target, flush=True)


def audit(stage, workers=3, kinds=FORMAL_SCENARIOS, targets=TARGETS):
    prepare_processes()
    initialize_worker()
    for kind in kinds:
        context = extend.prepare_history(kind)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                 initializer=extend.initialize, initargs=(context,)) as pool:
            futures = [pool.submit(audit_target, kind, target, stage) for target in targets]
            for future in as_completed(futures):
                future.result()


def summarize_budget():
    from controllers.highway.catalog import FORMAL_TARGET_SUTS
    from experiments.public_ads.evaluate_e01 import _collision_labels

    tasks, curves, regions, costs, files = [], [], [], [], {}
    for kind in FORMAL_SCENARIOS:
        labels = []
        for sut in FORMAL_TARGET_SUTS:
            root = (protocol.ROOT / "results/dqn_continual_learning/k_shot_adaptation"
                    if sut.sut_id == "target_dqn" else ORIGINAL)
            labels.append(_collision_labels(ExecutionBank(root / "physical/target" / kind / sut.sut_id)))
        count = np.asarray(labels).sum(axis=0)
        masks = {"shared_original_eight_collision": count == 8,
                 "original_eight_disagreement": (count > 0) & (count < 8),
                 "public_specific_vs_original_eight": count == 0}
        for target in TARGETS:
            audit_path = extend.BUDGET / "audits" / kind / f"{target}.json"
            audit = read_json(audit_path)
            if audit["status"] != "complete" or len(audit["audits"]) != 7:
                raise ValueError("E02 task is not independently audited")
            files[str(audit_path.relative_to(protocol.ROOT))] = protocol.sha(audit_path)
            y = np.asarray(audit["labels"], dtype=bool)
            for method in protocol.METHODS:
                path = extend.BUDGET / "budget" / target / kind / "11" / method / "result.json.gz"
                saved = read_json(path)
                if protocol.sha(path) != audit["result_sha256"][method]:
                    raise ValueError("Public E02 result changed after independent audit")
                files[str(path.relative_to(protocol.ROOT))] = protocol.sha(path)
                indices = saved["result"]["selected_indices"]
                cumulative = np.cumsum(y[indices])
                tasks.append(dict(target=target, scenario=kind, method=method, C200=int(cumulative[-1]),
                    AUC=float(cumulative.mean()), collision_recall=float(cumulative[-1] / y.sum()) if y.any() else None,
                    pool_collisions=int(y.sum()), first_collision_query=saved["metrics"]["first_collision_query"]))
                curves.extend(dict(target=target, scenario=kind, method=method, query=q, collisions=int(c))
                              for q, c in enumerate(cumulative, 1))
                for region, mask in masks.items():
                    found = np.cumsum(y[indices] & mask[indices])
                    regions.append(dict(target=target, scenario=kind, method=method, region=region,
                        available_collisions=int((y & mask).sum()), C200=int(found[-1]), AUC=float(found.mean())))
                resource = saved.get("extraction_resources", saved["result"].get("resources"))
                costs.append(dict(target=target, scenario=kind, method=method,
                    cost_scope="curve_extraction" if method == "random" else "selector_including_query_waits",
                    logical_queries=200, new_paid_reveals=0 if method == "random" else 200,
                    reused_paid_reveals=200 if method == "random" else 0, new_physical_executions=0,
                    equivalent_execution_s=saved["metrics"]["equivalent_execution_s"],
                    selection_s=saved["metrics"]["selection_s"], adaptation_s=saved["metrics"]["adaptation_s"],
                    rss_compute_s=saved["metrics"]["rss_compute_s"],
                    **resource))
    if len(tasks) != 84 or len(curves) != 16800:
        raise ValueError("Incomplete public E02 matrix")
    directory = extend.BUDGET / "reports"
    for name, rows in (("task_results", tasks), ("discovery_curves", curves), ("response_regions", regions), ("costs", costs)):
        write_csv(directory / f"{name}.csv", rows)
    summary = {"status": "complete", "tasks": 84, "curve_rows": 16800, "new_physical_executions": 0,
               "logical_queries": 16800, "reused_e01_queries": 2400, "new_paid_reveals": 14400,
               "aggregation": "equal four scenes within ADS; public cohort separate from original eight",
               "per_ads": {}, "files": files, "regions": {},
               "protocol_sha256": protocol.sha(extend.BUDGET / "protocol.json"),
               "tables_sha256": {p.name: protocol.sha(p) for p in directory.glob("*.csv")}}
    for target in (*TARGETS, "macro"):
        selected = [r for r in tasks if target == "macro" or r["target"] == target]
        summary["per_ads"][target] = descriptive(selected)
    for region in masks:
        summary["regions"][region] = descriptive([r for r in regions if r["region"] == region])
    write_json(directory / "summary.json", summary)
    return summary


def descriptive(rows):
    lookup = {(r["target"], r["scenario"], r["method"]): r for r in rows}
    tasks = sorted({(r["target"], r["scenario"]) for r in rows})
    means = {m: {key: float(np.mean([lookup[t, s, m][key] for t, s in tasks])) for key in ("C200", "AUC")}
             for m in protocol.METHODS}
    contrasts = {}
    for method in protocol.METHODS[1:]:
        difference = np.array([[lookup[t, s, "full"][key] - lookup[t, s, method][key]
                                for key in ("C200", "AUC")] for t, s in tasks])
        contrasts[method] = dict(gain=difference.mean(axis=0).tolist(),
            wins=(difference > 1e-10).sum(axis=0).tolist(), ties=(np.abs(difference) <= 1e-10).sum(axis=0).tolist(),
            losses=(difference < -1e-10).sum(axis=0).tolist())
    return dict(metric_order=["C200", "AUC"], methods=means, ours_minus=contrasts)


def summarize_frozen():
    rows, weights, files, budget, costs = [], [], {}, [], []
    import csv
    for kind in FORMAL_SCENARIOS:
        for target in TARGETS:
            path = extend.FROZEN / "audits" / kind / f"{target}.json"
            saved = read_json(path)
            if saved["status"] != "complete":
                raise ValueError("E03 pair has no passing audit")
            files[str(path.relative_to(protocol.ROOT))] = protocol.sha(path)
            row = dict(target=target, scenario=kind)
            for method, scores in saved["scores"].items():
                row.update({f"{method}_{key}": value for key, value in scores.items()})
            rows.append(row)
            with path.with_name(f"{target}_weights.csv").open(encoding="utf-8", newline="") as stream:
                weights.extend(csv.DictReader(stream))
            for name, root in (("dynamic", extend.BUDGET), ("frozen", extend.FROZEN)):
                result = read_json(root / "budget" / target / kind / "11/full/result.json.gz")
                budget.append(dict(target=target, scenario=kind, method=name,
                    C200=result["metrics"]["unique_collisions"], AUC=result["metrics"]["discovery_auc"]))
                costs.append(dict(target=target, scenario=kind, mode="budget", method=name,
                                  **result["result"]["resources"]))
            for name, root in (("dynamic", OUTPUT), ("frozen", extend.FROZEN)):
                result = read_json(root / "k_shot" / target / kind / "11/full.json.gz")
                costs.append(dict(target=target, scenario=kind, mode="prediction", method=name,
                                  **result["result"]["resources"]))
    scopes = {}
    metrics = ("heldout_nll", "heldout_brier", "sequential_nll", "sequential_brier")
    for target in (*TARGETS, "macro"):
        selected = [r for r in rows if target == "macro" or r["target"] == target]
        scope = {}
        for key in metrics:
            delta = np.array([r[f"dynamic_{key}"] - r[f"frozen_{key}"] for r in selected])
            scope[key] = dict(dynamic=float(np.mean([r[f"dynamic_{key}"] for r in selected])),
                frozen=float(np.mean([r[f"frozen_{key}"] for r in selected])),
                dynamic_minus_frozen=float(delta.mean()), lower_equal_higher=[int((delta < -1e-12).sum()),
                int((np.abs(delta) <= 1e-12).sum()), int((delta > 1e-12).sum())])
        scopes[target] = scope
    directory = extend.FROZEN / "reports"
    for name, values in (("paired_scores", rows), ("weight_trajectories", weights),
                         ("budget_comparison", budget), ("costs", costs)):
        write_csv(directory / f"{name}.csv", values)
    summary = dict(status="complete", prediction_pairs=12, budget_pairs=12, new_physical_executions=0,
                   scopes=scopes, files=files, difference="dynamic minus frozen; lower scores are better",
                   protocol_sha256=protocol.sha(extend.FROZEN / "protocol.json"),
                   tables_sha256={p.name: protocol.sha(p) for p in directory.glob("*.csv")})
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit-budget", "audit-frozen", "budget", "frozen"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--kinds", nargs="+", choices=FORMAL_SCENARIOS, default=FORMAL_SCENARIOS)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    args = parser.parse_args()
    if args.stage.startswith("audit-"):
        audit(args.stage[6:], args.workers, args.kinds, args.targets)
    else:
        print(summarize_budget() if args.stage == "budget" else summarize_frozen())


if __name__ == "__main__":
    main()
