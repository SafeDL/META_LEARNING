"""Independently score frozen traces and paid ledgers and report the matched contrasts."""

import csv
from collections import Counter
from dataclasses import asdict
from math import ceil, fsum
from statistics import median

from testing_methods.common.response import archive_fingerprints

from experiments.regression.fixed_inputs import AUC_WINDOW, H0_COUNTS, METHODS, NEW_WINDOW, VERSIONS, methods_for, query_directory, scene_root, version_directory
from experiments.regression.fixed_replay import truth_rows
from experiments.resume import paid_feedback_hash
from experiments.storage import read_json, write_json


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(previous, current, order, *, auc_window=AUC_WINDOW, new_window=NEW_WINDOW):
    if len(previous) != len(current) or len(set(order)) != len(order) or any(
            type(i) is not int or not 0 <= i < len(current) for i in order):
        raise ValueError("Invalid paired labels or repeated/out-of-suite paid query")
    total = sum(current)
    goal = ceil(9 * total / 10) if total else None
    new = {i for i, (old, now) in enumerate(zip(previous, current)) if now and not old}
    paid_collisions, q90, new_positions = 0, None, []
    counts = []
    for call, index in enumerate(order, 1):
        paid_collisions += current[index]
        if q90 is None and goal is not None and paid_collisions == goal:
            q90 = call
        if index in new:
            new_positions.append(call)
        counts.append(paid_collisions)
    if (total and q90 is None) or len(order) != max(auc_window, q90 or 0):
        raise ValueError("Paid order does not stop at the later of Q90 and the observation window")
    return {"total_collisions": total, "q90_target": goal, "q90": q90,
            "q90_fraction": q90 / len(current) if q90 is not None else None,
            "auc_500": fsum(counts[:auc_window]) / (auc_window * total) if total else None,
            "new_collisions": len(new), "new_in_first_100": sum(q <= new_window for q in new_positions),
            "first_new": next(iter(new_positions), None), "logical_queries": len(order),
            "cumulative_collisions": counts}


def audit_ledger(directory, result, truth, registration, method, history):
    rows = result["query_ledger"]
    spec = result["specification"]
    initial = registration["initial_indices"]
    order = [r["index"] for r in rows]
    expected_history = archive_fingerprints(history) if method != "birchler" else {}
    if (spec["method"] != method or spec["version"] != truth["version"] or spec["initial_indices"] != initial
            or spec["checkpoint_sha256"] != truth["checkpoint_sha256"]
            or spec["truth_sha256"] != truth["execution_sha256"] or spec["history"] != expected_history
            or result["paid_feedback_hash"] != paid_feedback_hash(rows) or result["new_physical_calls"] != 0
            or order[:len(initial)] != initial or order != result["result"]["selected_indices"]):
        raise ValueError("Replay specification, paid feedback digest or initial order changed")
    if method == "full" and (spec["source_snapshot"]["archives"] != expected_history
                             or spec["source_snapshot"]["candidates"] != registration["candidates"]):
        raise ValueError("Model sources are outside the admitted historical archive")
    if method == "sdc_scissor":
        from testing_methods.baselines.sdc_scissor.method import SETTINGS
        if spec.get("selector_settings") != SETTINGS or spec["source_snapshot"] is not None:
            raise ValueError("SDC-Scissor configuration differs from its registered comparison")
    if method == "spectre":
        from testing_methods.baselines.spectre.method import SETTINGS, order_digest, prioritize

        from experiments.schema import QueryFeedback, Scenario

        if spec.get("selector_settings") != SETTINGS or spec["source_snapshot"] is not None:
            raise ValueError("SPECTRE configuration differs from its registered comparison")
        candidates = [Scenario(**s) for s in registration["candidates"]]
        prefix = [(i, QueryFeedback.from_dict(truth["feedback"][i])) for i in initial]
        suffix = prioritize(candidates, history, prefix)
        if (sorted(initial + suffix) != list(range(len(candidates)))
                or result["result"].get("ranking_sha256") != order_digest(suffix)
                or order != (initial + suffix)[:len(order)]):
            raise ValueError("SPECTRE ranking cannot be reproduced from the permitted feedback")
    query_rows = result["result"]["queries"]
    if len(rows) != len(query_rows) or len(rows) != len(list(directory.glob("paid/*.json"))):
        raise ValueError("Replay charges do not match stored query records")
    for call, (row, query) in enumerate(zip(rows, query_rows), 1):
        index = row["index"]
        if type(index) is not int or not 0 <= index < len(truth["feedback"]):
            raise ValueError("Paid query lies outside the frozen suite")
        expected = truth["feedback"][index]
        if (row != read_json(directory / "paid" / f"{call:04d}.json") or row["call"] != call
                or row["purpose"] != ("initial" if call <= len(initial) else "target")
                or row["status"] != "complete" or row["account"] != "target"
                or row["scenario_id"] != expected["scenario_id"] or row["feedback"] != expected
                or query["query"] != call or query["index"] != index or query["feedback"] != expected):
            raise ValueError("Paid response has the wrong query, debit or sealed feedback")
    return order


def compare(rows, metrics=("q90", "auc_500", "new_in_first_100"), *, methods=METHODS):
    contrasts = []
    for (kind, version), by_method in rows.items():
        for control in methods[1:]:
            row = {"kind": kind, "version": version, "comparator": control}
            for key in metrics:
                a = by_method.get("full", {}).get(key)
                b = by_method.get(control, {}).get(key)
                if a is None or b is None:
                    delta, outcome = None, "undefined"
                else:
                    delta = a - b
                    better = delta < 0 if key == "q90" else delta > 0
                    outcome = "tie" if delta == 0 else ("win" if better else "loss")
                row[f"{key}_ours_minus_control"] = delta
                row[f"{key}_outcome"] = outcome
            contrasts.append(row)
    tally = {method: {key: dict(Counter(r[f"{key}_outcome"] for r in contrasts if r["comparator"] == method))
                      for key in metrics} for method in methods[1:]}
    return contrasts, tally


def version_counts(kind, version, labels, previous):
    added = removed = None
    total = sum(labels)
    if previous is not None:
        added = sum(b and not a for a, b in zip(previous, labels, strict=True))
        removed = sum(a and not b for a, b in zip(previous, labels, strict=True))
        if total - sum(previous) != added - removed:
            raise ValueError("Paired collision changes do not conserve the version totals")
    return {"kind": kind, "version": version, "total_collisions": total, "suite_size": len(labels),
            "collision_rate": total / len(labels), "new_collisions": added, "removed_collisions": removed}


def complete_version_table(policy):
    from experiments.regression.fixed_inputs import output_root
    from experiments.regression.run_fixed import prepare

    versions = []
    for kind in H0_COUNTS:
        bank, _, models, _ = prepare(kind, 1, policy=policy)
        registration = read_json(scene_root(kind, policy) / "registration.json")
        if set(models) != set(VERSIONS):
            raise ValueError("Driving comparison requires all five qualified versions")
        previous = None
        for version in VERSIONS:
            truth = truth_rows(kind, version, bank, registration, policy)
            labels = [r["collision"] for r in truth["feedback"]]
            versions.append(version_counts(kind, version, labels, previous))
            previous = labels
    write_csv(output_root(policy) / "reports/trajectory_versions.csv", versions)
    return versions


def report(kinds, *, policy="ppo"):
    from experiments.regression.fixed_inputs import BRAKING_KIND, model_root, output_root, paths_for_version, reference_root
    from experiments.regression.execution import response_archive
    from experiments.regression.preflight import DQN_CONTINUAL
    from experiments.regression.run_fixed import prepare

    kinds = list(kinds)
    methods = methods_for(policy)
    versions, tasks, costs, anomalies, recoveries = [], [], [], [], []
    by_task, curves, construction, archive_costs = {}, {}, [], []
    capability_checks = []
    for kind in kinds:
        root = scene_root(kind, policy)
        origin = "new" if policy != "ppo" or kind == BRAKING_KIND else "reused"
        reference_dir = reference_root(kind, policy)
        chain_path = reference_dir / ("chain.json" if policy == DQN_CONTINUAL else "random/chain.json")
        if not (root / "registration.json").exists() and (policy == "ppo" or not chain_path.exists()):
            status = root / "status.json"
            if not status.exists():
                status = chain_path
            anomalies.append({"kind": kind, "status": "reference_incomplete",
                              "detail": read_json(status) if status.exists() else None})
            continue
        bank, history, models, reference_rows = prepare(kind, 1, policy=policy)
        registration = read_json(root / "registration.json")
        reference = read_json(chain_path)
        attempted = list(models)[1:]
        if reference["status"] == "terminated_unqualified":
            attempted.append(reference["last_version"])
        receipts = {}
        for version in attempted:
            if policy == DQN_CONTINUAL:
                evidence = read_json(reference_dir / version / "qualification/course_evidence.json")
                receipt = {"capability": {"qualified": evidence["qualified"], "courses": evidence["courses"]}}
            else:
                receipt = read_json(paths_for_version(reference_dir, version)[1])
            receipts[version] = receipt
            capability_checks.append({"kind": kind, "version": version, **receipt["capability"]})
        if len(models) < len(VERSIONS):
            attempt = None
            if reference["status"] == "terminated_unqualified":
                rejected = reference["last_version"]
                failed = read_json(paths_for_version(reference_dir, rejected)[1])
                attempt = {"version": rejected, **{k: failed[k] for k in (
                    "checkpoint_sha256", "wall_s", "actual_environment_steps", "qualification_executions")}}
            elif reference["status"] == "terminated_development":
                rejected = reference["last_version"]
                failed = read_json(paths_for_version(reference_dir, rejected)[1])
                attempt = {"version": rejected, "endpoint_sha256": failed["interpolation"]["endpoint_sha256"],
                           "actual_environment_steps": failed["actual_environment_steps"],
                           "comparisons": failed["interpolation"]["comparisons"]}
            anomalies.append({"kind": kind, "status": reference["status"],
                              "last_qualified": next(reversed(models)),
                              "missing_versions": [v for v in VERSIONS if v not in models],
                              "missing_tasks": [v for v in VERSIONS[1:4] if v not in models],
                              "rejected_repair": attempt,
                              "failed_courses": reference.get("failed_courses", [])})
        if registration["candidates"] != [asdict(s) for s in bank.candidates]:
            raise ValueError("Report inputs differ from the frozen suite")
        reference_origins = [origin] * len(reference_rows)
        if policy == DQN_CONTINUAL:
            from experiments.protocol import ROOT
            reference_origins = ["new" if ROOT / entry["query_root"] == reference_dir else "reused"
                                 for entry in reference["rounds"] if "query_root" in entry]
            if len(reference_origins) != len(reference_rows):
                raise ValueError("Missing physical cost origin for a paid reference round")
        archive_costs.append({"kind": kind, "base_history_records": sum(len(a.scenarios) for a in history),
                              "base_history_origin": ("ancestor and confirmed base history reused" if policy == DQN_CONTINUAL
                                                      else origin),
                              "reference_paid_queries": [len(rows) for rows in reference_rows],
                              "reference_paid_physical_wall_s": [fsum(r["wall_s"] for r in rows)
                                                                  for rows in reference_rows],
                              "reference_cost_origin": reference_origins if policy == DQN_CONTINUAL else origin})
        previous = None
        for i, version in enumerate(VERSIONS):
            if version not in models:
                break
            directory = version_directory(kind, "truth", version, policy)
            if version != "h0" and len(list(directory.glob("*.json.gz"))) != len(bank.candidates):
                anomalies.append({"kind": kind, "version": version, "status": "physical_suite_incomplete"})
                previous = None
                continue
            truth = truth_rows(kind, version, bank, registration, policy)
            labels = [r["collision"] for r in truth["feedback"]]
            versions.append(version_counts(kind, version, labels, previous))
            repair = registration["repairs"][i - 1] if i else None
            repair_origin = origin if repair else None
            construction.append({"kind": kind, "version": version,
                                 **{k: truth[k] for k in ("physical_execution_s", "new_physical_executions",
                                                         "reused_physical_executions")},
                                 "interrupted_physical_attempts": len(list(directory.glob("*.interrupted.json"))),
                                 f"{policy}_repair": repair,
                                 f"{policy}_repair_origin": repair_origin})
            if 0 < i < 4 and previous is not None:
                by_method = by_task[kind, version] = {}
                for method in methods:
                    directory = query_directory(kind, version, method, policy)
                    path = directory / "result.json.gz"
                    failure = directory / "failure.json"
                    if not path.exists():
                        anomalies.append({"kind": kind, "version": version, "method": method,
                                          "status": "replay_incomplete",
                                          "detail": read_json(failure) if failure.exists() else None})
                        continue
                    saved = read_json(path)
                    order = audit_ledger(directory, saved, truth, registration, method, history)
                    if failure.exists():
                        recoveries.append(read_json(failure))
                    metrics = score(previous, labels, order)
                    curve = metrics.pop("cumulative_collisions")
                    curves.setdefault(kind, {}).setdefault(version, {})[method] = curve
                    tasks.append({"kind": kind, "version": version, "method": method, **metrics})
                    by_method[method] = metrics
                    model_result = saved["result"]
                    costs.append({"kind": kind, "version": version, "method": method,
                                  "logical_queries": len(order), "new_physical_calls": 0,
                                  "evidence_gradient": saved.get("evidence_gradient"),
                                  "cpu_threads": model_result.get("cpu_threads", 1),
                                  "ranking_s": model_result.get("ranking_s", sum(q.get("selection_s", 0)
                                                                                 for q in model_result["queries"])),
                                  "adaptation_s": sum(q.get("adaptation_s", 0) for q in model_result["queries"]),
                                  **model_result["resources"]})
                if not truth["collisions"]:
                    anomalies.append({"kind": kind, "version": version, "status": "zero_collision"})
            if 0 < i <= len(reference_rows):
                history.append(response_archive(f"paid_policy_{i}", reference_rows[i - 1], policy))
            previous = labels
    contrasts, tally = compare(by_task, methods=methods)
    saved_queries = {method: [] for method in methods[1:]}
    for contrast in contrasts:
        delta = contrast["q90_ours_minus_control"]
        contrast["q90_queries_saved"] = None if delta is None else -delta
        if delta is not None:
            saved_queries[contrast["comparator"]].append(-delta)
    root = output_root(policy) / "reports"
    root.mkdir(parents=True, exist_ok=True)
    for name, data in (("trajectory_versions", versions), ("query_metrics", tasks),
                       ("method_contrasts", contrasts), ("query_costs", costs)):
        if data:
            write_csv(root / f"{name}.csv", data)
    expected = len(H0_COUNTS) * 3 * len(methods)
    status = "passed" if len(tasks) == expected and len(versions) == len(H0_COUNTS) * len(VERSIONS) else "incomplete"
    response_models = []
    for kind in kinds:
        response_models.append({"kind": kind, "origin": "new" if policy != "ppo" or kind == BRAKING_KIND else "reused",
                                "training_s": {p.stem: read_json(p).get("training_s")
                                             for p in model_root(kind, policy).glob("*.json")},
                                "source_posteriors": {p.stem: read_json(p)
                                                      for p in (scene_root(kind, policy) / "sources").glob("*.json")}})
    initial_capability = None
    if policy != "ppo":
        from experiments.regression.preflight import preflight
        initial_capability = preflight(policy)["results"]["parent"]
    write_json(root / "audit.json", {"status": status, "sequences": len(tasks), "expected_sequences": expected,
                                     "outcomes": tally, "anomalies": anomalies, "recovered_errors": recoveries,
                                     "median_q90_queries_saved": {m: median(v) if v else None
                                                                  for m, v in saved_queries.items()},
                                     "construction_costs": construction,
                                     "initial_capability": initial_capability,
                                     "capability_checks": capability_checks,
                                     "historical_archive_costs": archive_costs,
                                     "response_model_costs": response_models,
                                     "logical_queries": sum(r["logical_queries"] for r in tasks)})
    curves_path = root / "recall_curves.json.gz"
    if not curves_path.exists() or read_json(curves_path) != curves:
        write_json(curves_path, curves)
    print("fixed suite audit", status, len(tasks), "/", expected, tally, flush=True)
    return status
