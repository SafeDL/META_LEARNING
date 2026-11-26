"""Recompute the revised DQN results and combine them with the seven sealed ADSs."""

import argparse
import csv
import json
from pathlib import Path

from experiments.budget import audit as budget_audit
from experiments.budget import protocol as budget_protocol
from experiments import collect_transfer_records as reader
from experiments.budget import report as report_budget_discovery
from experiments import report_history_transfer
from experiments.weight_ablation import rss_ablation as report_rss_ablation
from experiments.weight_ablation import report as report_weight_ablation
from experiments.weight_ablation import effectiveness as report_weight_effectiveness
from experiments.weight_ablation import run as study
from experiments import transfer_execution as execution
from experiments.regression.dqn_evaluation import PREDICTION, TARGET, admission, revised_budget_case, revised_prediction_case, study_inputs
from experiments.regression.dqn_learning import STUDY
from experiments.history_transfer_experiment import PREDICTION_METHODS, freeze
from experiments.parallel import initialize_worker
from experiments.record_audit import BankTruth
from experiments.resume import ResumableLedger
from experiments.storage import read_json, write_json

BUDGET = STUDY / "budget_discovery"
ABLATION = STUDY / "weight_adaptation"
ORIGINAL_PREDICTION = study.PREDICTION_REPORT


def check_approval(root, case):
    path = execution.approval_path(root, case)
    if read_json(path) != {"key": list(case.key), "specification": case.specification, "scope": "formal"}:
        raise ValueError(f"Task admission differs from its reconstructed inputs: {case.key}")
    return path


def audit_prediction(kind):
    bank, _, old_case, context = study_inputs(kind, "prediction")
    cases = [revised_prediction_case(bank, old_case, context, method, False)[0] for method in PREDICTION_METHODS]
    for case in cases:
        check_approval(PREDICTION, case)
        reader._check_case_specification(case)
    inventory = reader.inventory(cases, PREDICTION)
    if any(row["status"] != "complete" for row in inventory):
        raise ValueError("Missing revised E01 prediction result")
    checks = {"settings": study.SETTINGS, "prediction_inputs": {kind: context}}
    audited = reader.audit_records(cases, PREDICTION, PREDICTION / "physical/target", inventory, checks, "formal")
    attempts = report_history_transfer.attempt_costs(PREDICTION, cases)
    return freeze(PREDICTION / "audits" / f"{kind}.json", {"records": audited, "attempts": attempts})


def audit_budget(kind):
    bank, original, _, context = study_inputs(kind, "budget")
    baseline, context["scenario_fuzz_probabilities"] = budget_audit.verify_historical_baselines(bank, context)
    truth = BankTruth(bank)
    saved, audits = {}, []
    for method in budget_protocol.METHODS:
        if method == "random":
            control = reader.ApprovedCase("prediction", TARGET, kind, study.SEED, "target_only", None)
            path = BUDGET / "budget" / TARGET / kind / str(study.SEED) / method / "result.json.gz"
            if not path.exists():
                raise ValueError("The revised random replay is missing")
            saved[method] = budget_audit.derive_random(bank, truth, destination=path,
                                                       source=execution._paths(PREDICTION, control))
            audits.append({"method": method, "queries": 200, "source_receipt_verified": True, "new_paid_reveals": 0})
        else:
            inputs = revised_budget_case(bank, original, context, method, False)
            saved[method], checked = budget_audit.check_case(bank, truth, method, context, root=BUDGET, inputs=inputs)
            audits.append(checked)
    budget_audit.check_common_feedback(bank, truth, saved, audits, BUDGET)
    return freeze(BUDGET / "audits" / kind / "historical_baselines.json", baseline)


def audit_ablation(kind):
    bank, original, old_case, context = study_inputs(kind, "prediction")
    truth = BankTruth(bank)
    paired = []
    for mode, root in (("prediction", PREDICTION), ("budget", BUDGET)):
        def make(frozen):
            if mode == "prediction":
                return revised_prediction_case(bank, old_case, context, "full", frozen)[0]
            return revised_budget_case(bank, original, context, "full", frozen)[0]

        control, frozen = make(False), make(True)
        check_approval(root, control)
        approval = check_approval(ABLATION, frozen)
        control_run = report_weight_ablation._read_checked(root, control, truth)
        frozen_run = report_weight_ablation._read_checked(ABLATION, frozen, truth)
        saved, _, result_path, paid_path = control_run
        identity = {"control_sha256": {study.relative(p): study.sha(p) for p in (result_path, paid_path)},
                    "case_sha256": study.digest(frozen.specification)}
        bridge = ABLATION / "compatibility" / kind / TARGET / f"{mode}.json"
        if not bridge.exists():
            ledger = ResumableLedger(bank.candidates, study.protocol.BUDGET, None, checkpoint_path=paid_path,
                                     specification=control.specification, paid_execution=lambda i: bank(bank.candidates[i]),
                                     domain=bank.domain, allow_registration=False, joint_feedback=True)
            selector, public = study.selector_inputs(control, bank.candidates, context, dynamic=True)
            replay = execution.run_isolated(selector, public, ledger, wall_limit_s=study.LIMITS["wall_s"],
                                            memory_limit_bytes=study.LIMITS["child_memory_bytes"])
            errors = study.compare_replay(replay, saved["result"])
            freeze(bridge, {"identity": identity, "paid_queries_replayed": ledger.replayed_count,
                            "new_paid_queries": 0, "checkpoints": len(replay["prediction_snapshots"]),
                            "selected_order_equal": True, "maximum_errors": errors})
        receipt = read_json(bridge)
        if (receipt["identity"] != identity or receipt["paid_queries_replayed"] != study.protocol.BUDGET
                or receipt["new_paid_queries"] or not receipt["selected_order_equal"]
                or max(receipt["maximum_errors"].values()) > study.TOLERANCE):
            raise ValueError("Revised dynamic control failed compatibility replay")
        paired.extend(report_weight_ablation.audit_paired_mode(
            truth, mode, context, frozen_run, control_run, receipt, bridge, approval))
    tables = report_weight_ablation.summarize_target(truth, *paired, ABLATION)
    return freeze(ABLATION / "audits" / kind / "dqn_tables.json", tables[:4])


def table(path):
    def scalar(value):
        if value in ("", "True", "False"):
            return {"": None, "True": True, "False": False}[value]
        try:
            return json.loads(value)
        except ValueError:
            return value
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: scalar(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def combine_prediction():
    records = read_json(ORIGINAL_PREDICTION / "audited_records.json")
    cases = [read_json(PREDICTION / "audits" / f"{k}.json") for k in study.SCENES]
    for key, rows in list(records.items()):
        if not isinstance(rows, list):
            continue
        retained = [r for r in rows if (r["key"][1] if "key" in r else r["sut_id"]) != TARGET]
        merged = retained + [r for case in cases for r in case["records"][key]]
        if len(merged) != len(rows):
            raise ValueError(f"E01 incomplete replacement for {key}")
        records[key] = merged
    records["reader_sha256"] = study.sha(Path(reader.__file__))
    if len(records["rows"]) != 672 or len({tuple(x["key"]) for x in records["approved_cases"]}) != 96:
        raise ValueError("Incomplete mixed E01 matrix")
    directory = PREDICTION / "reports/main_results/prediction"
    directory.mkdir(parents=True, exist_ok=True)
    reader.export_source_data(directory, records)
    stats = read_json(study.INPUT / "reports/statistics_protocol.json")
    write_json(directory / "prediction_summary.json", report_history_transfer.summarize_audited(records, stats, "prediction"))
    old = read_json(ORIGINAL_PREDICTION / "completion.json")
    protocol = {**old["protocol"], "approved_cases": records["approved_cases"],
                "dqn_checkpoint_sha256": study.sha(admission()),
                "dqn_audits": {k: study.sha(PREDICTION / "audits" / f"{k}.json") for k in study.SCENES},
                "original_report_sha256": study.sha(ORIGINAL_PREDICTION / "completion.json")}
    write_json(directory / "protocol.json", protocol)
    previous = read_json(ORIGINAL_PREDICTION / "execution_attempts.json")
    attempts = [r for r in previous if r["sut_id"] != TARGET] + [r for case in cases for r in case["attempts"]]
    write_json(directory / "execution_attempts.json", attempts)
    report_history_transfer._csv(directory / "execution_attempts.csv", attempts)
    write_json(directory / "shared_source_construction.json", read_json(ORIGINAL_PREDICTION / "shared_source_construction.json"))
    completion = {**old, "protocol": protocol, "execution_attempts": len(attempts),
                  "all_attempt_wall_s": sum(a["attempt_wall_s"] for a in attempts),
                  "successful_logical_queries": sum(c["successful_logical_queries"] for c in records["costs"]),
                  "physical_bank_successful_calls": sum(b["successful_calls"] for b in records["physical_banks"])}
    completion["files"] = {name: study.sha(directory / name) for name in old["files"]}
    write_json(directory / "completion.json", completion)


def combine_budget():
    def directory(target, kind, method):
        if target == TARGET:
            return BUDGET / "budget" / target / kind / str(study.SEED) / method
        return budget_audit.result_directory(target, kind, method)
    checked = [read_json((BUDGET if target == TARGET else budget_protocol.OUTPUT) / "audits" / kind / f"{target}.json")
               for kind in study.SCENES for target in study.TARGETS]
    baselines = [read_json(BUDGET / "audits" / k / "historical_baselines.json") for k in study.SCENES]
    return report_budget_discovery.summarize(checked, baselines, result_path=directory, destination=BUDGET / "reports")


def combine_ablation():
    source = study.OUTPUT / "reports"
    old = read_json(source / "completion.json")
    names = ("paired_tasks", "prediction_metrics", "discovery_curves", "costs")
    for filename, digest in old["files"].items():
        if study.sha(source / filename) != digest:
            raise ValueError("Sealed original E03 report changed")
    originals = [table(source / f"{name}.csv") for name in names]
    data, scores, roots = {}, {}, {}
    for scene in study.SCENES:
        for target in study.TARGETS:
            key = scene, target
            root = ABLATION if target == TARGET else study.OUTPUT
            roots[key] = root
            audit = root / "audits" / scene / f"{target}.json"
            digest = study.sha(audit)
            if target == TARGET:
                tables = read_json(root / "audits" / scene / "dqn_tables.json")
                scores[key] = report_weight_effectiveness.paired_scores(scene, target, digest, frozen_root=root,
                                                                      prediction_root=PREDICTION, budget_root=BUDGET)
            else:
                if digest != old["audit_files"][f"{scene}/{target}.json"]:
                    raise ValueError("Unchanged target does not match its original E03 audit")
                tables = [[r for r in rows if r.get("target", r.get("sut_id")) == target
                           and r.get("scene", r.get("scenario")) == scene] for rows in originals]
                (tables[0],) = tables[0]
                scores[key] = report_weight_effectiveness.paired_scores(scene, target, digest)
            data[key] = (*tables, read_json(audit))
    directory = ABLATION / "reports"
    report_weight_ablation.summarize_experiment(data, directory, roots)
    report_weight_effectiveness.summarize_scores(scores, study.sha(directory / "completion.json"), directory)
    rss = report_rss_ablation.report(directory, {TARGET: (ABLATION, PREDICTION, BUDGET)})
    summarize_exploratory_scores(scores, rss, directory)


def summarize_exploratory_scores(scores, rss, directory):
    import numpy as np
    from experiments.weight_ablation.report import FEW_SHOT, METHODS
    from experiments.weight_ablation.effectiveness import sealed_result

    metrics = ("heldout_brier", "heldout_brier_collision", "heldout_brier_noncollision",
               "sequential_brier", "sequential_nll", "rss_tail_hits")
    original = {(r["scene"], r["target"]): r for r in table(
        study.OUTPUT / "reports/weight_effectiveness/paired_tasks.csv")}
    rows = {}
    for (scene, target), scored in scores.items():
        revised = target == TARGET
        frozen = ABLATION if revised else study.OUTPUT
        prediction = PREDICTION if revised else study.INPUT
        budget = BUDGET if revised else study.protocol.OUTPUT
        audit = read_json(frozen / "audits" / scene / f"{target}.json")
        control = sealed_result(budget, "budget", scene, target, audit)
        labels = np.asarray(control["evaluation"]["candidate_collisions"], dtype=bool)
        pair, observations, _ = rss[scene, target]
        row = dict(scene=scene, target=target, seed=study.SEED, evaluation_collisions=scored["evaluation_collisions"])
        for method, root in zip(METHODS, (prediction, frozen), strict=True):
            saved = sealed_result(root, "prediction", scene, target, audit)
            indices = saved["specification"]["split"]["evaluation_indices"]
            y = labels[indices]
            errors = np.array([np.square(np.array(s["probabilities"])[indices] - y)
                               for s in saved["result"]["prediction_snapshots"] if s["paid_queries"] in FEW_SHOT])
            for metric in ("heldout_brier", "sequential_brier", "sequential_nll"):
                row[f"{method}_{metric}"] = scored[f"{method}_{metric}"]
            row[f"{method}_heldout_brier_collision"] = float(errors[:, y].mean())
            row[f"{method}_heldout_brier_noncollision"] = float(errors[:, ~y].mean())
            row[f"{method}_rss_tail_hits"] = sum(r["rss_margin"] <= pair["threshold"] for r in observations
                                                if r["method"] == method and r["query"] > study.protocol.INITIAL)
            if not revised:
                keys = [f"{method}_{metric}" for metric in metrics]
                np.testing.assert_allclose([row[key] for key in keys], [original[scene, target][key] for key in keys],
                                           rtol=0, atol=1e-12)
        rows[scene, target] = row
    destination = directory / "weight_effectiveness"
    report_budget_discovery.write_csv(destination / "paired_tasks.csv", list(rows.values()))
    scopes = report_weight_effectiveness.score_scopes(rows, metrics)
    write_json(destination / "paired_summary.json", {"status": "complete", "exploratory": True,
        "metrics": metrics, "difference": "dynamic minus frozen", "few_shot_checkpoints": FEW_SHOT,
        "sequential_queries": [1, study.protocol.BUDGET], "rss_queries": [study.protocol.INITIAL + 1, study.protocol.BUDGET],
        "tail_rank": report_rss_ablation.TAIL_RANK, "bootstrap": read_json(directory / "probability_summary.json")["bootstrap"],
        "original_completion_sha256": study.sha(directory / "completion.json"),
        "rss_completion_sha256": study.sha(directory / "rss_discovery/completion.json"),
        "paired_tasks_sha256": study.sha(destination / "paired_tasks.csv"), "scopes": scopes})


def combined_receipts():
    """Verify the complete eight-ADS tables and their links to the new DQN."""
    prediction = PREDICTION / "reports/main_results/prediction/completion.json"
    budget = BUDGET / "reports/completion.json"
    ablation = ABLATION / "reports/completion.json"
    rss = ABLATION / "reports/rss_discovery/completion.json"
    checks = {
        prediction: {"cases": 96, "metric_rows": 672},
        budget: {"cases": 224, "logical_budget_queries": 44800},
        ablation: {"paired_targets": 32, "frozen_trajectories": 64},
        rss: {"paired_tasks": 32, "native_trajectories_recomputed": 64000},
    }
    for path, counts in checks.items():
        record = read_json(path)
        if record["status"] != "complete" or any(record[key] != value for key, value in counts.items()):
            raise ValueError(f"Incomplete paper report: {path}")
        for name, digest in record["files"].items():
            if study.sha(path.parent / name) != digest:
                raise ValueError(f"Paper report differs from its audit: {path.parent / name}")
    identity = study.sha(admission())
    protocol = read_json(prediction)["protocol"]
    if (protocol["dqn_checkpoint_sha256"] != identity or protocol["dqn_audits"] !=
            {k: study.sha(PREDICTION / "audits" / f"{k}.json") for k in study.SCENES}):
        raise ValueError("E01 and the admitted DQN refer to different evidence")
    dqn_tasks = {(r["scenario"], r["method"]): r for r in table(budget.parent / "task_results.csv")
                 if r["sut_id"] == TARGET}
    ablation_files = read_json(ablation)["audit_files"]
    for scene in study.SCENES:
        bank = PREDICTION / "physical/target" / scene / TARGET
        snapshot = read_json(bank / "quality.json")["bank_sha256"]
        registration = read_json(STUDY / "policy_regression" / scene / "registration.json")
        budget_check = read_json(BUDGET / "audits" / scene / f"{TARGET}.json")
        if (budget_check["bank_snapshot"] != snapshot or registration["bank_sha256"] != snapshot
                or registration["checkpoints"]["h0"]["sha256"] != identity):
            raise ValueError("E01, E02 and the regression starting policy use different target banks")
        for method in budget_protocol.METHODS:
            saved = read_json(BUDGET / "budget" / TARGET / scene / str(study.SEED) / method / "result.json.gz")
            row = dqn_tasks[scene, method]
            paid = saved["query_ledger"]
            observed = [q["feedback"]["collision"] for q in paid]
            if (saved["specification"]["bank_contract"]["source_files"]["checkpoint"] != identity
                    or observed != [budget_check["labels"][q["index"]] for q in paid]
                    or sum(observed) != row["C200"]
                    or saved["metrics"]["discovery_auc"] != row["AUC"]):
                raise ValueError(f"E02 paper result differs from the revised paid observations: {scene}/{method}")
        if ablation_files[f"{scene}/{TARGET}.json"] != study.sha(ABLATION / "audits" / scene / f"{TARGET}.json"):
            raise ValueError("Revised E03 target differs from its paired audit")
    scores = ABLATION / "reports/probability_summary.json"
    supplementary = ABLATION / "reports/weight_effectiveness/paired_summary.json"
    for path, name, field in ((scores, "probability_scores.csv", "probability_scores_sha256"),
                              (supplementary, "paired_tasks.csv", "paired_tasks_sha256")):
        record = read_json(path)
        if (record["status"] != "complete" or record["original_completion_sha256"] != study.sha(ablation)
                or record[field] != study.sha(path.parent / name)):
            raise ValueError(f"Unverified E03 probability scores: {path}")
    if read_json(supplementary)["rss_completion_sha256"] != study.sha(rss):
        raise ValueError("E03 supplementary and native RSS reports differ")
    return {p.relative_to(study.ROOT).as_posix(): study.sha(p) for p in (*checks, scores, supplementary)}


def publish():
    from tempfile import TemporaryDirectory

    from experiments.regression.dqn_evaluation import REVISED
    from experiments.regression.fixed_inputs import SUITE_SIZE, VERSIONS, methods_for, query_directory
    from experiments.paper_results import PUBLICATION
    from experiments.plotting.budget_discovery import render as plot_budget
    from experiments.plotting.k_shot import render as plot_prediction
    from experiments.plotting.weight_effectiveness import plot as plot_ablation

    root = STUDY / "policy_regression"
    methods = methods_for("dqn_continual")
    expected = 12 * len(methods)
    audit_path = root / "reports/audit.json"
    if not audit_path.exists():
        raise ValueError("Publication requires the complete matched DQN replays")
    audit = read_json(audit_path)
    if (audit["status"] != "passed" or audit["sequences"] != expected or audit["anomalies"]
            or len(audit["capability_checks"]) != 16 or not all(r["qualified"] for r in audit["capability_checks"])):
        raise ValueError("The four DQN version chains and matched replays must pass their audit")
    receipts = combined_receipts()
    versions = table(root / "reports/trajectory_versions.csv")
    tasks = table(root / "reports/query_metrics.csv")
    if (len(versions) != 20 or any(r["suite_size"] != SUITE_SIZE for r in versions)
            or {(r["kind"], r["version"]) for r in versions} != {(s, v) for s in study.SCENES for v in VERSIONS}
            or len(tasks) != expected or {(r["kind"], r["version"], r["method"]) for r in tasks} !=
            {(s, v, m) for s in study.SCENES for v in VERSIONS[1:4] for m in methods}
            or sum(r["logical_queries"] for r in tasks) != audit["logical_queries"]):
        raise ValueError("Published regression tables differ from the registered comparison")
    files = [audit_path, *(root / "reports" / name for name in (
        "trajectory_versions.csv", "query_metrics.csv", "method_contrasts.csv", "query_costs.csv", "recall_curves.json.gz"))]
    files.extend(query_directory(row["kind"], row["version"], row["method"], "dqn_continual") / "result.json.gz"
                 for row in tasks)
    receipts.update({p.relative_to(study.ROOT).as_posix(): study.sha(p) for p in files})
    figures = study.ROOT / "figures"
    regression = figures / "policy_regression/dqn/continual_learning"
    pngs = {}
    for name in ("version_collision_rates", "collision_transitions", "collision_recall",
                 "failure_response_evolution", "q90_queries_saved"):
        qa = read_json(root / "reports/figure_quality" / f"{name}.json")
        if qa["collision_audit"]["verdict"] != "PASS" or qa["alignment"]["summary"]["fail"]:
            raise ValueError(f"Regression figure layout failed: {name}")
        path = regression / f"{name}.png"
        pngs[path.relative_to(figures).as_posix()] = study.sha(path)
    with TemporaryDirectory(dir=figures, prefix=".publication-") as scratch:
        staging = Path(scratch)
        paths = [*plot_prediction(PREDICTION, staging / PREDICTION.name),
                 *plot_budget(source_root=BUDGET, destination=staging / BUDGET.name),
                 plot_ablation(ABLATION / "reports", staging / ABLATION.name)]
        for path in paths:
            pngs[path.relative_to(staging).as_posix()] = study.sha(path)
        release = {"status": "published", "training_run": REVISED.training_run, "checkpoint_sha256": study.sha(admission()),
                   "qualification": (STUDY / "base/qualification/qualification.json").relative_to(study.ROOT).as_posix(),
                   "report_receipts": receipts, "figures": pngs}
        if PUBLICATION.is_file() and read_json(PUBLICATION) != release:
            previous = read_json(PUBLICATION)
            fixed_keys = ("status", "training_run", "checkpoint_sha256", "qualification")
            prefix = root.relative_to(study.ROOT).as_posix() + "/"
            if (any(previous[k] != release[k] for k in fixed_keys)
                    or {k: v for k, v in previous["report_receipts"].items() if not k.startswith(prefix)}
                    != {k: v for k, v in receipts.items() if not k.startswith(prefix)}
                    or {k: v for k, v in previous["figures"].items() if not k.startswith("policy_regression/")}
                    != {k: v for k, v in pngs.items() if not k.startswith("policy_regression/")}):
                raise ValueError("SPECTRE publication refresh changed unrelated sealed evidence")
        for path in paths:
            destination = figures / path.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
        write_json(PUBLICATION, release)
    print("Published revised DQN, eight-ADS reports and 14 paper figures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prediction", "budget", "ablation", "combine", "publish"))
    parser.add_argument("--scene", choices=study.SCENES)
    parser.add_argument("--experiment", choices=("prediction", "budget", "ablation"),
                        help="Combine just one independently audited experiment")
    parser.add_argument("--figures", type=Path, help="Render combined figures under this figure directory")
    options = parser.parse_args()
    initialize_worker()
    admission()
    if options.stage == "publish":
        if options.scene or options.experiment or options.figures:
            parser.error("Publish requires the complete registered cohort and the canonical figure directory")
        publish()
    elif options.stage == "combine":
        if options.scene:
            parser.error("Combined paper estimates require all four scenes")
        study.input_receipts()
        experiments = {"prediction": combine_prediction, "budget": combine_budget, "ablation": combine_ablation}
        for name in ([options.experiment] if options.experiment else experiments):
            experiments[name]()
            if options.figures:
                if name == "prediction":
                    from experiments.plotting.k_shot import render
                    render(PREDICTION, options.figures / PREDICTION.name)
                elif name == "budget":
                    from experiments.plotting.budget_discovery import render
                    render(source_root=BUDGET, destination=options.figures / BUDGET.name)
                else:
                    from experiments.plotting.weight_effectiveness import plot
                    plot(ABLATION / "reports", options.figures / ABLATION.name)
    else:
        if options.experiment or options.figures:
            parser.error("Choose combine for a paper summary or its figures")
        audit = {"prediction": audit_prediction, "budget": audit_budget, "ablation": audit_ablation}[options.stage]
        for kind in ([options.scene] if options.scene else study.SCENES):
            audit(kind)
            print(options.stage, kind, "audited", flush=True)
