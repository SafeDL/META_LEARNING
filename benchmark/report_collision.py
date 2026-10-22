"""Four finite-pool studies, paired lineage intervals, censoring and actual costs."""

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from controllers.highway.catalog import BY_ID

from .collision_metrics import equal_count_coverage, lineage_survival, paired_lineage_bootstrap
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT
from .studies import FAMILIES, REPRESENTATIVES, tasks


OUTPUT = RESULT_ROOT / "report"
METRICS = ("unique_collisions", "discovery_auc", "behavior_coverage",
           "version_new_collisions", "historical_set_specific_collisions")


def restricted_records(rows):
    """With the common administrative cutoff, mean min(T, B) equals KM RMST(B)."""
    if any(not r["first_collision"]["observed"] and r["first_collision"]["time"] != DEFAULT.budget
           for r in rows):
        raise ValueError("Paired RMST requires the registered common administrative censoring budget")
    return [{**r, "restricted_first_queries": min(r["first_collision"]["time"], DEFAULT.budget)}
            for r in rows]


def aggregate(rows):
    methods = sorted({r["method"] for r in rows})
    output = {}
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        lineages = sorted({r["lineage"] for r in group})
        behavior_counts = {name: Counter() for name in ("collision_object", "contact_direction", "ego_action", "event_phase")}
        violations = Counter()
        for row in group:
            violations.update(row.get("violation_counts", {}))
            for name in behavior_counts:
                behavior_counts[name].update(row.get("behavior_counts", {}).get(name, {}))
        measured = {key: [r[key] for r in group if r.get(key) is not None]
                    for key in ("minimum_clearance_m", "minimum_ttc_s", "brier")}
        output[method] = {"runs": len(group), "independent_lineages": len(lineages),
                          "metrics": {metric: float(np.mean([
                              np.mean([r[metric] for r in group if r["lineage"] == lineage])
                              for lineage in lineages])) for metric in METRICS},
                          "first_collision": lineage_survival(group, restriction=DEFAULT.budget),
                          "auxiliary": {key: {"mean": float(np.mean(values)) if values else None,
                                               "measured_runs": len(values)} for key, values in measured.items()},
                          "auxiliary_scope": "Selected-query measurements; Brier is prequential on the adaptively selected queries.",
                          "violations": dict(violations),
                          "collision_execution_behavior_counts": {name: dict(v) for name, v in behavior_counts.items()},
                          "checkpoint_collisions": {str(step): float(np.mean([
                              np.mean([r["discovery_curve"][step - 1] for r in group if r["lineage"] == lineage])
                              for lineage in lineages])) for step in DEFAULT.checkpoints}}
    comparisons = []
    if "history_response" in methods:
        for baseline in methods:
            if baseline != "history_response":
                comparisons.extend(paired_lineage_bootstrap(rows, "history_response", baseline, metric)
                                   for metric in METRICS)
                contrast = paired_lineage_bootstrap(restricted_records(rows), "history_response", baseline,
                                                    "restricted_first_queries")
                contrast.update(restriction=DEFAULT.budget, lower_is_earlier=True)
                comparisons.append(contrast)
    return {"methods": output, "paired_lineage_intervals": comparisons}


def equal_coverage(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("target_id", "scenario", "condition", "environment_seed", "algorithm_seed"))
        groups[key].append(row)
    return [{"task": list(key), "methods": [r["method"] for r in group],
             **equal_count_coverage(group)} for key, group in sorted(groups.items())]


def source_swaps(rows):
    output = []
    for family in FAMILIES:
        for alternative in FAMILIES:
            if alternative == family:
                continue
            for method in ("history_response", "fixed_weights", "history_ranking"):
                group = []
                for source, arm in ((family, "same_family"), (alternative, "alternative_family")):
                    scope = f"swap_{family}_{alternative}_{source}"
                    group.extend({**row, "method": arm} for row in rows
                                 if row["scope"] == scope and row["method"] == method)
                first = paired_lineage_bootstrap(restricted_records(group), "same_family", "alternative_family",
                                                  "restricted_first_queries")
                first.update(restriction=DEFAULT.budget, lower_is_earlier=True)
                output.append({"target_family": family, "alternative_family": alternative,
                               "method": method,
                               "label_scope": "Version-new uses the same registered reference in both arms. Historical-set-specific counts retain their arm-specific source set and are reported within each arm.",
                               "comparisons": [paired_lineage_bootstrap(
                                   group, "same_family", "alternative_family", metric) for metric in METRICS
                                   if metric != "historical_set_specific_collisions"] + [first],
                               "historical_set_specific_by_arm": {
                                   arm: float(np.mean([r["historical_set_specific_collisions"] for r in group
                                                       if r["method"] == arm]))
                                   for arm in ("same_family", "alternative_family")}})
    return output


def source_weight_curve(queries):
    names = (*FAMILIES, "target_only")
    output = []
    for query in queries:
        weights = query["source_weights"]
        sources = query["source_names"]
        if len(weights) != len(sources) or not np.isclose(sum(weights), 1):
            raise ValueError("Recorded mixture weights do not form a probability distribution")
        row = dict.fromkeys(names, 0.0)
        for source, weight in zip(sources, weights):
            if weight < 0 or not np.isfinite(weight):
                raise ValueError("Invalid mixture probability in adaptation trace")
            family = "target_only" if source == "target_only" else BY_ID[source].family
            row[family] += weight
        output.append([row[name] for name in names])
    return np.asarray(output)


def source_adaptation(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["study"] == "source_swap" and row["method"] in ("history_response", "fixed_weights"):
            saved = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
            curve = source_weight_curve(saved["result"]["queries"])
            grouped[(row["scope"], row["method"])].append((row["lineage"], curve))
    result = []
    for (scope, method), traces in sorted(grouped.items()):
        lineages = sorted({lineage for lineage, _ in traces})
        values = np.array([np.mean([curve for lineage, curve in traces if lineage == selected], axis=0)
                           for selected in lineages])
        result.append({"scope": scope, "method": method, "runs": len(traces),
                       "independent_lineages": len(lineages), "columns": [*FAMILIES, "target_only"],
                       "mean_weight_after_each_paid_query": values.mean(0).tolist()})
    return result


def load_rows():
    index = json.loads((RESULT_ROOT / "run_index.json").read_text(encoding="utf-8"))
    certificates = json.loads((RESULT_ROOT / "new_collision_metrics.json").read_text(encoding="utf-8"))
    if index["status"] != "complete" or certificates["status"] != "complete":
        raise ValueError("Discovery and independent certification must both complete")
    registered = [asdict(task) for task in tasks()]
    keys = tuple(registered[0])
    expected = {tuple(row[k] for k in keys) for row in registered}
    observed = [tuple(row[k] for k in keys) for row in index["runs"]]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Report requires every preregistered paired run exactly once")
    certified = {r["path"]: r for r in certificates["runs"]}
    if len(certified) != len(index["runs"]):
        raise ValueError("Each finite run needs independent new-collision certification")
    rows = []
    for row in index["runs"]:
        if row["queries"] != DEFAULT.budget:
            raise ValueError("An incomplete run cannot enter the final report")
        first = row["first_collision"]
        rows.append({**row, **{k: v for k, v in certified[row["path"]].items() if k != "path"},
                     "first_collision_observed": first["observed"],
                     "first_collision_followup_queries": first["time"],
                     "restricted_first_queries": min(first["time"], DEFAULT.budget)})
    return rows


def costs(rows):
    model_reports = []
    frozen = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    for artifact in frozen["method"]["models"]:
        path = (ROOT / artifact["path"]).with_suffix(".json")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") == "complete" and "elapsed_s" in report:
            model_reports.append({"path": str(path.relative_to(ROOT)), "elapsed_s": report["elapsed_s"]})
    banks = []
    for path in DATA_ROOT.rglob("manifest.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("format") != "collision-execution-bank":
            continue
        if report["status"] != "complete":
            raise ValueError("An incomplete physical bank remains in the cost ledger")
        banks.append({"path": str(path.relative_to(ROOT)), **{k: report[k] for k in (
            "attempted_calls", "successful_calls", "failed_or_interrupted_calls", "physical_execution_s",
            "sut_initialization_s", "total_successful_call_wall_s", "failed_call_wall_s",
            "interrupted_call_duration_unknown")}})
    group = defaultdict(list)
    for row in rows:
        saved = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        group[(row["study"], row["method"])].append({**row,
            "model_construction_s": saved["cost"]["model_construction_s"]
                + saved["result"].get("model_construction_s", 0),
            "selector_process_wall_s": saved["cost"]["selector_process_wall_s"]})
    online = [{"study": study, "method": method, "runs": len(selected),
               "mean_selection_s": float(np.mean([r["selection_s"] for r in selected])),
               "mean_adaptation_s": float(np.mean([r["adaptation_s"] for r in selected])),
               "mean_model_construction_s": float(np.mean([r["model_construction_s"] for r in selected])),
               "mean_selector_process_wall_s": float(np.mean([r["selector_process_wall_s"] for r in selected])),
               "mean_equivalent_execution_s": float(np.mean([r["equivalent_execution_s"] for r in selected]))}
              for (study, method), selected in sorted(group.items())]
    policy_runs = []
    registered_runs = {s.training_run for s in BY_ID.values() if s.family == "learning"}
    policy_paths = list((ROOT / "controllers/highway/checkpoints/collision").glob("*/training.json"))
    archived = ROOT.parent / "META_LEARNING_archive/unqualified_policy_construction"
    policy_paths += list(archived.glob("*/training.json"))
    policy_paths += list((ROOT / "results/benchmark/collision/policy_construction/probes").glob("*/training.json"))
    for path in policy_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        policy_runs.append({"path": str(path), "status": report["status"],
                            "use": "registered_sut" if path.parent.name in registered_runs else "construction_reference",
                            "elapsed_s": report.get("elapsed_s"), "actual_timesteps": report.get("actual_timesteps"),
                            "qualification_executions": report.get("qualification_executions")})
    public_qualification = []
    for path in (ROOT / "results/benchmark/collision/qualification/public_interface").glob("*/qualification.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        public_qualification.append({"path": str(path), "training_run": report["training_run"],
                                     "actual_physical_calls": report["actual_physical_calls"],
                                     "elapsed_s": report["elapsed_s"], "qualified": report["qualified"],
                                     "interrupted_attempt_durations_unknown": report.get(
                                         "interrupted_attempt_durations_unknown", False)})
    return {"physical_banks": banks, "offline_model_builds": model_reports, "policy_construction": policy_runs,
            "public_interface_qualification": public_qualification,
            "online": online, "accounting": "Physical banks and certification are charged once. Replay time is not simulator wall time. Online costs include every initialization/query; offline model costs are reported without assuming an amortization count."}


def plot_curves(rows, stem, title, *, curve="discovery_curve", ylabel="Unique collisions", destination=None):
    destination = OUTPUT if destination is None else Path(destination)
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        lineages = sorted({r["lineage"] for r in group})
        values = np.array([np.mean([r[curve] for r in group if r["lineage"] == lineage], axis=0)
                           for lineage in lineages])
        axis.plot(np.arange(1, values.shape[1] + 1), values.mean(0), label=method)
    axis.set(xlabel="Paid target queries (initialization included)", ylabel=ylabel, title=title)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(destination / f"{stem}.{extension}", dpi=160)
    plt.close(figure)


def plot_survival(rows, stem, title, *, destination=None):
    destination = OUTPUT if destination is None else Path(destination)
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for method in sorted({r["method"] for r in rows}):
        summary = lineage_survival([r for r in rows if r["method"] == method], restriction=DEFAULT.budget)
        curve = summary["curve"]
        axis.step([r["time"] for r in curve], [r["survival"] for r in curve], where="post", label=method)
    axis.set(xlabel="Paid target queries (initialization included)",
             ylabel="Probability of no collision discovery yet", title=title,
             xlim=(0, DEFAULT.budget), ylim=(0, 1.02))
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    for extension in ("pdf", "svg", "png"):
        figure.savefig(destination / f"{stem}.{extension}", dpi=160)
    plt.close(figure)


def main():
    rows = load_rows()
    report = {"status": "complete", "protocol": asdict(DEFAULT), "run_count": len(rows),
              "statistical_unit": "implementation_or_training_lineage", "studies": {},
              "costs": costs(rows), "source_adaptation": source_adaptation(rows),
              "equal_count_behavior_coverage": {}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for study in ("overall", "strict_family"):
        report["studies"][study] = {}
        for condition in DEFAULT.conditions:
            group = [r for r in rows if r["study"] == study and r["condition"] == condition]
            report["studies"][study][condition] = aggregate(group)
            report["equal_count_behavior_coverage"][f"{study}/{condition}"] = equal_coverage(group)
            plot_curves(group, f"{study}_{condition}", f"{study}: {condition}")
            plot_survival(group, f"first_discovery_{study}_{condition}", f"First discovery: {study} / {condition}")
            if study == "overall":
                plot_curves(group, f"version_new_{condition}", f"Version-new collisions: {condition}",
                            curve="version_new_curve", ylabel="Unique version-new collisions")
                plot_curves(group, f"history_specific_{condition}", f"Historical-set-specific: {condition}",
                            curve="historical_set_specific_curve", ylabel="Unique historical-set-specific collisions")
    mechanism = [r for r in rows if r["study"] == "mechanisms" or (
        r["study"] == "overall" and r["method"] == "history_response" and r["condition"] == "new"
        and r["target_id"] in REPRESENTATIVES)]
    report["studies"]["mechanisms"] = aggregate(mechanism)
    report["studies"]["source_swaps"] = source_swaps([r for r in rows if r["study"] == "source_swap"])
    plot_curves(mechanism, "mechanisms", "Mechanisms on preregistered representative targets")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["study", "scope", "target_id", "lineage", "family", "scenario", "condition",
              "environment_seed", "algorithm_seed", "method", *METRICS,
              "first_collision_query", "first_collision_observed", "first_collision_followup_queries",
              "restricted_first_queries", "selection_s", "adaptation_s"]
    with (OUTPUT / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Collision discovery results", "", f"Completed paired runs: {len(rows)}.", "",
             "Main condition: new coordinates. Outer statistical unit: implementation or training lineage.", "",
             "| Method | Unique collisions | Discovery AUC | Behavior coverage | Version-new | History-specific | RMST queries | Censored runs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, values in report["studies"]["overall"]["new"]["methods"].items():
        first = values["first_collision"]
        lines.append("| " + method + " | " + " | ".join(f"{values['metrics'][k]:.3f}" for k in METRICS)
                     + f" | {first['restricted_mean_queries']:.3f} | {first['censored_runs']} |")
    lines += ["", f"First-discovery RMST uses the common {DEFAULT.budget}-query window; lower values mean earlier discovery. Runs without a collision remain censored."]
    lines += ["", "Paired lineage bootstrap intervals, censored first-discovery curves, equal-count behavior coverage, strict family holdouts, source replacements, mechanisms and separately charged costs are in [report.json](report.json).", "",
              "The registered HighwayEnv scope and its three candidate conditions are reported separately.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
