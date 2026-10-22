"""Independent lineage statistics; never calls report producers or selectors."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from controllers.highway.catalog import BY_ID

from .audit_collision_results import compare, require
from .protocol import DATA_ROOT, DEFAULT, RESULT_ROOT, ROOT, method_contract, validate_sut_freeze
from .studies import CONTINUOUS_TASKS, FAMILIES, REPRESENTATIVES


METRICS = ("unique_collisions", "discovery_auc", "behavior_coverage",
           "version_new_collisions", "historical_set_specific_collisions")
PAIR_FIELDS = ("lineage", "target_id", "scenario", "condition", "environment_seed", "algorithm_seed")
OUTPUT = ROOT / "results/benchmark/collision/statistical_report_audit.json"


def paired_summary(records, method, baseline, metric, *, repetitions=2000, seed=1501):
    buckets = {}
    for row in records:
        if row["method"] not in (method, baseline):
            continue
        key = tuple(row[field] for field in PAIR_FIELDS)
        bucket = buckets.setdefault(key, {})
        require(row["method"] not in bucket, "Duplicate statistical counterpart")
        value = row[metric]
        require(type(value) in (int, float) and math.isfinite(value), "Nonfinite statistical outcome")
        bucket[row["method"]] = float(value)
    require(bool(buckets) and all(set(b) == {method, baseline} for b in buckets.values()),
            "A statistical pair is missing its counterpart")
    lineages = sorted({key[0] for key in buckets})
    values = [np.asarray([pair[method] - pair[baseline] for key, pair in buckets.items() if key[0] == name])
              for name in lineages]
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(lineages), size=len(lineages))
        means = []
        for index in chosen:
            observations = values[index]
            draw = observations[rng.integers(0, len(observations), size=len(observations))]
            means.append(fmean(draw))
        replicates.append(fmean(means))
    replicates.sort()
    def quantile(probability):
        position = probability * (len(replicates) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        fraction = position - lower
        return (1 - fraction) * replicates[lower] + fraction * replicates[upper]
    return {"method": method, "baseline": baseline, "metric": metric,
            "difference": fmean(fmean(group) for group in values),
            "interval_95": [quantile(.025), quantile(.975)],
            "independent_lineages": len(lineages), "paired_runs": len(buckets),
            "lineage_differences": {name: fmean(group) for name, group in zip(lineages, values)}}


def restricted_records(records, restriction):
    output = []
    for row in records:
        first = row["first_collision"]
        require(type(first["observed"]) is bool and 0 < first["time"] <= restriction,
                "Invalid first-discovery event or follow-up")
        require(first["observed"] or first["time"] == restriction,
                "Statistical reconstruction requires the common administrative censoring horizon")
        output.append({**row, "restricted_first_queries": first["time"]})
    return output


def survival_summary(records, *, restriction):
    records = restricted_records(records, restriction)
    names = sorted({row["lineage"] for row in records})
    require(bool(names), "Survival analysis requires all completed runs")
    groups = {name: [r["first_collision"] for r in records if r["lineage"] == name] for name in names}
    # With the registered common cutoff, within-lineage KM is the empirical
    # event survival, and its restricted area equals mean min(T, cutoff).
    times = sorted({0, restriction, *(r["first_collision"]["time"] for r in records)})
    curve = []
    for time in times:
        probabilities = [fmean(float(not event["observed"] or event["time"] > time) for event in group)
                         for group in groups.values()]
        curve.append({"time": time, "survival": fmean(probabilities)})
    by_lineage = {name: {"runs": len(group), "censored_runs": sum(not e["observed"] for e in group),
                         "restricted_mean_queries": fmean(e["time"] for e in group)}
                  for name, group in groups.items()}
    return {"aggregation": "equal_weight_lineage_kaplan_meier", "restriction": restriction,
            "independent_lineages": len(names), "runs": len(records), "curve": curve,
            "restricted_mean_queries": fmean(r["restricted_mean_queries"] for r in by_lineage.values()),
            "censored_runs": sum(r["censored_runs"] for r in by_lineage.values()), "by_lineage": by_lineage}


def balanced_mean(rows, value):
    groups = defaultdict(list)
    for row in rows:
        groups[row["lineage"]].append(value(row))
    require(bool(groups), "No independent lineages in the statistical cell")
    return fmean(fmean(group) for group in groups.values())


def check_intervals(actual, rows, method, baseline, metrics, restriction):
    indexed = {(r["method"], r["baseline"], r["metric"]): r for r in actual}
    require(len(indexed) == len(actual), "Duplicate reported statistical comparison")
    expected_metrics = (*metrics, "restricted_first_queries")
    expected_keys = {(method, baseline, metric) for metric in expected_metrics}
    require(set(indexed) == expected_keys, "A statistical comparison is missing or unexpected")
    for metric in expected_metrics:
        expected = paired_summary(restricted_records(rows, restriction) if metric == "restricted_first_queries" else rows,
                                  method, baseline, metric)
        if metric == "restricted_first_queries":
            expected.update(restriction=restriction, lower_is_earlier=True)
        compare(indexed[(method, baseline, metric)], expected, "independent hierarchical interval")


def check_aggregate(actual, rows):
    methods = sorted({r["method"] for r in rows})
    require(set(actual["methods"]) == set(methods), "Reported methods differ from the completed statistical cell")
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        report = actual["methods"][method]
        compare(report["runs"], len(group), "statistical run count")
        compare(report["independent_lineages"], len({r["lineage"] for r in group}), "outer statistical units")
        compare(report["metrics"], {metric: balanced_mean(group, lambda r: r[metric]) for metric in METRICS},
                "lineage-balanced metric table")
        compare(report["first_collision"], survival_summary(group, restriction=DEFAULT.budget), "lineage-balanced survival")
        compare(report["checkpoint_collisions"], {str(step): balanced_mean(group, lambda r: r["discovery_curve"][step - 1])
                                                 for step in DEFAULT.checkpoints}, "paid-query checkpoints")
        auxiliary = {}
        for key in ("minimum_clearance_m", "minimum_ttc_s", "brier"):
            measured = [r[key] for r in group if r.get(key) is not None]
            auxiliary[key] = {"mean": fmean(measured) if measured else None, "measured_runs": len(measured)}
        compare(report["auxiliary"], auxiliary, "selected-query auxiliary statistics")
        violations, behavior = Counter(), {key: Counter() for key in ("collision_object", "contact_direction", "ego_action", "event_phase")}
        for row in group:
            violations.update(row.get("violation_counts", {}))
            for key in behavior:
                behavior[key].update(row.get("behavior_counts", {}).get(key, {}))
        compare(report["violations"], dict(violations), "violation totals")
        compare(report["collision_execution_behavior_counts"], {key: dict(values) for key, values in behavior.items()},
                "observable behavior totals")
    expected_count = 0
    for baseline in methods:
        if baseline == "history_response":
            continue
        comparisons = [r for r in actual["paired_lineage_intervals"] if r["baseline"] == baseline]
        check_intervals(comparisons, rows, "history_response", baseline, METRICS, DEFAULT.budget)
        expected_count += len(METRICS) + 1
    require(len(actual["paired_lineage_intervals"]) == expected_count, "Unexpected aggregate contrast")
    return expected_count


def check_coverage(actual, rows):
    grouped = defaultdict(list)
    fields = ("target_id", "scenario", "condition", "environment_seed", "algorithm_seed")
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    expected = []
    for key, group in sorted(grouped.items()):
        maximum = min(r["unique_collisions"] for r in group)
        coverage = []
        for row in group:
            values = []
            for count in range(1, maximum + 1):
                position = next(i for i, value in enumerate(row["discovery_curve"]) if value >= count)
                values.append(row["behavior_curve"][position])
            coverage.append(values)
        expected.append({"task": list(key), "methods": [r["method"] for r in group],
                         "common_maximum": maximum, "counts": list(range(1, maximum + 1)), "coverage": coverage})
    compare(actual, expected, "coverage at the common reachable collision counts")


def check_source_weights(actual, rows):
    names = (*FAMILIES, "target_only")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["study"] != "source_swap" or row["method"] not in ("history_response", "fixed_weights"):
            continue
        run = json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        curve = []
        for query in run["result"]["queries"]:
            values = dict.fromkeys(names, 0.0)
            weights, sources = query["source_weights"], query["source_names"]
            require(len(weights) == len(sources) and math.isclose(math.fsum(weights), 1, abs_tol=1e-9),
                    "Invalid recorded source probability distribution")
            for source, weight in zip(sources, weights):
                require(math.isfinite(weight) and weight >= 0, "Invalid source probability")
                family = "target_only" if source == "target_only" else BY_ID[source].family
                values[family] += weight
            curve.append([values[name] for name in names])
        grouped[(row["scope"], row["method"])][row["lineage"]].append(np.asarray(curve))
    expected = []
    for (scope, method), lineages in sorted(grouped.items()):
        arrays = [np.mean(curves, axis=0) for _, curves in sorted(lineages.items())]
        expected.append({"scope": scope, "method": method, "runs": sum(map(len, lineages.values())),
                         "independent_lineages": len(lineages), "columns": list(names),
                         "mean_weight_after_each_paid_query": np.mean(arrays, axis=0).tolist()})
    compare(actual, expected, "lineage-balanced source adaptation")


def check_finite(report, rows):
    require(report["status"] == "complete" and report["run_count"] == 10880, "Incomplete finite statistical report")
    compare(report["protocol"], asdict(DEFAULT), "finite report protocol")
    require(report["statistical_unit"] == "implementation_or_training_lineage", "Wrong independent statistical unit")
    contrasts = 0
    for study in ("overall", "strict_family"):
        for condition in DEFAULT.conditions:
            selected = [r for r in rows if r["study"] == study and r["condition"] == condition]
            contrasts += check_aggregate(report["studies"][study][condition], selected)
            check_coverage(report["equal_count_behavior_coverage"][f"{study}/{condition}"], selected)
    mechanisms = [r for r in rows if r["study"] == "mechanisms" or (
        r["study"] == "overall" and r["method"] == "history_response" and r["condition"] == "new"
        and r["target_id"] in REPRESENTATIVES)]
    contrasts += check_aggregate(report["studies"]["mechanisms"], mechanisms)
    swaps = report["studies"]["source_swaps"]
    indexed = {(r["target_family"], r["alternative_family"], r["method"]): r for r in swaps}
    require(len(indexed) == len(swaps) == 36, "Source replacement statistical matrix differs")
    for family in FAMILIES:
        for alternative in FAMILIES:
            if alternative == family:
                continue
            for method in ("history_response", "fixed_weights", "history_ranking"):
                selected = []
                for source, arm in ((family, "same_family"), (alternative, "alternative_family")):
                    scope = f"swap_{family}_{alternative}_{source}"
                    selected.extend({**r, "method": arm} for r in rows if r["study"] == "source_swap"
                                    and r["scope"] == scope and r["method"] == method)
                cell = indexed[(family, alternative, method)]
                metrics = tuple(m for m in METRICS if m != "historical_set_specific_collisions")
                check_intervals(cell["comparisons"], selected, "same_family", "alternative_family", metrics, DEFAULT.budget)
                compare(cell["historical_set_specific_by_arm"], {
                    arm: fmean(r["historical_set_specific_collisions"] for r in selected if r["method"] == arm)
                    for arm in ("same_family", "alternative_family")}, "within-arm historical-specific outcomes")
                contrasts += len(metrics) + 1
    check_source_weights(report["source_adaptation"], rows)
    return {"verified_statistical_contrasts": contrasts, "verified_runs": len(rows),
            "verified_coverage_groups": sum(len(r) for r in report["equal_count_behavior_coverage"].values()),
            "verified_source_weight_groups": len(report["source_adaptation"])}


def check_continuous(report, rows):
    metrics = ("unique_collisions", "discovery_auc", "behavior_coverage")
    require(report["status"] == "complete" and report["scope"] == "continuous_extension_only"
            and report["runs"] == len(rows) == 100, "Incomplete continuous statistical report")
    methods = {r["method"] for r in rows}
    require(set(report["methods"]) == methods, "Continuous statistical methods differ")
    contrasts = 0
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        for metric in metrics:
            compare(report["methods"][method][metric], balanced_mean(group, lambda r: r[metric]),
                    "continuous representative-case metric")
        compare(report["methods"][method]["first_collision"], survival_summary(group, restriction=DEFAULT.budget),
                "continuous censored discovery")
        if method != "history_response":
            selected = [r for r in report["paired_lineage_intervals"] if r["baseline"] == method]
            check_intervals(selected, rows, "history_response", method, metrics, DEFAULT.budget)
            contrasts += len(metrics) + 1
    require(len(report["paired_lineage_intervals"]) == contrasts, "Unexpected continuous statistical contrast")
    require({(r["target_id"], r["scenario"]) for r in rows} == set(CONTINUOUS_TASKS), "Wrong continuous representative cases")
    return {"verified_statistical_contrasts": contrasts, "verified_runs": len(rows)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_rows(scope):
    base = RESULT_ROOT if scope == "finite" else RESULT_ROOT / "continuous"
    index_path, report_path = base / "run_index.json", base / "report/report.json"
    proof_path = OUTPUT.with_name("final_result_audit.json" if scope == "finite" else "continuous_result_audit.json")
    required = [index_path, report_path, proof_path]
    if scope == "finite":
        required.append(OUTPUT.with_name("certification_result_audit.json"))
    pending = [str(path) for path in required if not path.exists()]
    if pending:
        return None, None, pending, []
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof["status"] != "complete":
        return None, None, [str(proof_path)], []
    expected_count = 10880 if scope == "finite" else 100
    require(proof["verified_runs"] == expected_count and proof["index_checked"] and proof["frozen_protocol_checked"]
            and not proof["errors"], "Full independent raw-run reconstruction is required")
    module = ROOT / "benchmark" / ("audit_collision_results.py" if scope == "finite" else "audit_continuous_results.py")
    require(proof["validator_sha256"] == sha(module), "Raw audit validator changed")
    frozen = json.loads((DATA_ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    validate_sut_freeze(frozen)
    compare(frozen["source_contract"], method_contract(), "current frozen statistical inputs")
    for artifact in frozen["method"]["models"]:
        require(sha(ROOT / artifact["path"]) == artifact["sha256"], "A frozen statistical model changed")
    certified = {}
    if scope == "finite":
        compare(proof["method_contract"], method_contract(), "raw audit source contract")
        certificate_path = required[-1]
        certificates = json.loads(certificate_path.read_text(encoding="utf-8"))
        if certificates["status"] != "complete":
            return None, None, [str(certificate_path)], []
        require(certificates["verified_runs"] == 10880 and certificates["verified_certificates"] == 96
                and not certificates["errors"], "Complete real-source certification is required")
        require(certificates["validator_sha256"] == sha(ROOT / "benchmark/audit_collision_certification.py")
                and certificates["protocol_sha256"] == sha(DATA_ROOT / "frozen_protocol.json"),
                "Certification proof uses different frozen inputs")
        for artifact in certificates["certificates"]:
            require(sha(artifact["path"]) == artifact["sha256"], "A real-source certificate changed")
        certified = {str(Path(row["path"]).resolve()): row for row in certificates["runs"]}
        require(len(certified) == 10880, "Certification proof duplicates a discovery run")
    else:
        manifest = json.loads((DATA_ROOT / "frozen_continuous.json").read_text(encoding="utf-8"))
        for name, fingerprint in manifest["source_contract"].items():
            require(sha(ROOT / name) == fingerprint, "Continuous statistical source inputs changed")
        for artifact in manifest["models"]:
            require(sha(ROOT / artifact["path"]) == artifact["sha256"], "Continuous frozen weights changed")
        require(manifest["main_protocol_sha256"] == sha(DATA_ROOT / "frozen_protocol.json")
                and manifest["development_report_sha256"] == sha(ROOT / manifest["development_report"]),
                "Continuous protocol or development evidence changed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["status"] == "complete" and len(index["runs"]) == expected_count, "Incomplete statistical input index")
    independent = {str(Path(row["path"]).resolve()): row for row in proof["runs"]}
    require(len(independent) == expected_count, "Duplicate raw statistical proof")
    rows = []
    for native in index["runs"]:
        key = str(Path(native["path"]).resolve())
        row = independent[key]
        require(sha(row["path"]) == row["sha256"], "An audited discovery result changed")
        compare(native, {k: v for k, v in row.items() if k not in ("sha256", "raw_evidence_sha256")},
                "current index versus independently rebuilt metrics")
        selected = {k: v for k, v in row.items() if k not in ("sha256", "raw_evidence_sha256")}
        if scope == "finite":
            selected.update({k: certified[key][k] for k in ("version_new_collisions", "historical_set_specific_collisions")})
        rows.append(selected)
    require({str(Path(r["path"]).resolve()) for r in rows} == set(independent), "Statistical index duplicates or omits a run")
    artifacts = [{"path": str(path), "sha256": sha(path)} for path in required]
    return rows, json.loads(report_path.read_text(encoding="utf-8")), [], artifacts


def check_csv(path, rows):
    fields = ["study", "scope", "target_id", "lineage", "family", "scenario", "condition",
              "environment_seed", "algorithm_seed", "method", *METRICS,
              "first_collision_query", "first_collision_observed", "first_collision_followup_queries",
              "restricted_first_queries", "selection_s", "adaptation_s"]
    identities = fields[:10]
    expected = {}
    for row in restricted_records(rows, DEFAULT.budget):
        expanded = {**row, "first_collision_observed": row["first_collision"]["observed"],
                    "first_collision_followup_queries": row["first_collision"]["time"]}
        values = {field: "" if expanded[field] is None else str(expanded[field]) for field in fields}
        key = tuple(values[field] for field in identities)
        require(key not in expected, "Duplicate expected CSV identity")
        expected[key] = values
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, "Statistical CSV fields differ")
        actual = list(reader)
    require(len(actual) == len(expected), "Statistical CSV row count differs")
    seen = set()
    for values in actual:
        key = tuple(values[field] for field in identities)
        require(key not in seen and key in expected, "Duplicate or unexpected CSV run")
        seen.add(key)
        compare(values, expected[key], "independently rebuilt statistical CSV")
    return len(actual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("finite", "continuous", "all"), default="all")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    scopes = ("finite", "continuous") if arguments.scope == "all" else (arguments.scope,)
    verified, pending, errors = {}, [], []
    for scope in scopes:
        try:
            rows, report, missing, artifacts = checked_rows(scope)
            if missing:
                pending.extend(missing)
                continue
            result = check_finite(report, rows) if scope == "finite" else check_continuous(report, rows)
            if scope == "finite":
                path = RESULT_ROOT / "report/runs.csv"
                result["verified_csv_rows"] = check_csv(path, rows)
                artifacts.append({"path": str(path), "sha256": sha(path)})
            verified[scope] = {**result, "artifacts": artifacts}
            print("verified independent statistics", scope, result, flush=True)
        except Exception as error:
            errors.append({"scope": scope, "error": repr(error)})
    result = {"status": "failed" if errors else "complete" if not pending and len(verified) == len(scopes) else "partial",
              "scope": "Requested statistical report tables, hierarchical paired intervals, censoring, common-count coverage, source weights and finite CSV; figure review and raw-cost ledgers are separate.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(), "requested_scopes": list(scopes),
              "validator_sha256": sha(Path(__file__)), "additional_physical_calls": 0,
              "verified": verified, "pending": pending, "errors": errors}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(result["status"], len(verified), "/", len(scopes), "statistical scopes;", len(errors), "errors", flush=True)
    if errors or arguments.require_complete and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
