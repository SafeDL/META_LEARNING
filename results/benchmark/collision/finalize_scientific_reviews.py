"""Bind the final scientific narrative to already independently audited results.

Read-only result postprocessing; no simulation, selection or parameter fitting.
"""
import hashlib
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/benchmark/collision"


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise AssertionError(message)


def tables(section):
    output, current = [], []
    for line in section.splitlines() + [""]:
        if line.startswith("|"):
            current.append(line)
        elif current:
            output.append(current[2:])
            current = []
    return output


def numbers(row):
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", "|".join(row.split("|")[2:]).replace("\u2212", "-"))]


def close_rows(rows, expected, tolerance):
    require(len(rows) == len(expected), "Document table length")
    for row, values in zip(rows, expected):
        observed = numbers(row)
        require(len(observed) == len(values), "Document column length")
        require(all(abs(a - b) <= tolerance + 1e-9 for a, b in zip(observed, values)),
                "Document numeric mismatch: " + row)


def main():
    document = ROOT / "docs/method.md"
    text = document.read_text("utf-8")
    section6 = text[text.index("## 6."):text.index("## 7.")]
    section7 = text[text.index("## 7."):]
    finite_path = ROOT / "results/gcrt/collision/report/report.json"
    continuous_path = ROOT / "results/gcrt/collision/continuous/report/report.json"
    finite, continuous = read(finite_path), read(continuous_path)
    stats = read(BASE / "full_statistical_report_audit.json")
    require(stats["status"] == "complete" and not stats["errors"] and not stats["pending"], "Statistics incomplete")
    for part in stats["verified"].values():
        for item in part["artifacts"]:
            require(sha(item["path"]) == item["sha256"], "Audited statistical input changed")
    old = read(BASE / "finite_scientific_review.json")
    require(old["report_sha256"] == sha(finite_path), "Finite review input changed")
    methods = ("random", "space_filling", "history_ranking", "nearest_residual", "pooled_ft",
               "gp_collision", "fsbo", "history_response")
    keys = ("unique_collisions", "discovery_auc", "behavior_coverage", "version_new_collisions",
            "historical_set_specific_collisions")
    expected = []
    for method in methods:
        row = finite["studies"]["overall"]["new"]["methods"][method]
        expected.append([row["metrics"][k] for k in keys] + [row["first_collision"]["restricted_mean_queries"]])
        for k in keys:
            require(abs(old["primary_means"][method][k] - row["metrics"][k]) < 1e-10, "Finite review means")
    finite_tables = tables(section6)
    close_rows(finite_tables[0], expected, .005)
    contrasts = old["mechanism_count_auc_contrasts"]
    expected = []
    for baseline in ("no_response", "fixed_weights", "no_target", "greedy"):
        values = []
        for metric in ("unique_collisions", "discovery_auc"):
            row = next(r for r in contrasts if r["baseline"] == baseline and r["metric"] == metric)
            values.extend([row["difference"], *row["interval_95"]])
        expected.append(values)
    close_rows(finite_tables[1], expected, .0005)

    raw_path = BASE / "full_continuous_raw_result_audit.json"
    raw = read(raw_path)
    require(raw["status"] == "complete" and raw["verified_runs"] == 100 and not raw["errors"], "Continuous raw evidence")
    groups = defaultdict(list)
    for row in raw["runs"]:
        require(sha(row["path"]) == row["sha256"], "Continuous saved result changed")
        groups[(row["method"], row["lineage"])].append(row)
    order = ("random", "indago", "samota", "deep_hyperion", "history_response")
    means, expected = {}, []
    for method in order:
        rows = [r for r in raw["runs"] if r["method"] == method]
        require(len(rows) == 20 and all(len(groups[method, lineage]) == 5 for lineage in ("idm_mobil", "linear_mpc", "uct", "ppo")), "Continuous task pairing")
        values = {k: statistics.mean(statistics.mean(r[k] for r in group) for (m, _), group in groups.items() if m == method)
                  for k in ("unique_collisions", "discovery_auc", "behavior_coverage")}
        for k, value in values.items():
            require(abs(value - continuous["methods"][method][k]) < 1e-10, "Continuous reported mean")
        # All event/censor times have already been independently reconciled by the full statistics auditor.
        values["first_discovery_rmst"] = continuous["methods"][method]["first_collision"]["restricted_mean_queries"]
        means[method] = values
        expected.append(list(values.values()))
    continuous_tables = tables(section7)
    close_rows(continuous_tables[0], expected, .005)
    case_order = ("idm_mobil", "linear_mpc", "uct", "ppo")
    case_methods = ("history_response", "indago", "samota", "deep_hyperion", "random")
    cases = {lineage: {m: statistics.mean(r["unique_collisions"] for r in groups[m, lineage]) for m in case_methods}
             for lineage in case_order}
    close_rows(continuous_tables[1], [list(cases[lineage].values()) for lineage in case_order], .005)
    pairs = continuous["paired_lineage_intervals"]
    require(len(pairs) == 16 and all(r["independent_lineages"] == 4 and r["paired_runs"] == 20 for r in pairs), "Continuous inferential units")
    def contrast(baseline, metric):
        return next(r for r in pairs if r["baseline"] == baseline and r["metric"] == metric)
    for baseline in ("random", "indago", "deep_hyperion"):
        require(contrast(baseline, "unique_collisions")["interval_95"][0] > 0, "Claimed discovery interval")
    require(contrast("samota", "unique_collisions")["interval_95"][0] < 0 < contrast("samota", "unique_collisions")["interval_95"][1], "SAMOTA count uncertainty")
    require(contrast("samota", "discovery_auc")["interval_95"][0] < 0 < contrast("samota", "discovery_auc")["interval_95"][1], "SAMOTA AUC uncertainty")
    require(contrast("samota", "restricted_first_queries")["interval_95"][1] < 0, "Separate first-discovery evidence")
    require(all(not r["first_collision"]["observed"] for (m, lineage), rows in groups.items() if lineage == "idm_mobil" for r in rows), "Zero-discovery runs retained")
    require(sum(r["first_collision"]["observed"] for r in groups["history_response", "uct"]) == 2, "UCT observed/censored runs")
    annex_path = BASE / "continuous_reporting_annex_verification.json"
    annex = read(annex_path)
    require(annex["coverage_groups"] == 20 and annex["positive_common_ranges"] == 6 and annex["additional_physical_calls"] == 0, "Continuous common-count evidence")
    for kind in ("coverage", "auxiliary"):
        require(sha(annex[kind + "_path"]) == annex[kind + "_sha256"], "Annex changed")
    now = datetime.now(timezone.utc).isoformat()
    old.update(observed_at_utc=now, canonical_design_sha256=sha(document),
               canonical_section6_sha256=hashlib.sha256(section6.encode()).hexdigest(),
               document_main_table_rows_verified=8, document_mechanism_rows_verified=4,
               final_document_review_script_sha256=sha(__file__))
    (BASE / "finite_scientific_review.json").write_text(json.dumps(old, indent=2, ensure_ascii=False) + "\n", "utf-8")
    review = {"status": "reviewed_continuous_results", "observed_at_utc": now,
              "scope": "Four preregistered continuous tasks only; independent of finite-pool conclusions.",
              "validator_sha256": sha(__file__), "report_sha256": sha(continuous_path),
              "raw_proof_sha256": sha(raw_path), "statistics_proof_sha256": sha(BASE / "full_statistical_report_audit.json"),
              "cost_proof_sha256": sha(BASE / "continuous_cost_report_verification.json"),
              "figure_proof_sha256": sha(BASE / "continuous_figure_verification.json"),
              "reporting_annex_proof_sha256": sha(annex_path), "canonical_design_sha256": sha(document),
              "canonical_section7_sha256": hashlib.sha256(section7.encode()).hexdigest(),
              "document_method_rows_verified": 5, "document_case_rows_verified": 4,
              "primary_means": means, "case_collision_means": cases, "paired_contrasts": pairs,
              "same_count_tasks": 20, "same_count_tasks_with_positive_common_range": 6,
              "additional_physical_calls": 0, "frozen_protocol_retained": True,
              "findings": ["Discovery count intervals favor history_response over random, Indago and DeepHyperion in the registered individual comparisons.",
                           "SAMOTA has higher mean collision counts and AUC; their four-task paired intervals include zero. No overall winner is established.",
                           "First-discovery RMST favors history_response over SAMOTA; this does not establish a budget-end count advantage.",
                           "IDM has no discoveries for every method. UCT has two observed history_response runs and three censored runs; control runs are all censored.",
                           "Behavior coverage is lower in mean than random and SAMOTA, with intervals spanning zero; breadth leadership is not established.",
                           "Actual continuous calls and costs are charged separately. No version-new or historical-set-specific labels are inferred from predicted historical responses."]}
    (BASE / "continuous_scientific_review.json").write_text(json.dumps(review, indent=2) + "\n", "utf-8")
    print("verified_document_tables: finite8+mechanisms4, continuous5+cases4; reviewed all16 continuous contrasts", flush=True)


if __name__ == "__main__":
    main()
