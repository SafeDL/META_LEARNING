"""Audit complete main-experiment records, summarize them and export source tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import collect_transfer_records as reader
from . import transfer_batches as batches
from . import transfer_execution as execution
from .history_transfer_experiment import OUTPUT, freeze
from .history_transfer_summary import (
    budget_summary,
    prediction_summary,
    register_statistics,
)
from .k_shot import LOWER_BETTER, METRICS
from .paired_statistics import json_ready, paired_summary
from .storage import read_json, write_json
from .transfer_admission import load_acceptance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pooled_comparison(rows, statistics):
    axes = [list(statistics[name]) for name in ("ads", "scenarios", "feedback_seeds")] + [
        statistics["prediction"]["methods"],
        statistics["prediction"]["checkpoints"],
    ]
    values = np.full(tuple(map(len, axes)) + (len(METRICS),), np.nan)
    for row in rows:
        if row["status"] == "complete":
            index = tuple(
                axis.index(row[key])
                for axis, key in zip(axes, ("sut_id", "scenario", "seed", "method", "K"))
            )
            values[index] = [np.nan if row[m] is None else row[m] for m in METRICS]
    kwargs = {
        "ads_ids": axes[0],
        "ads_families": [statistics["ads"][name] for name in axes[0]],
        "methods": axes[3],
        "baseline": "pooled_history",
        "lower_better": LOWER_BETTER,
        "replicates": statistics["statistics"]["replicates"],
    }
    return json_ready(
        {
            "metrics": list(METRICS),
            "checkpoints": axes[4],
            "scopes": {
                "all": paired_summary(values, **kwargs),
                **{
                    scene: paired_summary(values[:, index : index + 1], **kwargs)
                    for index, scene in enumerate(axes[1])
                },
            },
        }
    )


def summarize_audited(records, statistics, mode):
    """Use the registered estimator on already audited, terminal metric grids."""
    if records["status"] != "complete" or mode not in ("prediction", "budget"):
        raise ValueError("Summaries require a complete audited prediction or budget grid")
    rows = [r for r in records["rows"] if r["experiment"] == mode]
    kwargs = {
        "ads": statistics["ads"],
        "scenes": statistics["scenarios"],
        "seeds": statistics["feedback_seeds"],
        "methods": statistics[mode]["methods"],
        "candidate_count": statistics["candidate_count"],
        "replicates": statistics["statistics"]["replicates"],
    }
    if mode == "prediction":
        summary = prediction_summary(
            rows, **kwargs, checkpoints=statistics["prediction"]["checkpoints"]
        )
        if "pooled_history" in statistics["prediction"]["methods"]:
            summary["comparison_with_pooled"] = _pooled_comparison(rows, statistics)
        return summary
    return budget_summary(
        rows,
        **kwargs,
        budget=statistics["budget"]["queries"],
        initial_count=statistics["budget"]["common_initial"],
    )


def attempt_costs(root, cases):
    """Retain failed/interrupted computation instead of charging only the last attempt."""
    root = Path(root)
    attempts = []
    for case in cases:
        result_path, _ = execution._paths(root, case)
        paths = sorted((result_path.parent / "attempts").glob(f"{case.method}_*.json"))
        if not paths:
            raise ValueError("Formal cost reporting requires the saved execution attempts")
        last_after = 0
        for ordinal, path in enumerate(paths):
            record = read_json(path)
            before, after = (
                record["saved_responses_before_run"],
                record["saved_responses_after_run"],
            )
            budget = (
                len(case.specification["split"]["query_order"])
                if case.experiment == "prediction"
                else case.specification["budget"]
            )
            if (
                record["specification"] != case.specification
                or record["status"] not in ("complete", "interrupted", "failed")
                or type(before) is not int
                or type(after) is not int
                or before != last_after
                or not before <= after <= budget
                or not np.isfinite(record["wall_s"])
                or record["wall_s"] < 0
            ):
                raise ValueError("Execution attempt differs from its task or saved-feedback chain")
            resources = record.get("selector_resources", {})
            for name in ("wall_s", "sampled_peak_cpu_bytes"):
                value = resources.get(name)
                if value is not None and (not np.isfinite(value) or value < 0):
                    raise ValueError("Saved selector cost must be finite and nonnegative")
            attempts.append(
                {
                    "experiment": case.experiment,
                    "sut_id": case.sut_id,
                    "scenario": case.scenario,
                    "seed": case.seed,
                    "method": case.method,
                    "attempt": ordinal + 1,
                    "status": record["status"],
                    "failure_stage": record.get("failure_stage"),
                    "error": record.get("error"),
                    "saved_responses_before_run": before,
                    "saved_responses_after_run": after,
                    "attempt_wall_s": record["wall_s"],
                    "selector_wall_s": resources.get("wall_s"),
                    "selector_peak_cpu_bytes": resources.get("sampled_peak_cpu_bytes"),
                    "record": path.relative_to(root).as_posix(),
                    "record_sha256": sha(path),
                }
            )
            last_after = after
        expected = "failed" if case.final_failure else "complete"
        if attempts[-1]["status"] != expected:
            raise ValueError("Final attempt does not match the independently declared task status")
    return attempts


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shared_construction(root, cases):
    """Join each actual source snapshot to its single shared construction receipt."""
    root = Path(root)
    selected = {}
    for case in cases:
        if case.method == "full":
            specification = case.specification["source_snapshot"]
            key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
            selected[case.scenario, key] = specification
    rows = []
    for (scene, key), specification in selected.items():
        path = root / "reports/source_construction" / scene / f"{key}.json"
        saved = read_json(path)
        if saved["specification"] != specification or saved["scope"] not in (
            "cache_load",
            "source_construction",
        ):
            raise ValueError("Shared construction receipt differs from the actual source snapshot")
        for metric in ("wall_s", "sampled_peak_cpu_bytes"):
            value = saved["resources"][metric]
            if not np.isfinite(value) or value < 0:
                raise ValueError("Shared construction cost must be finite and nonnegative")
        rows.append(
            {
                "scenario": scene,
                "record": path.relative_to(root).as_posix(),
                "record_sha256": sha(path),
                **saved,
            }
        )
    return rows


def report(mode="prediction", root=OUTPUT):
    from .transfer_admission import require_experiment

    require_experiment(mode)
    root = Path(root)
    if mode not in ("prediction", "budget", "all"):
        raise ValueError("Unknown main report mode")
    # The real study admission is checked before declarations, banks or target truth.
    load_acceptance(root)
    statistics = register_statistics(root)
    modes = ("prediction", "budget") if mode == "all" else (mode,)
    cases = [case for selected in modes for case in batches.approved_cases(root, selected)]
    audited = reader.collect_cases(cases, root, root / "physical/target")
    if audited["status"] != "complete":
        progress = {"mode": mode, "status": "incomplete", "inventory": audited["inventory"]}
        write_json(root / "reports/main_results" / f"{mode}_progress.json", progress)
        return progress
    summaries = {selected: summarize_audited(audited, statistics, selected) for selected in modes}
    attempts = attempt_costs(root, cases)
    shared = shared_construction(root, cases)
    destination = root / "reports/main_results" / mode
    protocol = freeze(
        destination / "protocol.json",
        {
            "scope": "Complete admitted main experiment; evaluator-owned outcomes",
            "modes": list(modes),
            "statistics_sha256": sha(root / "reports/statistics_protocol.json"),
            "admission_sha256": sha(root / batches.ACCEPTANCE),
            "settings_sha256": sha(root / batches.REGISTRY),
            "driver_sha256": sha(__file__),
            "approved_cases": audited["approved_cases"],
            "primary_prediction_reference": "target_only",
            "prediction_methods": statistics["prediction"]["methods"],
            "new_physical_executions": 0,
        },
    )
    reader.export_source_data(destination, audited)
    _csv(destination / "execution_attempts.csv", attempts)
    write_json(destination / "execution_attempts.json", attempts)
    for selected, summary in summaries.items():
        write_json(destination / f"{selected}_summary.json", summary)
    write_json(
        destination / "shared_source_construction.json",
        {
            "scope": "Shared per scene, counted once across ADS/seeds/modes; cold build and cache load are separate",
            "records": shared,
        },
    )
    files = [
        path for path in destination.iterdir() if path.is_file() and path.name != "completion.json"
    ]
    completion = {
        "status": "complete",
        "protocol": protocol,
        "cases": len(cases),
        "metric_rows": len(audited["rows"]),
        "successful_logical_queries": sum(
            c["successful_logical_queries"] for c in audited["costs"]
        ),
        "execution_attempts": len(attempts),
        "all_attempt_wall_s": sum(a["attempt_wall_s"] for a in attempts),
        "physical_bank_scope": audited["physical_bank_scope"],
        "physical_bank_successful_calls": sum(
            b["successful_calls"] for b in audited["physical_banks"]
        ),
        "physical_bank_note": "Target libraries are shared by both modes; do not add their counts across reports. Historical collection, training, checks and development are separate.",
        "cost_note": "Attempt wall includes selector waits, validation and evaluation. Missing selector-only failure costs stay null; memory peaks are not added across attempts.",
        "files": {path.name: sha(path) for path in sorted(files)},
        "new_physical_executions": 0,
        "figures": "PNG generation follows inspection of these audited summaries",
    }
    write_json(destination / "completion.json", completion)
    return completion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("prediction",), default="prediction")
    print(report(parser.parse_args().mode), flush=True)
