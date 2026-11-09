"""Verify fixed-history permissions and mechanism settings from saved method results."""

from datetime import datetime
from pathlib import Path

import numpy as np

from testing_methods.common.response import archive_fingerprints

from .discovery_config import read_json, write_json
from .history import calculation_view, load_training_archives
from .joint_validation import OUTPUT
from .prepare_joint import DATASET_ROOT
from .rss import measurement_contract

BASELINES = {
    "random",
    "av_bo_rf_ei",
    "samota",
    "scenario_fuzz",
    "paired_system_active_evaluation",
    "indago",
    "behavexplor",
}


def historical_views(history):
    result = {}
    for use_rss in (True, False):
        selected = [calculation_view(archive, use_rss=use_rss) for archive in history]
        result[use_rss] = {
            "archives": archive_fingerprints(history),
            "views": {a.source_id: a.provenance["calculation_indices"] for a in selected},
            "record_ids": {a.source_id: a.provenance["record_ids"] for a in history},
            "view_record_ids": {a.source_id: a.provenance["record_ids"] for a in selected},
        }
    return result


def verify_baseline(method, spec, original, classifiers):
    assert spec["method"] == method
    construction = spec["construction"]
    if method in ("indago", "scenario_fuzz"):
        assert construction["history_fingerprints"] == original
        assert construction["model_version"] == classifiers[method]["model_version"]
    elif method == "paired_system_active_evaluation":
        assert construction["history_fingerprints"] == original
        assert set(construction["history_ids"]) == set(original)
        assert construction["new_proxy_executions"] == 0
    else:
        assert construction == {}
    return len(original) if construction else 0


def verify_response_model(method, spec, result, history, views, models):
    assert spec["feedback_mode"] == ("collision" if method == "collision" else "joint")
    assert spec["historical_rss"] == (method != "historical_collision")
    assert spec["update_source_weights"] == (method != "fixed_weights")
    assert spec["prior_mode"] == {"embedding_only": "embedding", "aggregate": "aggregate"}.get(
        method, "sources"
    )
    if method == "no_history":
        assert all(
            spec[name] is None
            for name in ("target_model_version", "encoder_version", "source_snapshot")
        )
        assert result["component_names"] == ["raw_target"]
        return 0
    assert spec["target_model_version"] == models["target_kernel"]["model_version"]
    assert spec["encoder_version"] == models["historical_encoder"]["model_version"]
    if method == "embedding_only":
        assert spec["source_snapshot"] is None
        assert result["component_names"] == ["independent_target", "physical_target"]
        return 0
    sources = spec["source_snapshot"]
    use_rss = method != "historical_collision"
    assert sources == {
        "format": "joint-source-blocks",
        "scenario": spec["task_snapshot"]["scenario"],
        "model_version": spec["encoder_version"],
        "rss_contract": measurement_contract(),
        "candidates": spec["candidates"],
        "covariance_mode": "point_joint_blocks",
        "historical_rss": use_rss,
        **views[use_rss],
    }
    names = [a.source_id for a in history]
    if method == "aggregate":
        names = ["aggregate_history_teacher"]
    assert result["component_names"] == [*names, "independent_target", "physical_target"]
    return len(sources["archives"])


def audit(
    root=OUTPUT / "pilot",
    *,
    dataset_manifest=DATASET_ROOT / "manifest.json",
    preparation_root=OUTPUT / "reports",
):
    root, preparation_root = Path(root), Path(preparation_root)
    summary, protocol = (
        read_json(root / "reports" / name) for name in ("summary.json", "protocol.json")
    )
    assert protocol["suite"] in ("pilot", "fixed", "mechanisms", "sensitivity")
    models, collision_models, classifiers = (
        read_json(preparation_root / f"{name}_preparation_check.json")
        for name in ("shared_model", "historical_collision_model", "baseline")
    )
    assert all(report["status"] == "complete" for report in (models, collision_models, classifiers))
    histories = {
        kind: load_training_archives(dataset_manifest, kind)
        for kind in protocol["config"]["scenarios"]
    }
    views = {}
    for kind, history in histories.items():
        assert len(history) == 8
        assert all(a.role == "history" and a.source_id.startswith("history_") for a in history)
        views[kind] = historical_views(history)
    methods = protocol["main_methods"] + protocol["controls"]
    sizes = protocol.get("candidate_sizes")
    expected = {
        (sut["sut_id"], kind, seed, method, size)
        for sut in protocol["systems"]
        for kind in protocol["config"]["scenarios"]
        for seed in protocol["config"]["algorithm_seeds"]
        for method in methods
        for size in (sizes or [protocol["candidate_counts"][kind]])
    }
    checks, seen, zero_references, transitions = [], set(), {}, []
    for row in summary["runs"]:
        kind, method = row["scenario"], row["method"]
        count = row["metrics"]["candidate_count"]
        key = row["sut_id"], kind, row["seed"], method, count
        assert key in expected and key not in seen, key
        seen.add(key)
        saved = read_json(root / row["result_path"])
        spec, result = saved["specification"], saved["result"]
        task = spec["task_snapshot"]
        assert all(task[name] == row[name] for name in ("sut_id", "scenario", "seed", "method"))
        assert task["held_family"] is None
        assert all(candidate["kind"] == kind for candidate in spec["candidates"])
        if method in BASELINES:
            source_count = verify_baseline(
                method,
                spec,
                views[kind][True]["archives"],
                classifiers["scenarios"][kind]["models"],
            )
        else:
            prepared = collision_models if method == "historical_collision" else models
            source_count = verify_response_model(
                method,
                spec,
                result,
                histories[kind],
                views[kind],
                prepared["scenarios"][kind]["models"],
            )
            zero = result["prediction_snapshots"][0]
            assert zero["paid_queries"] == 0 and zero["observed_indices"] == []
            reference_key = kind, method, row["seed"], count
            if reference_key in zero_references:
                other, probabilities = zero_references[reference_key]
                np.testing.assert_allclose(
                    zero["probabilities"], probabilities, rtol=1e-12, atol=1e-12
                )
                transitions.append(
                    {
                        "scenario": kind,
                        "method": method,
                        "candidate_count": count,
                        "first_sut": other,
                        "next_sut": row["sut_id"],
                    }
                )
            else:
                zero_references[reference_key] = row["sut_id"], zero["probabilities"]
            if method == "fixed_weights":
                for query in result["queries"]:
                    np.testing.assert_allclose(
                        query["component_weights_after_query"],
                        zero["component_weights"],
                        rtol=1e-12,
                        atol=1e-12,
                    )
        checks.append(
            {
                "sut_id": row["sut_id"],
                "scenario": kind,
                "method": method,
                "candidate_count": count,
                "result_path": row["result_path"],
                "admitted_source_count": source_count,
                "original_history_and_model_verified": True,
                "completed_target_records_admitted": False,
            }
        )
    if summary["status"] == "complete":
        assert seen == expected
    report = {
        "format": "fixed-source-admission-check",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "complete" if seen == expected else "partial",
        "suite": protocol["suite"],
        "expected_runs": len(expected),
        "checked_runs": len(checks),
        "runs": checks,
        "zero_feedback_cross_target_checks": transitions,
        "verification_scope": "All methods: original same-scenario history, prepared model identity, paid-target exclusion, full and sampled record IDs, candidate/RSS/cache binding, ablation flags, components and fixed-weight behavior. Model preparation is verified separately.",
        "new_queries_or_simulations": 0,
    }
    write_json(root / "reports/source_admission_check.json", report)
    return report


if __name__ == "__main__":
    report = audit()
    print(
        {k: v for k, v in report.items() if k not in ("runs", "zero_feedback_cross_target_checks")}
    )
