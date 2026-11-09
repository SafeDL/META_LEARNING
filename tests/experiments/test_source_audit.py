import copy
from dataclasses import asdict

import numpy as np
import pytest

from experiments.audit_sources import BASELINES, audit, historical_views
from experiments.rss import measurement_contract
from experiments.scenarios import sample
from experiments.storage import read_json, save_archive, write_json
from testing_methods.common.response import ResponseArchive, archive_fingerprints

METHODS = [
    *sorted(BASELINES),
    "joint",
    "collision",
    "no_history",
    "historical_collision",
    "fixed_weights",
    "embedding_only",
    "aggregate",
]


def fixture_files(temporary, suite="fixed"):
    kind = "car_following"
    history, entries = [], []
    scenes = sample(kind, 6, "history", replicate=1)
    for i in range(8):
        archive = ResponseArchive(
            f"history_{i}",
            "rule",
            "idm",
            "history",
            scenes,
            np.array([False, True, False, True, False, True]),
            np.ones((6, 2)),
            provenance={"record_ids": [f"{i}:{j}" for j in range(6)]},
            rss_margin=np.array([0.5, -0.5, 0.2, -0.2, 0.1, -0.1]),
            rss_contract=measurement_contract(),
        )
        path = temporary / f"history_{i}.npz"
        save_archive(path, archive, provenance=archive.provenance)
        history.append(archive)
        entries.append(
            {
                "source_id": archive.source_id,
                "scenario": kind,
                "path": path.name,
                "fingerprint": archive_fingerprints([archive])[archive.source_id],
            }
        )
    manifest = temporary / "manifest.json"
    write_json(manifest, {"archives": entries})
    prepared = temporary / "prepared"
    models = {}
    for label, names in (
        ("shared_model", ("target_kernel", "historical_encoder")),
        ("historical_collision_model", ("target_kernel", "historical_encoder")),
        ("baseline", ("indago", "scenario_fuzz")),
    ):
        models[label] = {name: {"model_version": f"{label}:{name}"} for name in names}
        write_json(
            prepared / f"{label}_preparation_check.json",
            {"status": "complete", "scenarios": {kind: {"models": models[label]}}},
        )
    root = temporary / suite
    sizes = [4, 6] if suite == "sensitivity" else [6]
    protocol = {
        "suite": suite,
        "config": {"scenarios": [kind], "algorithm_seeds": [11]},
        "systems": [{"sut_id": sut} for sut in ("first", "second")],
        "main_methods": METHODS,
        "controls": [],
        "candidate_counts": {kind: 6},
    }
    if suite == "sensitivity":
        protocol["candidate_sizes"] = sizes
    write_json(root / "reports/protocol.json", protocol)
    views = historical_views(history)
    rows = []
    for size in sizes:
        for sut in ("first", "second"):
            for method in METHODS:
                task = {
                    "sut_id": sut,
                    "scenario": kind,
                    "seed": 11,
                    "method": method,
                    "held_family": None,
                }
                candidates = [asdict(s) for s in scenes[:size]]
                spec = {"task_snapshot": task, "candidates": candidates}
                result = {}
                if method in BASELINES:
                    construction = {}
                    if method in ("indago", "scenario_fuzz"):
                        construction = {
                            "history_fingerprints": views[True]["archives"],
                            **models["baseline"][method],
                        }
                    elif method == "paired_system_active_evaluation":
                        construction = {
                            "history_fingerprints": views[True]["archives"],
                            "history_ids": list(views[True]["archives"]),
                            "new_proxy_executions": 0,
                        }
                    spec.update(method=method, construction=construction)
                else:
                    use_rss = method != "historical_collision"
                    version = models["shared_model" if use_rss else "historical_collision_model"]
                    spec.update(
                        feedback_mode="collision" if method == "collision" else "joint",
                        historical_rss=use_rss,
                        update_source_weights=method != "fixed_weights",
                        prior_mode={"embedding_only": "embedding", "aggregate": "aggregate"}.get(
                            method, "sources"
                        ),
                        target_model_version=version["target_kernel"]["model_version"],
                        encoder_version=version["historical_encoder"]["model_version"],
                        source_snapshot={
                            "format": "joint-source-blocks",
                            "scenario": kind,
                            "model_version": version["historical_encoder"]["model_version"],
                            "rss_contract": measurement_contract(),
                            "candidates": candidates,
                            "covariance_mode": "point_joint_blocks",
                            "historical_rss": use_rss,
                            **copy.deepcopy(views[use_rss]),
                        },
                    )
                    names = [a.source_id for a in history]
                    if method == "aggregate":
                        names = ["aggregate_history_teacher"]
                    elif method == "embedding_only":
                        names = []
                        spec["source_snapshot"] = None
                    names += ["independent_target", "physical_target"]
                    if method == "no_history":
                        names = ["raw_target"]
                        spec.update(
                            target_model_version=None, encoder_version=None, source_snapshot=None
                        )
                    weights = [1 / len(names)] * len(names)
                    result = {
                        "component_names": names,
                        "prediction_snapshots": [
                            {
                                "paid_queries": 0,
                                "observed_indices": [],
                                "probabilities": [0.2] * size,
                                "component_weights": weights,
                            }
                        ],
                        "queries": [{"component_weights_after_query": weights}],
                    }
                path = f"methods/{size}/{sut}/{method}/result.json"
                write_json(root / path, {"specification": spec, "result": result})
                rows.append(
                    {k: v for k, v in task.items() if k != "held_family"}
                    | {"metrics": {"candidate_count": size}, "result_path": path}
                )
    write_json(root / "reports/summary.json", {"status": "complete", "runs": rows})
    return root, {"dataset_manifest": manifest, "preparation_root": prepared}


@pytest.mark.parametrize("suite", ["pilot", "fixed", "mechanisms", "sensitivity"])
def test_fixed_source_audit_covers_methods_components_and_nested_sizes(tmp_path, suite):
    root, options = fixture_files(tmp_path, suite)
    report = audit(root, **options)
    sizes = 2 if suite == "sensitivity" else 1
    assert report["status"] == "complete" and report["checked_runs"] == 28 * sizes
    assert report["new_queries_or_simulations"] == 0
    assert len(report["zero_feedback_cross_target_checks"]) == 7 * sizes
    for row in report["runs"]:
        expected = (
            8
            if row["method"]
            in (
                "joint",
                "collision",
                "historical_collision",
                "fixed_weights",
                "aggregate",
                "indago",
                "scenario_fuzz",
                "paired_system_active_evaluation",
            )
            else 0
        )
        assert row["admitted_source_count"] == expected


@pytest.mark.parametrize(
    "method,corruption",
    [
        ("joint", "future_source"),
        ("joint", "view_ids"),
        ("joint", "rss_contract"),
        ("joint", "cache_candidates"),
        ("joint", "zero_feedback"),
        ("joint", "wrong_kind"),
        ("joint", "model_version"),
        ("historical_collision", "ablation_rss"),
        ("collision", "feedback_mode"),
        ("fixed_weights", "weight_change"),
        ("aggregate", "components"),
        ("embedding_only", "components"),
        ("no_history", "model_version"),
        ("indago", "classifier_version"),
        ("scenario_fuzz", "classifier_source"),
        ("paired_system_active_evaluation", "proxy_query"),
        ("random", "unexpected_history"),
    ],
)
def test_source_permissions_and_mechanism_errors_are_rejected(tmp_path, method, corruption):
    root, options = fixture_files(tmp_path)
    path = root / f"methods/6/second/{method}/result.json"
    saved = read_json(path)
    spec, result = saved["specification"], saved["result"]
    if corruption == "future_source":
        spec["source_snapshot"]["archives"]["target_completed"] = "not_original_history"
    elif corruption == "view_ids":
        spec["source_snapshot"]["view_record_ids"]["history_0"][0] = "unexecuted_record"
    elif corruption == "rss_contract":
        spec["source_snapshot"]["rss_contract"] = {}
    elif corruption == "cache_candidates":
        spec["source_snapshot"]["candidates"].reverse()
    elif corruption == "zero_feedback":
        result["prediction_snapshots"][0]["probabilities"][0] += 0.1
    elif corruption == "wrong_kind":
        spec["candidates"][0]["kind"] = "front_brake"
    elif corruption == "model_version":
        spec["encoder_version"] = "changed"
    elif corruption == "ablation_rss":
        spec["historical_rss"] = True
    elif corruption == "feedback_mode":
        spec["feedback_mode"] = "joint"
    elif corruption == "weight_change":
        result["queries"][0]["component_weights_after_query"][0] += 0.1
    elif corruption == "components":
        result["component_names"].insert(0, "extra_source")
    elif corruption == "classifier_version":
        spec["construction"]["model_version"] = "changed"
    elif corruption == "classifier_source":
        spec["construction"]["history_fingerprints"]["target_completed"] = "unadmitted"
    elif corruption == "proxy_query":
        spec["construction"]["new_proxy_executions"] = 1
    else:
        spec["construction"] = {"unexpected_history": True}
    write_json(path, saved)
    with pytest.raises(AssertionError):
        audit(root, **options)


def test_partial_scope_and_duplicate_tasks_are_distinguished(tmp_path):
    root, options = fixture_files(tmp_path)
    path = root / "reports/summary.json"
    summary = read_json(path)
    summary.update(status="running", runs=summary["runs"][:14])
    write_json(path, summary)
    assert audit(root, **options)["status"] == "partial"
    summary["runs"].append(summary["runs"][0])
    write_json(path, summary)
    with pytest.raises(AssertionError):
        audit(root, **options)
