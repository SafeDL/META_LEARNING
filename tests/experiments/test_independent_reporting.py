from dataclasses import replace

from controllers.highway.catalog import INDEPENDENT_SUTS, SUTS
from experiments import independent_validation_report as reporting
from experiments.independent_protocol import canonical_task, jobs, reuse_jobs, tasks
from experiments.scenarios import sample


def test_budget_checkpoints_use_lineage_means_and_preserve_inapplicable_recall():
    rows = [
        {
            "method": "gcrt",
            "lineage": lineage,
            "discovery_curve": [scale * budget // 4 for budget in range(1, 129)],
            "recall_curve": [(scale * budget // 4) / 128 for budget in range(1, 129)],
        }
        for lineage, scale in (("a", 1), ("a", 3), ("b", 4))
    ]
    rows.append(
        {
            "method": "random",
            "lineage": "a",
            "discovery_curve": [0] * 128,
            "recall_curve": [None] * 128,
        }
    )
    result = reporting.curves(rows, ("gcrt", "random"))
    for budget in (16, 32, 64, 128):
        assert result["gcrt"]["checkpoints"][str(budget)] == {
            "unique_collisions": 3 * budget / 4,
            "collision_recall": 3 * budget / 512,
        }
        assert result["random"]["checkpoints"][str(budget)] == {
            "unique_collisions": 0,
            "collision_recall": None,
        }


def test_cost_report_separates_reused_sources_new_targets_and_initialization(monkeypatch, tmp_path):
    policy_ids = {
        s.training_run: s.sut_id for s in (*SUTS, *INDEPENDENT_SUTS) if s.family == "learning"
    }
    read_policy_ids = []
    schedule_root = tmp_path / "model_construction"
    schedule_root.mkdir()
    schedule_names = (
        "classifiers_rear_parallel.json",
        "classifiers_rear_full_parallel.json",
        "classifiers_cut_out_learning_parallel.json",
    )
    for name in schedule_names:
        (schedule_root / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reporting, "OUTPUT", tmp_path)

    def read(path):
        if path.name == "construction_data_audit.json":
            return {
                "banks": [
                    {"scenario": "front_brake", "actual_physical_calls": 10},
                    {"scenario": "cut_in_interaction", "actual_physical_calls": 20},
                ]
            }
        if path.name == "target_bank_index.json":
            return {
                "banks": [
                    {
                        "target_id": "independent_idm_mobil",
                        "scenario": "front_brake",
                        "path": "data/collision/independent_targets/old_registration",
                    }
                ],
                "attempted_calls": 512,
                "last_collection_wall_s": 17,
            }
        if path.name == "manifest.json":
            assert (
                path
                == reporting.target_bank_path(
                    {"target_id": "independent_idm_mobil", "scenario": "front_brake"}
                )
                / "manifest.json"
            )
            return {
                "successful_calls": 512,
                "total_successful_call_wall_s": 100,
                "sut_initialization_s": 15,
                "failed_call_wall_s": 0,
                "interrupted_call_duration_unknown": 0,
                "last_collection_wall_s": 17,
            }
        if path.name == "training.json":
            identity = policy_ids[path.parent.name]
            assert not identity.startswith("target_")
            read_policy_ids.append(identity)
            return {
                "sut_id": identity,
                "training_seed": 11,
                "actual_timesteps": 100,
                "elapsed_s": 3,
                "qualification_executions": 650,
            }
        if path.name == "qualification.json":
            return {"courses": [{"physics_ticks": 20} for _ in range(5)]}
        if path.name in schedule_names:
            return {
                "status": "complete",
                "workers": 4 if path.name == "classifiers_rear_parallel.json" else 1,
            }
        if "model_construction" in path.parts:
            return {
                "report": {"training_s": 2 if "indago" in path.parts else 4},
                "new_training_s": 0,
                "reused": True,
            }
        if path.name == "target_kernel.json":
            return {"training_s": 1, "new_training_s": 0}
        if path.name == "historical_encoder.json":
            return {"training_s": 5, "prefix_cache_s": 2}
        if path.name == "source_priors.json":
            return {"construction_s": 1}
        raise AssertionError(f"Unexpected report input: {path}")

    monkeypatch.setattr(reporting, "read_json", read)
    result = reporting.costs([])
    assert {r["sut_id"] for r in result["reused_source_policy_runs"]} == {
        "history_ppo",
        "history_dqn",
        "development_ppo",
    }
    assert {r["sut_id"] for r in result["new_policy_runs"]} == {
        "independent_ppo",
        "independent_dqn",
    }
    assert len(read_policy_ids) == 5
    assert result["reused_old_history_development_calls"] == 10
    assert result["new_interaction_history_development_calls"] == 20
    assert result["new_target_pool_collection_wall_s"] == 17
    assert result["target_banks"][0]["sut_initialization_s"] == 15
    assert result["nonlearning_qualification"]["reused_source"]["course_calls"] == 45
    assert result["nonlearning_qualification"]["new_independent"]["course_calls"] == 30
    assert len(result["model_construction"]) == 25
    assert all(r["main_target_new_training_s"] == 0 for r in result["model_construction"])
    assert sum("indago" in r["classifier_training"] for r in result["model_construction"]) == 5
    assert {
        name: row["workers"]
        for name, row in result["additional_classifier_execution_schedules"].items()
    } == {
        "classifiers_rear_parallel.json": 4,
        "classifiers_rear_full_parallel.json": 1,
        "classifiers_cut_out_learning_parallel.json": 1,
    }


def test_reused_comparisons_reference_original_rows_without_mutating_or_recharging_them():
    task = tasks()[0]
    original = {**task, "method": "random", "path": "original.json", "unique_collisions": 4}
    strict = {**task, "condition": "strict_leave_family"}
    index = {"runs": [original]}
    references = {
        "references": [{"task": strict, "source_path": "original.json", "source_sha256": "fixture"}]
    }
    rows = reporting.comparison_rows(index, references)
    assert len(rows) == 2 and len(index["runs"]) == 1
    assert original["condition"] == "all_sources" and "reused" not in original
    assert rows[1]["condition"] == "strict_leave_family" and rows[1]["reused"]
    assert rows[1]["path"] == original["path"] and rows[1]["unique_collisions"] == 4


def test_complete_reduced_report_pairs_correct_scopes_and_charges_only_unique_runs(
    monkeypatch, tmp_path
):
    rows = []
    for position, (task, method) in enumerate(jobs()):
        count = int(task["lineage"] != "ppo")
        curve = [0] * 16 + [count] * 112
        rows.append(
            {
                **task,
                "method": method,
                "path": f"fixture/{position}.json",
                "unique_collisions": count,
                "collision_recall": 0.5 if count else None,
                "discovery_auc": sum(curve) / 128,
                "selection_s": 1,
                "adaptation_s": 1,
                "discovery_curve": curve,
                "recall_curve": [n / 2 for n in curve] if count else [None] * 128,
                "behavior_curve": curve,
                "behavior_coverage": count,
                "collision_executions": count,
                "minimum_clearance_m": 0 if count else 10,
                "minimum_ttc_s": 0 if count else 10,
                "brier": 0.25,
                "first_collision": {"time": 17 if count else 128, "observed": bool(count)},
            }
        )
    task_keys = tuple(tasks()[0])
    by_key = {(tuple({k: row[k] for k in task_keys}.items()), row["method"]): row for row in rows}
    references = [
        {
            "task": task,
            "method": method,
            "source_path": by_key[tuple(canonical_task(task, method).items()), method]["path"],
            "source_sha256": "synthetic_fixture",
        }
        for task, method in reuse_jobs()
    ]
    inputs = {
        "audit.json": {
            "status": "passed",
            "runs": 1880,
            "queries": 240640,
            "reused_references": 600,
            "independent_main_choices": 1440,
        },
        "run_index.json": {"status": "complete", "runs": rows},
        "reuse_index.json": {"status": "complete", "references": references},
    }
    monkeypatch.setattr(reporting, "OUTPUT", tmp_path)
    monkeypatch.setattr(reporting, "validate_protocol", lambda: None)
    monkeypatch.setattr(reporting, "read_json", lambda path: inputs[path.name])
    bootstrap = reporting.paired_lineage_bootstrap
    monkeypatch.setattr(
        reporting,
        "paired_lineage_bootstrap",
        lambda data, method, baseline, metric: bootstrap(
            data, method, baseline, metric, repetitions=2
        ),
    )
    monkeypatch.setattr(reporting, "interaction_analysis", lambda data: {"fixture": True})
    monkeypatch.setattr(
        reporting,
        "costs",
        lambda data, refs: {"unique_rows": len(data), "reference_rows": len(refs)},
    )
    monkeypatch.setattr(reporting, "plot", lambda *_: None)
    monkeypatch.setattr(reporting, "plot_mechanisms", lambda *_: None)
    result = reporting.report()
    assert (result["runs"], result["reused_references"], result["logical_comparison_rows"]) == (
        1880,
        600,
        2480,
    )
    assert result["cost"] == {"unique_rows": 1880, "reference_rows": 600}
    assert len(result["source_conditions"]["all_sources"]["summary"]) == 7
    assert len(result["source_conditions"]["strict_leave_family"]["summary"]) == 5
    assert result["mechanisms"]["summary"]["gcrt"]["unique_collisions"]["valid_runs"] == 40
    assert result["mechanisms"]["summary"]["gcrt"]["collision_recall"]["valid_runs"] == 35
    assert "interaction" not in result["mechanisms"]
    assert len(result["equal_collision_count_behavior_coverage"]) == 400


def test_interaction_report_shares_budget_and_does_not_discover_a_collision_from_its_safe_group_member(
    monkeypatch,
):
    scenes = sample("cut_in_interaction", 512, "new")
    for index, gap in ((0, 20.1), (1, 20.1), (2, 20.2), (3, 20.2)):
        scenes[index] = replace(
            scenes[index],
            parameters={**scenes[0].parameters, "gap_m": gap},
            categories={
                "side": scenes[0].categories["side"],
                "adjacent_space": "open" if index % 2 == 0 else "restricted",
            },
        )
    private = {
        "feedback": [{"collision": index == 0} for index in range(512)],
        "events": [{"brake_latency_from_scheduled_event_s": 0.2} for _ in range(512)],
        "collision_partners": ["cut_in", *[None for _ in range(511)]],
    }
    queries = [{"index": i, "feedback": {"collision": False}} for i in range(2, 130)]
    monkeypatch.setattr(reporting, "sample", lambda *_: scenes)
    monkeypatch.setattr(
        reporting,
        "read_json",
        lambda path: (
            private if "evaluation_banks" in path.parts else {"result": {"queries": queries}}
        ),
    )
    rows = [
        {
            **next(
                t
                for t in tasks()
                if t["target_id"] == sut.sut_id
                and t["scenario"] == "cut_in_interaction"
                and t["condition"] == "all_sources"
                and t["algorithm_seed"] == 11
            ),
            "method": "gcrt",
        }
        for sut in INDEPENDENT_SUTS
    ]
    result = reporting.interaction_analysis(rows)
    for sut in INDEPENDENT_SUTS:
        allocated = [r for r in result["paid_allocations"] if r["target_id"] == sut.sut_id]
        assert sum(r["queries_allocated"] for r in allocated) == 128
        opened = next(r for r in allocated if r["adjacent_space"] == "open")
        restricted = next(r for r in allocated if r["adjacent_space"] == "restricted")
        assert opened["pool_unique_collisions"] == 1 and opened["unique_collisions"] == 0
        assert opened["recall_under_joint_budget"] == 0
        assert (
            restricted["pool_unique_collisions"] == 0
            and restricted["recall_under_joint_budget"] is None
        )
