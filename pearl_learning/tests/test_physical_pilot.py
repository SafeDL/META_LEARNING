from __future__ import annotations

from dataclasses import replace

import pytest

from pearl_learning.scripts.audit_physical_task_policy_heterogeneity import (
    physical_heterogeneity_gate_report,
)
from pearl_learning.scripts.audit_vanilla_physical_adaptation import vanilla_adaptation_report
from pearl_learning.scripts.build_method_flow_casebooks_v2 import _active_case_splits
from pearl_learning.src.io import read_config
from pearl_learning.src.taskbook import build_taskbook, validate_physical_task_contract


def _matrix(aa: float, ba: float, ab: float, bb: float):
    return {
        "task_a": {
            "task_a": {"valid_critical_strict_rate": aa},
            "task_b": {"valid_critical_strict_rate": ab},
        },
        "task_b": {
            "task_a": {"valid_critical_strict_rate": ba},
            "task_b": {"valid_critical_strict_rate": bb},
        },
    }


def test_physical_gate_requires_both_vcsr_diagonal_advantages() -> None:
    passed = physical_heterogeneity_gate_report(
        _matrix(0.50, 0.25, 0.00, 0.25), ["task_a", "task_b"], minimum_advantage=0.25,
    )
    assert passed["status"] == "pass"
    single_side = physical_heterogeneity_gate_report(
        _matrix(0.50, 0.25, 0.25, 0.25), ["task_a", "task_b"], minimum_advantage=0.25,
    )
    assert single_side["status"] == "fail"
    zero = physical_heterogeneity_gate_report(
        _matrix(0.0, 0.0, 0.0, 0.0), ["task_a", "task_b"], minimum_advantage=0.25,
    )
    assert zero["status"] == "fail"
    assert zero["action_distance_is_diagnostic_only"] is True


def test_physical_contract_and_gate_case_splits() -> None:
    config = read_config("pearl_learning/configs/merge_method_flow_pilot.yaml")
    tasks = build_taskbook(config)["meta_train"]
    selected = {task.geometry_id: task for task in tasks if task.geometry_id in {
        "lane_drop_24", "lane_drop_32", "bottleneck_24", "bottleneck_32",
    }}
    assert set(selected) == {"lane_drop_24", "lane_drop_32", "bottleneck_24", "bottleneck_32"}
    for task in selected.values():
        validate_physical_task_contract(task, config)
        assert task.priority_spec["target_contact_entry_order"] == "any"
        assert task.priority_spec["target_contact_speed_relation"] == "any"
    assert _active_case_splits(selected["lane_drop_24"], config) == {"train_pool", "validation_query"}
    assert _active_case_splits(selected["bottleneck_32"], config) == {"train_pool", "validation_query"}
    assert _active_case_splits(selected["lane_drop_32"], config) == {"train_pool"}
    hidden = replace(
        selected["lane_drop_24"],
        task_id=selected["lane_drop_24"].task_id + "__rule_adversary_first",
        priority_spec={
            **selected["lane_drop_24"].priority_spec,
            "target_contact_entry_order": "adversary_first",
            "target_contact_entry_order_semantics": "pre_step_arrival_time",
        },
    )
    with pytest.raises(ValueError, match="hidden-rule"):
        validate_physical_task_contract(hidden, config)


def _vanilla_suite(k0: tuple[int, int], k4: tuple[int, int], *, step: int = 200_000):
    tasks = {}
    for index, task_id in enumerate(("validation_lane", "validation_bottleneck")):
        summary0 = {"episodes": 4, "valid_critical_strict_rate": k0[index] / 4}
        summary4 = {"episodes": 4, "valid_critical_strict_rate": k4[index] / 4}
        tasks[task_id] = {"0": {"summary": summary0}, "4": {"summary": summary4}}
    return {
        "split": "meta_validation",
        "evaluation_regimes": {
            "validation_known_logical_type": {
                "query_modes": {
                    "posterior_mean_deterministic": {
                        "provenance": {"checkpoint_step": step},
                        "tasks": tasks,
                    },
                },
            },
        },
    }


def test_vanilla_decision_uses_fixed_200k_k4_and_one_aggregate_case() -> None:
    config = read_config("pearl_learning/configs/merge_method_flow_pilot.yaml")
    decision = config["vanilla_pilot_decision"]
    assert vanilla_adaptation_report(_vanilla_suite((0, 0), (1, 0)), decision)["status"] == "pass"
    assert vanilla_adaptation_report(_vanilla_suite((1, 0), (1, 0)), decision)["status"] == "fail"
    with pytest.raises(ValueError, match="fixed step"):
        vanilla_adaptation_report(_vanilla_suite((0, 0), (1, 0), step=175_000), decision)
