from __future__ import annotations

from pathlib import Path

import pytest

from meta_testing.policy.shared_features import SharedFeatureEncoder
from meta_testing.scenario.option import AdversarialOption
from meta_testing.scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from meta_testing.scenario.task_spec import MetaTestTaskSpec
from meta_testing.scenario.taskbook import load_taskbook
from meta_testing.sut.registry import default_registry
from meta_testing.training.pipeline import selected_tasks


HASH = "a" * 64


def test_generic_task_contract_and_strict_fields() -> None:
    task = MetaTestTaskSpec("task", "meta_train", "idm_cautious", "merge", "map", HASH, "template", "merge_v1", 1)
    assert MetaTestTaskSpec.from_dict(task.to_dict()) == task
    with pytest.raises(ValueError):
        MetaTestTaskSpec.from_dict({**task.to_dict(), "extra": True})


def test_sut_registry_and_identity_non_leakage() -> None:
    adapter, profile = default_registry().create("idm_defensive")
    assert adapter.metadata(profile)["profile_is_model_input"] is False
    assert profile.target_speed_mps == 10.0
    assert profile.acceleration_factor == 1.10
    assert profile.deceleration_factor == -3.0
    with pytest.raises(ValueError, match="SUT identity"):
        SharedFeatureEncoder.validate_metadata({"sut_ref": "idm_defensive"})


def test_hybrid_action_round_trip_and_invalid_action() -> None:
    space = ParameterSpace("space", ("a", "b"), {"speed": (1.0, 3.0)}, tuple(AdversarialOption))
    action = NormalizedScenarioAction(1, (0.0,), AdversarialOption.GAP_CLOSE)
    assert space.encode(space.decode(action)) == action
    with pytest.raises(ValueError):
        space.decode(NormalizedScenarioAction(2, (0.0,), AdversarialOption.GAP_CLOSE))


def test_default_idm_taskbook_covers_profile_split() -> None:
    tasks = load_taskbook("meta_testing/configs/idm_taskbook.json")
    assert len(tasks) == 18
    assert {task.sut_ref for task in tasks if task.split == "meta_train"} == {"idm_cautious", "idm_defensive", "idm_normal", "idm_assertive"}
    assert {task.sut_ref for task in tasks if task.split == "meta_test"} == {"idm_late_response"}


def test_evaluation_task_selection_respects_split_and_profile_key() -> None:
    config = {
        "training": {"family_filter": "all"},
        "sut_profiles": {"validation": ["idm_fast_small_gap"]},
    }
    tasks = selected_tasks(config, Path("meta_testing/configs/idm_taskbook.json"), "meta_validation", "validation")
    assert len(tasks) == 3
    assert {task.sut_ref for task in tasks} == {"idm_fast_small_gap"}
