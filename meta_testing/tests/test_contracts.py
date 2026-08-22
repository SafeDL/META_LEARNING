from __future__ import annotations

import pytest

from meta_testing.policy.shared_features import SharedFeatureEncoder
from meta_testing.scenario.option import AdversarialOption
from meta_testing.scenario.parameter_space import NormalizedScenarioAction, ParameterSpace
from meta_testing.scenario.task_spec import MetaTestTaskSpec
from meta_testing.scenario.taskbook import load_taskbook
from meta_testing.sut.registry import default_registry


HASH = "a" * 64


def test_generic_task_contract_and_strict_fields() -> None:
    task = MetaTestTaskSpec("task", "meta_train", "idm_cautious", "merge", "map", HASH, "template", "merge_v1", 1)
    assert MetaTestTaskSpec.from_dict(task.to_dict()) == task
    with pytest.raises(ValueError):
        MetaTestTaskSpec.from_dict({**task.to_dict(), "extra": True})


def test_sut_registry_and_identity_non_leakage() -> None:
    adapter, profile = default_registry().create("idm_defensive")
    assert adapter.metadata(profile)["profile_is_model_input"] is False
    assert profile.acceleration_factor == 0.80
    assert profile.deceleration_factor == -3.5
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
