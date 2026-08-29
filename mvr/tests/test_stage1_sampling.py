from types import SimpleNamespace

import pytest

from mvr.scenario.parameter_space import ParameterSpace
from mvr.training.meta_sampler import MetaTaskSampler
from mvr.training.stage1_sampling import PretrainSceneSampler


def test_meta_task_sampler_returns_one_complete_shuffled_epoch() -> None:
    tasks = tuple(range(4))
    sampler = MetaTaskSampler(tasks)

    epoch = sampler.shuffled_epoch()

    assert set(epoch) == set(tasks)
    assert len(epoch) == len(tasks)


def test_pretrain_scene_sampler_balances_candidate_and_controls() -> None:
    tasks = tuple(SimpleNamespace(task_id=f"task-{index}") for index in range(2))
    space = ParameterSpace(
        "sampling-test",
        ("candidate-0", "candidate-1", "candidate-2"),
        {
            "adversary_distance_to_conflict_m": (0.5, 5.0),
            "sut_distance_to_conflict_m": (0.5, 5.0),
            "adversary_speed_mps": (4.0, 20.0),
            "sut_speed_mps": (4.0, 20.0),
        },
    )
    candidates = tuple(
        SimpleNamespace(
            adversary_distance_available_m=10.0,
            sut_distance_available_m=10.0,
        )
        for _ in space.candidates
    )
    sampler = PretrainSceneSampler(tasks, episodes_per_task=2, seed=11)

    actions = [
        sampler(task, episode_index, candidates, space)
        for task in tasks
        for episode_index in range(2)
    ]

    assert [action.candidate_index for action in actions] == [0, 1, 1, 2]
    for action in actions:
        action.validate(space.continuous_dim)
        assert space.encode(space.decode(action)).continuous == pytest.approx(action.continuous)


def test_pretrain_scene_sampler_aligns_reachable_candidate_arrivals() -> None:
    task = SimpleNamespace(task_id="merge-task", functional_scenario="merge")
    space = ParameterSpace(
        "sampling-test",
        ("candidate-0",),
        {
            "adversary_distance_to_conflict_m": (0.5, 5.0),
            "sut_distance_to_conflict_m": (0.5, 5.0),
            "adversary_initial_speed_mps": (4.0, 18.0),
            "sut_initial_speed_mps": (4.0, 18.0),
        },
    )
    candidate = SimpleNamespace(
        adversary_distance_min_m=0.0,
        adversary_distance_available_m=26.0,
        sut_distance_min_m=0.0,
        sut_distance_available_m=52.0,
    )

    action = PretrainSceneSampler((task,), episodes_per_task=1, seed=11)(
        task, 0, (candidate,), space
    )
    config = space.decode(action)
    adversary_distance = candidate.adversary_distance_min_m + 0.5 * (
        action.continuous[0] + 1.0
    ) * (candidate.adversary_distance_available_m - candidate.adversary_distance_min_m)
    sut_distance = candidate.sut_distance_min_m + 0.5 * (action.continuous[1] + 1.0) * (
        candidate.sut_distance_available_m - candidate.sut_distance_min_m
    )
    assert config["adversary_initial_speed_mps"] == pytest.approx(8.3)
    assert config["sut_initial_speed_mps"] == pytest.approx(8.3)
    assert sut_distance / 8.3 - adversary_distance / 8.3 == pytest.approx(1.5)
