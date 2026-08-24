from types import SimpleNamespace

import pytest

from mvr.scenario.option import AdversarialOption
from mvr.scenario.parameter_space import ParameterSpace
from mvr.training.meta_sampler import MetaTaskSampler
from mvr.training.stage1_sampling import PretrainSceneSampler


def test_meta_task_sampler_returns_one_complete_shuffled_epoch() -> None:
    tasks = tuple(range(4))
    sampler = MetaTaskSampler(tasks)

    epoch = sampler.shuffled_epoch()

    assert set(epoch) == set(tasks)
    assert len(epoch) == len(tasks)


def test_pretrain_scene_sampler_balances_candidate_option_and_controls() -> None:
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
    assert [action.option for action in actions] == [
        AdversarialOption.APPROACH_CONFLICT,
        AdversarialOption.YIELD_THEN_PRESS,
        AdversarialOption.YIELD_THEN_PRESS,
        AdversarialOption.GAP_CLOSE,
    ]
    for action in actions:
        action.validate(space.continuous_dim)
        assert space.encode(space.decode(action)).continuous == pytest.approx(action.continuous)
