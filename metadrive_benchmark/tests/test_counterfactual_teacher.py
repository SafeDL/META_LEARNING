from __future__ import annotations

import numpy as np
import pytest

from metadrive_benchmark.mining.types import CutInDesign
from metadrive_benchmark.formal.counterfactual_teacher import CONTINUATION_POLICIES, CounterfactualTeacher
from metadrive_benchmark.formal.state import TeacherWorld


def _teacher() -> CounterfactualTeacher:
    pool = tuple(CutInDesign(index % 2, (index / 10.0, ) * 5) for index in range(6))
    world = TeacherWorld(
        pool=pool,
        base_mean=np.asarray((0.2, 0.3, 0.4, 0.5, 0.6, 0.7)),
        bases=np.asarray(((1.0, ), (0.5, ), (0.25, ), (-0.25, ), (-0.5, ), (-1.0, ))),
        noise=np.full(6, 0.02),
        formal_scores=np.asarray(((0.0, 0.5, 0.0, 1.0, 0.0, 0.5), )),
        responses=np.asarray(((0.1, 0.8, 0.4, 0.9, 0.3, 0.7), )),
        evaluability=np.ones(6),
        proxy_event_threshold=0.75,
    )
    return CounterfactualTeacher(world)


def test_terminal_teacher_values_are_exact_immediate_rewards() -> None:
    teacher = _teacher()
    value = teacher.action_value(teacher.initial_state(), 0, 3, remaining_budget=1)
    assert value.q_formal == 1.0
    assert value.q_vulnerability == 0.9
    assert value.best_continuation == "terminal"


def test_counterfactual_state_updates_posterior_and_never_repeats_selected_case() -> None:
    teacher = _teacher()
    state = teacher.initial_state()
    before = state.posterior.mean.copy()
    state.observe(1, 0.8, 0.5)
    assert not np.allclose(before, state.posterior.mean)
    assert not state.available_mask()[1]
    with pytest.raises(ValueError):
        state.observe(1, 0.8, 0.5)
    with pytest.raises(ValueError):
        teacher.action_value(state, 0, 1, remaining_budget=2)


def test_teacher_portfolio_is_not_worse_than_each_continuation_policy() -> None:
    teacher = _teacher()
    state = teacher.initial_state()
    candidate = 0
    value = teacher.action_value(state, 0, candidate, remaining_budget=4)
    returns = []
    for policy in CONTINUATION_POLICIES:
        branch = state.clone()
        teacher._observe_oracle(branch, 0, candidate)
        future_formal, future_vulnerability = teacher._continuation(branch, 0, 3, policy)
        returns.append((future_formal, future_vulnerability))
    assert value.q_formal >= teacher.world.formal_scores[0, candidate] + max(row[0]
                                                                             for row in returns)
    assert value.q_vulnerability >= teacher.world.responses[0, candidate] + max(row[1]
                                                                                for row in returns)


def test_teacher_policy_obeys_exact_fixed_budget_without_duplicates() -> None:
    teacher = _teacher()
    evaluation = teacher.run(0, total_budget=6, policy="teacher")
    assert len(evaluation.selected_indexes) == 6
    assert len(set(evaluation.selected_indexes)) == 6
