from __future__ import annotations

import numpy as np
import pytest

from metadrive_sim_env.mining.types import CutInDesign
from metadrive_sim_env.formal.counterfactual_teacher import CounterfactualTeacher
from metadrive_sim_env.formal.formal_counterfactual_teacher import (
    CONTINUATION_POLICIES,
    FormalCounterfactualTeacher,
    FormalTeacherEvaluation,
)
from metadrive_sim_env.formal.formal_calibrator import FormalEventCalibrator
from metadrive_sim_env.formal.state import TeacherWorld
from metadrive_sim_env.scripts.validate_formal_teacher import REPOSITORY_ROOT, _jaccard


class _MeanCalibrator:
    training_source_indexes = (0, 1)

    @staticmethod
    def predict_proba(state):
        mean, _ = state.predicted_vulnerability()
        collision = np.clip(mean, 0.0, 1.0)
        near = 0.25 * (1.0 - collision)
        normal = 1.0 - collision - near
        return np.column_stack((normal, near, collision))

    def expected_utility(self, state):
        return self.predict_proba(state) @ np.asarray((0.0, 0.5, 1.0))


def test_validator_resolves_the_repository_after_package_moves() -> None:
    assert (REPOSITORY_ROOT / "metadrive_sim_env" / "configs" / "formal_teacher.yaml").is_file()


def _world(count: int = 24) -> TeacherWorld:
    pool = tuple(
        CutInDesign(index % 2, ((index / max(count - 1, 1)) * 2.0 - 1.0,) * 5)
        for index in range(count)
    )
    base = np.linspace(0.1, 0.9, count)
    basis = np.linspace(-0.8, 0.8, count)[:, None]
    patterns = []
    for offset in (0, 3, 6):
        pattern = np.zeros(count)
        pattern[(offset + 1) % count] = 0.5
        pattern[(offset + 2) % count] = 1.0
        patterns.append(pattern)
    formal = np.asarray(patterns)
    responses = np.clip(base[None, :] + np.asarray((-.15, 0.0, .15))[:, None], 0.0, 1.0)
    responses[formal == 0.5] = 0.8
    responses[formal == 1.0] = 1.0
    return TeacherWorld(
        pool=pool,
        base_mean=base,
        bases=basis,
        noise=np.full(count, 0.02),
        formal_scores=formal,
        responses=responses,
        evaluability=np.ones(count),
        proxy_event_threshold=0.75,
    )


def test_formal_calibrator_is_fold_local_and_produces_valid_probabilities() -> None:
    world = _world()
    calibrator = FormalEventCalibrator.fit(
        world, (0, 1), context_sizes=(0, 1), max_iterations=200
    )
    probabilities = calibrator.predict_proba(CounterfactualTeacher(world).initial_state())
    assert calibrator.training_source_indexes == (0, 1)
    assert probabilities.shape == (24, 3)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    changed_held_out = TeacherWorld(
        pool=world.pool,
        base_mean=world.base_mean,
        bases=world.bases,
        noise=world.noise,
        formal_scores=np.vstack((world.formal_scores[:2], np.roll(world.formal_scores[2], 7))),
        responses=np.vstack((world.responses[:2], np.roll(world.responses[2], 7))),
        evaluability=world.evaluability,
        proxy_event_threshold=world.proxy_event_threshold,
    )
    repeated = FormalEventCalibrator.fit(
        changed_held_out, (0, 1), context_sizes=(0, 1), max_iterations=200
    )
    assert np.allclose(calibrator.weights, repeated.weights)


def test_auc_teacher_prefers_earlier_reward_when_terminal_return_ties() -> None:
    world = _world(2)
    world.formal_scores[0] = (1.0, 0.0)
    world.responses[0] = (1.0, 0.1)
    teacher = FormalCounterfactualTeacher(CounterfactualTeacher(world), _MeanCalibrator())
    state = teacher.initial_state()
    early = teacher.action_value(state, 0, 0, remaining_budget=2)
    late = teacher.action_value(state, 0, 1, remaining_budget=2)
    assert early.q_terminal_formal == late.q_terminal_formal == 1.0
    assert early.q_auc_formal > late.q_auc_formal


def test_formal_terminal_identity_and_portfolio_values() -> None:
    world = _world(6)
    teacher = FormalCounterfactualTeacher(CounterfactualTeacher(world), _MeanCalibrator())
    state = teacher.initial_state()
    terminal = teacher.action_value(state, 0, 2, remaining_budget=1)
    assert terminal.q_auc_formal == world.formal_scores[0, 2]
    assert terminal.q_auc_vulnerability == world.responses[0, 2]
    assert terminal.normalized_q_auc_formal == terminal.q_auc_formal
    assert terminal.normalized_q_auc_vulnerability == terminal.q_auc_vulnerability

    value = teacher.action_value(state, 0, 0, remaining_budget=4)
    returns = []
    for policy in CONTINUATION_POLICIES:
        branch = state.clone()
        teacher._observe_oracle(branch, 0, 0)
        returns.append(teacher._continuation(branch, 0, 3, policy))
    assert value.q_auc_formal >= 4 * world.formal_scores[0, 0] + max(row[2] for row in returns)
    assert value.q_auc_vulnerability >= 4 * world.responses[0, 0] + max(row[3] for row in returns)


def test_formal_mining_responds_to_diagnostic_observation() -> None:
    world = _world(8)
    teacher = FormalCounterfactualTeacher(CounterfactualTeacher(world), _MeanCalibrator())
    state = teacher.initial_state()
    before = teacher.formal_mining_values(state)
    selected = state.select_index("diagnostic")
    teacher._observe_oracle(state, 2, selected)
    after = teacher.formal_mining_values(state)
    assert not np.allclose(before, after)
    assert not state.available_mask()[selected]


@pytest.mark.parametrize("budget", (10, 20))
def test_formal_replay_obeys_both_exact_budgets(budget: int) -> None:
    world = _world(24)
    teacher = FormalCounterfactualTeacher(CounterfactualTeacher(world), _MeanCalibrator())
    evaluation = teacher.run(0, budget, "fixed_k4_formal")
    assert len(evaluation.selected_indexes) == budget
    assert len(set(evaluation.selected_indexes)) == budget


def test_formal_curve_metrics_and_overlap_are_exact() -> None:
    evaluation = FormalTeacherEvaluation(
        policy="toy",
        source_index=0,
        total_budget=4,
        formal_curve=(0.0, 1.0, 1.0, 1.5),
        vulnerability_curve=(0.1, 1.1, 1.3, 2.1),
        selected_indexes=(0, 1, 2, 3),
    )
    assert evaluation.score_at(4) == 1.5
    assert evaluation.auc_at(4) == 3.5
    assert evaluation.first_critical_step == 2
    assert _jaccard({1, 2}, {2, 3}) == pytest.approx(1.0 / 3.0)

    oracle_rewards = np.asarray((1.0, 0.5, 0.0, 0.0))
    baseline_rewards = np.asarray((0.5, 0.0, 1.0, 0.0))
    oracle_curve = np.cumsum(oracle_rewards)
    baseline_curve = np.cumsum(baseline_rewards)
    assert oracle_curve[-1] - baseline_curve[-1] == 0.0
    assert oracle_curve[:3].sum() - baseline_curve[:3].sum() == 1.5
