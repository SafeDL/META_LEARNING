"""Behavioral checks for stopping, physical splits, and the PPO reset curriculum."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from experiments.regression.protocol import numeric_coordinate, repair_subset, sample_block
from experiments.rss import measurement_contract
from experiments.schema import QueryFeedback, SafetyMeasurements
from experiments.scenarios import DOMAINS
from simulator.highway.repair_env import RepairEnv, physical_coordinate
from testing_methods.history_transfer.selection import select
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


def test_partial_regression_cannot_promote_the_paper_cohort(tmp_path, monkeypatch):
    from experiments.regression import dqn_reports
    from experiments import paper_results
    from experiments.storage import write_json

    original = tmp_path / "k_shot_adaptation"
    revised = tmp_path / "dqn_continual_learning"
    monkeypatch.setattr(paper_results, "RUNS", tmp_path)
    monkeypatch.setattr(paper_results, "REVISED", revised)
    monkeypatch.setattr(paper_results, "PUBLICATION", revised / "publication.json")
    monkeypatch.setattr(dqn_reports, "STUDY", revised)
    write_json(revised / "policy_regression/reports/audit.json", {"status": "incomplete", "sequences": 36})
    with pytest.raises(ValueError, match="matched replays"):
        dqn_reports.publish()
    assert paper_results.paper_study(original) == original
    assert not paper_results.PUBLICATION.exists()
    assert paper_results.regression_policy("dqn") == "dqn"


def test_published_dqn_is_the_default_for_plotting_and_replay(tmp_path, monkeypatch):
    from experiments import paper_results

    revised = tmp_path / "dqn_continual_learning"
    revised.mkdir()
    receipt = revised / "publication.json"
    receipt.write_text('{"status":"published"}', encoding="utf-8")
    monkeypatch.setattr(paper_results, "RUNS", tmp_path)
    monkeypatch.setattr(paper_results, "PUBLICATION", receipt)
    monkeypatch.setattr(paper_results, "REVISED", revised)
    assert paper_results.paper_study(tmp_path / "budget_discovery") == revised / "budget_discovery"
    explicit = tmp_path / "independent_study" / "budget_discovery"
    assert paper_results.paper_study(explicit) == explicit
    assert paper_results.regression_policy("dqn") == "dqn_continual"
    assert paper_results.regression_policy("ppo") == "ppo"


def scenes(count=12):
    return sample_block("cut_in_onset", count, 31, set(), "test")


class PaidOracle:
    def __init__(self, candidates):
        self.candidates = candidates
        self.indices = []

    def query(self, index, *, purpose):
        assert index not in self.indices
        self.indices.append(index)
        return QueryFeedback(self.candidates[index].scenario_id, True,
                             SafetyMeasurements(0, 0, rss_margin=-0.5), 0,
                             rss_contract=measurement_contract())


def test_gp_stops_at_the_actual_threshold_without_extra_queries():
    candidates = scenes()
    oracle = PaidOracle(candidates)
    result = select(candidates, oracle, raw_kernel_type=SharedPrivateKernel,
                    initial_indices=[0], budget=8, collision_target=3, snapshot_counts=())
    assert len(oracle.indices) == result["actual_queries"] == 3
    assert result["termination"] == "collision_target_reached"
    assert result["selected_indices"] == oracle.indices
    assert all(np.isfinite(q["prequery_probability"]) for q in result["queries"])


def test_coordinates_are_unique_and_exclusions_ignore_categories_and_ids():
    used = set()
    first = sample_block("cut_in_onset", 30, 7, used, "first")
    second = sample_block("cut_in_onset", 30, 7, used, "second")
    assert len({numeric_coordinate(s) for s in first + second}) == 60
    assert len(used) == 60
    assert all(all(float(v).is_integer() for v in s.parameters.values()) for s in first + second)
    assert all(s.categories["side"] in DOMAINS[s.kind].categories["side"] for s in first)
    renamed = replace(first[0], scenario_id="different_id")
    assert physical_coordinate(renamed) == physical_coordinate(first[0])


def test_farthest_selection_is_stable_and_does_not_duplicate_reset_points():
    points = scenes()
    forward = repair_subset(points + points, 5)
    backward = repair_subset(list(reversed(points)), 5)
    assert forward == backward
    assert len({s.coordinate() for s in forward}) == 5


def test_stratified_rehearsal_retains_the_rare_lane_change_in_each_minibatch():
    from experiments.regression.dqn_learning import rehearsal_indices

    actions = torch.tensor([0] * 4 + [2] * 242 + [3] * 196 + [4] * 378)
    groups = [torch.where(actions == a)[0] for a in actions.unique()]
    for _ in range(10):
        indices = rehearsal_indices(len(actions), groups, "cpu")
        assert len(indices) == 64
        assert actions[indices[32:]].bincount(minlength=5).tolist() == [8, 0, 8, 8, 8]


def test_driving_course_splits_and_skill_specific_acceptance():
    from collections import Counter
    from experiments.regression.dqn_learning import legacy_rehearsal_courses, ordinary_courses, retains_skills
    from experiments.qualification import COURSES, DEVELOPMENT_COURSES

    previous = {physical_coordinate(c.scenario) for c in (*COURSES, *DEVELOPMENT_COURSES, *legacy_rehearsal_courses())}
    for split, per_skill in (("training", 32), ("development", 8), ("confirmation", 8)):
        block = ordinary_courses(split)
        coordinates = {physical_coordinate(c.scenario) for c in block}
        assert len(coordinates) == 5 * per_skill
        assert not coordinates & previous
        assert Counter(c.name for c in block) == {c.name: per_skill for c in COURSES}
        previous |= coordinates
    before = [{"course": "brake", "scenario_id": "a", "passed": True},
              {"course": "change", "scenario_id": "b", "passed": False}]
    after = [dict(row, passed=not row["passed"]) for row in before]
    assert retains_skills(before, before)
    assert not retains_skills(before, after)


def test_rehearsal_uses_physical_actions_and_persistent_speed_targets():
    from controllers.highway.learning import DiscreteDriver
    from experiments.regression.dqn_learning import action_labels, demonstration_loss
    from simulator.highway.observation import VehicleState, VisibleWorld

    world = VisibleWorld(0, VehicleState("ego", 100, 4, 25, 0), (), (4,))
    driver = DiscreteDriver()
    driver.observation(world)
    mask, intent = action_labels(driver, world, 2)
    assert mask == [True, True, True, False, False] and intent == 1
    braking, intent = action_labels(driver, world, 4)
    assert braking == [False, False, False, False, True] and intent == 4
    assert driver.target_speed == 25
    driver.target_speed = 0
    stopped, intent = action_labels(driver, world, 4)
    assert stopped == [True, True, True, False, True] and intent == 1
    # At this speed even increasing the zero target produces the same clipped braking command.
    # Its new persistent target differs, so it must still incur the margin.
    q = torch.tensor([[1., 0., 0., 0., -1.], [0., 2., 0., 3., 0.]], requires_grad=True)
    cost = demonstration_loss(q, torch.tensor([mask, stopped]))
    assert torch.isclose(cost, torch.tensor(.9))
    cost.backward()
    assert q.grad[0].abs().sum() == 0 and q.grad[1, 3] > 0


def test_curriculum_records_real_transitions_and_restores_random_stream():
    points = scenes(4)
    env = RepairEnv(points[:2], points[2:], set())
    other = RepairEnv(points[:2], points[2:], set())
    try:
        env.reset(seed=41)
        state = env.curriculum_state()["rng"]
        other.restore_random_state(state)
        left, _ = env.reset()
        right, _ = other.reset()
        np.testing.assert_equal(left, right)
        assert env.world.scenario == other.world.scenario
        env.step(1)
        assert sum(env.curriculum_state()["transitions"].values()) == 1
    finally:
        env.close()
        other.close()


def test_detour_uses_official_static_prioritizer_and_only_known_labels():
    from testing_methods.baselines.detour.method import prioritize

    candidates = scenes(30)
    feedback = PaidOracle(candidates).query(0, purpose="initial")
    history = [(scene, i % 3 == 0) for i, scene in enumerate(scenes(12))]
    left = prioritize(history, candidates, [(0, feedback)], 10)
    right = prioritize(history, candidates, [(0, feedback)], 10)
    assert left == right
    assert len(set(left)) == 10
    assert 0 not in left


def test_regression_random_selector_respects_the_shared_stop_in_isolation(tmp_path):
    from experiments.query import PaidExecution, run_isolated
    from experiments.regression.queries import CollisionStoppingLedger, random_reference_selector
    from experiments.storage import read_json, write_json

    candidates = scenes(130)
    physical_calls = []

    def physical(index):
        saved = read_json(tmp_path / f"{index}.json")
        return PaidExecution(QueryFeedback.from_dict(saved["feedback"]), saved["trajectory"])

    def execute(scene):
        index = candidates.index(scene)
        physical_calls.append(index)
        # Collision-only feedback is sufficient to check transport and metering.
        feedback = QueryFeedback(scene.scenario_id, True, SafetyMeasurements(0, 0), 0)
        trajectory = [{"time_s": 0, "collision": True}]
        write_json(tmp_path / f"{index}.json", {"feedback": feedback.to_dict(), "trajectory": trajectory})
        return PaidExecution(feedback, trajectory)

    ledger = CollisionStoppingLedger(candidates, 120, execute,
                                     checkpoint_path=tmp_path / "ledger.json",
                                     specification={"test": "collision-stop"},
                                     paid_execution=physical, allow_registration=False)
    public = {"candidates": candidates, "initial_indices": list(range(20)),
              "budget": 120}
    result = run_isolated(random_reference_selector, public, ledger)
    ledger.finish()
    assert len(physical_calls) == result["actual_queries"] == 100
    recovered = CollisionStoppingLedger(candidates, 120, execute,
                                        checkpoint_path=tmp_path / "ledger.json",
                                        specification={"test": "collision-stop"},
                                        paid_execution=physical, allow_registration=False)
    replay = run_isolated(random_reference_selector, public, recovered)
    recovered.finish()
    assert replay["selected_indices"] == result["selected_indices"]
    assert len(physical_calls) == 100


class ConstantRisk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        return self.logit.expand(len(data.weather_attr))


def test_fuzzing_stops_before_executing_more_of_a_planned_batch():
    from testing_methods.baselines.scenario_fuzz.pool import Config, run

    candidates = scenes(20)
    oracle = PaidOracle(candidates)
    result = run(candidates, oracle, classifier=ConstantRisk(), initial_indices=[0],
                 budget=10, collision_target=3, config=Config(mutants=8))
    assert result["actual_queries"] == len(oracle.indices) == 3
    assert result["termination"] == "collision_target_reached"
