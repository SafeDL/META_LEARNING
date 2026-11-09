import copy

import numpy as np
import pytest

from experiments.query import PaidExecution, QueryLedger, run_isolated
from experiments.scenarios import candidate_pool
from experiments.schema import QueryFeedback, SafetyMeasurements
from testing_methods.baselines.behavexplor.behavior import (
    BehaviorMiner,
    motion_trace,
    temporal_features,
    trace_distance,
)
from testing_methods.baselines.behavexplor.pool import Seed, mutate, run


def trajectory(*, speed=10, duration=0.4, collision=False):
    frames = []
    for time in np.arange(0, duration + 0.025, 0.05):
        frames.append(
            {
                "time_s": float(time),
                "collision": collision and time >= duration - 1e-8,
                "actors": [
                    {
                        "actor_id": "ego",
                        "x": speed * time,
                        "y": 4,
                        "speed": speed,
                        "heading": 0,
                        "length": 5,
                        "width": 2,
                    },
                    {
                        "actor_id": "lead",
                        "x": speed * time + 20,
                        "y": 4,
                        "speed": speed,
                        "heading": 0,
                        "length": 5,
                        "width": 2,
                    },
                ],
            }
        )
    return frames


def test_temporal_statistics_match_definitions_and_pad_short_terminal_windows():
    result = temporal_features(np.array([[1.0], [2.0], [3.0]]), window=3)
    np.testing.assert_allclose(result[0], [2, 2 / 3, 1, 3, 1, 1, 6, np.sqrt(3)])
    np.testing.assert_allclose(result[-1], [3, 0, 3, 3, 0, 0, 27, 0])
    assert result.shape == (3, 8)
    singleton = temporal_features(np.array([[2.0, 0.0]]))
    assert singleton.shape == (1, 16) and np.isfinite(singleton).all()
    constant = motion_trace(trajectory())
    assert constant.shape == (5, 8)
    np.testing.assert_allclose(constant[:, [0, 2]], 10)
    np.testing.assert_allclose(constant[:, [1, 3, 4, 5, 6, 7]], 0)
    assert temporal_features(constant).shape == (5, 64)
    with pytest.raises(ValueError, match="trajectory"):
        motion_trace(None)


def test_abstract_distance_includes_duration_and_reclustering_relabels_previous_traces():
    assert trace_distance(np.array([1, 2, 2]), np.array([1, 1])) == pytest.approx(2 / 3)
    assert trace_distance(np.array([1, 2]), np.array([1, 2])) == 0
    miner = BehaviorMiner(clusters=2)
    first = np.array([[0.0], [0.1], [0.2]])
    second = np.array([[8.0], [9.0], [10.0], [9.0]])
    miner.initialize([first])
    miner.add(second)
    for x, abstract in zip(miner.features, miner.traces):
        np.testing.assert_array_equal(abstract, miner.model.predict(x))
    assert miner.feedback(second) == (False, 0.0)
    assert miner.rebuilds == 2


def test_author_energy_update_counts_novel_benign_mutants_and_bounded_improvement():
    parent = Seed(0, 1)
    parent.update(collision=False, novel=False, improvement=0, distance=0, local=True)
    assert parent.energy == pytest.approx(1.4)
    parent.update(collision=False, novel=True, improvement=0, distance=0.8, local=True)
    assert parent.energy == pytest.approx(1.6)
    parent.update(collision=True, novel=False, improvement=1, distance=1, local=True)
    assert parent.energy == pytest.approx(1.6 + 0.5 * 2 / 3 + 0.5 - 0.05)
    old = parent.energy
    parent.update(collision=False, novel=False, improvement=-1, distance=1, local=False)
    assert parent.energy == pytest.approx(old - 0.1)
    assert parent.local_trials == 3


@pytest.mark.parametrize("kind", ("car_following", "front_brake", "cut_out", "lane_change_rear"))
def test_both_mutations_remain_in_the_same_functional_domain(kind):
    scene = candidate_pool(kind, 8)[0]
    generator = np.random.default_rng(11)
    for local in (True, False):
        mutated = mutate(scene, generator, local=local)
        assert mutated.kind == kind
        assert mutated.parameters.keys() == scene.parameters.keys()


class Oracle:
    def __init__(self, candidates, *, all_fail=False, altered_after=None):
        self.candidates = candidates
        self.all_fail = all_fail
        self.altered_after = altered_after
        self.calls = []

    def query(self, index, **kwargs):
        self.calls.append(index)
        collision = self.all_fail or (index % 7 == 0)
        speed = 8 + index % 4
        if self.altered_after is not None and len(self.calls) > self.altered_after:
            speed += 8
        return QueryFeedback(
            self.candidates[index].scenario_id,
            collision,
            SafetyMeasurements(0 if collision else 0.5 + 0.5 * (index % 2), 1),
            0.1,
            trajectory=trajectory(speed=speed, collision=collision),
        )


def test_fixed_budget_initialization_all_failure_corpus_and_future_trace_isolation():
    scenes = candidate_pool("car_following", 205, replicate=99)
    initial = list(range(20))
    failed = run(scenes, Oracle(scenes, all_fail=True), initial_indices=initial, budget=200)
    assert failed["selected_indices"][:20] == initial
    assert len(set(failed["selected_indices"])) == 200
    assert not failed["corpus_indices"]
    assert all(r["reason"] == "empty_corpus_uniform" for r in failed["queries"][20:])
    plain = run(scenes, Oracle(scenes), initial_indices=initial, budget=28)
    changed = run(scenes, Oracle(scenes, altered_after=24), initial_indices=initial, budget=28)
    assert plain["selected_indices"][:25] == changed["selected_indices"][:25]
    for a, b in zip(plain["queries"][:24], changed["queries"][:24]):
        for key in ("behavior_distance", "parent_energy_after", "corpus_size"):
            assert a[key] == b[key]
    assert all(not r["feedback"]["collision"] for r in plain["queries"] if r["retained_seed"])


def test_selector_runs_through_the_shared_paid_trajectory_interface():
    scenes = candidate_pool("front_brake", 9, replicate=99)
    oracle = Oracle(scenes)

    def execute(scene):
        feedback = oracle.query(scenes.index(scene))
        return PaidExecution(feedback, copy.deepcopy(feedback.trajectory))

    ledger = QueryLedger(scenes, 6, execute, joint_feedback=True)
    result = run_isolated(run, dict(candidates=scenes, initial_indices=[1, 2], budget=6), ledger)
    assert result["selected_indices"] == [r["index"] for r in ledger.records]
    assert len(result["selected_indices"]) == 6
    assert all("trajectory" in q["feedback"] for q in result["queries"])
