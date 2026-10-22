from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from benchmark.continuous_execution import ContinuousExecutor
from benchmark.continuous_selection import select
from benchmark.continuous_space import SearchSpace, initial_scenarios
from benchmark.query import PaidExecution, QueryLedger, run_isolated
from benchmark.scenarios import DOMAINS, difficult, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.baselines.deep_hyperion.search import DeepHyperion
from methods.baselines.indago.search import Indago
from methods.baselines.pooled_ft.method import CollisionNetwork
from methods.baselines.samota.search import CONFIG as SAMOTA_CONFIG, Committee, preference_order
from methods.common.response import ResponseArchive, ResponseEncoder


def archive(name, replicate):
    scenes = sample("front_brake", 40, "history", replicate=replicate)
    return ResponseArchive(name, "rule", name, "history", scenes,
                           np.arange(40) % 3 == 0, np.ones((40, 2)))


@pytest.mark.parametrize("method", ["history_response", "indago", "samota", "deep_hyperion", "random"])
def test_continuous_methods_generate_and_pay_for_new_coordinates(method):
    torch.set_num_threads(1)
    initial = initial_scenarios("front_brake", 11, 4)
    def execute(scene):
        collision = scene.parameters["gap_m"] < 20
        return PaidExecution(QueryFeedback(scene.scenario_id, collision,
                                           SafetyMeasurements(0 if collision else 10, 1), 0.1), [])
    ledger = QueryLedger(initial, 8, execute)
    public = {"initial": initial, "method": method, "seed": 11, "budget": 8}
    if method == "history_response":
        public.update(history=[archive("a", 0), archive("b", 1)],
                      encoder=ResponseEncoder(6).eval().requires_grad_(False),
                      configuration={"proposal_count": 8, "global_fraction": 0.5, "horizon": 3})
    elif method == "indago":
        public.update(classifier=CollisionNetwork(6),
                      configuration={"neighborhood_size": 8, "max_neighborhoods": 3})
    elif method == "samota":
        public["configuration"] = {**SAMOTA_CONFIG, "population_size": 6,
                                    "generations": 2, "local_generations": 2}
    result = run_isolated(select, public, ledger)
    assert result["actual_queries"] == len(ledger.records) == 8
    assert len({s.coordinate() for s in ledger.candidates}) == 8
    assert [q["index"] for q in result["queries"][:4]] == [0, 1, 2, 3]
    assert len(ledger.candidates) == 8
    if method == "history_response":
        assert all("source_weights" in q for q in result["queries"])


def test_map_elites_keeps_the_best_executed_individual_in_each_cell():
    space = SearchSpace("front_brake", 23)
    search = DeepHyperion(space)
    a = space.random()
    b = replace(a, scenario_id="different_execution")
    safe = QueryFeedback(a.scenario_id, False, SafetyMeasurements(5, 5), 1)
    failed = QueryFeedback(b.scenario_id, True, SafetyMeasurements(0, 0), 1)
    search.observe(a, safe)
    search.observe(b, failed)
    assert len(search.archive) == 1
    assert search.archive[search.cell(a)][1] == b


def test_samota_has_three_surrogates_and_preserves_objective_preference():
    generator = np.random.default_rng(11)
    x, y = generator.random((12, 6)), generator.random((12, 3))
    committee = Committee(x, y, generator)
    mean, uncertainty = committee.predict(x[:3])
    assert len(committee.models) == 3
    assert mean.shape == uncertainty.shape == (3, 3)
    np.testing.assert_allclose(committee.weights.sum(0), 1)
    assert np.all(uncertainty >= 0)
    values = np.array([[0, 3], [3, 0], [1, 1], [4, 4]])
    assert set(preference_order(values, [0, 1])[:2]) == {0, 1}


def test_continuous_prefix_reuses_only_the_same_already_paid_sequence(tmp_path):
    scenes = initial_scenarios("front_brake", 11, 2)
    executor = ContinuousExecutor("history_idm_mobil", tmp_path, {"method": "fixture", "seed": 11})
    expected = [executor(scene).feedback for scene in scenes]
    resumed = ContinuousExecutor("history_idm_mobil", tmp_path, {"method": "fixture", "seed": 11})
    for scene, feedback in zip(scenes, expected):
        assert resumed(scene).feedback == feedback
    assert len(list(tmp_path.glob("attempt_*.json"))) == 2
    wrong = ContinuousExecutor("history_idm_mobil", tmp_path, {"method": "fixture", "seed": 11})
    with pytest.raises(ValueError, match="diverged"):
        wrong(scenes[1])


def test_indago_follows_classifier_saliency_without_updating_weights():
    space = SearchSpace("front_brake", 11)
    classifier = torch.nn.Linear(6, 1)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.weight[0, 1] = -5
        classifier.bias.zero_()
    # Indago accepts a scalar logit for each sample, as supplied by CollisionNetwork.
    model = CollisionNetwork(6)
    model.network = torch.nn.Sequential(classifier)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    search = Indago(space, model, config={"neighborhood_size": 8, "max_neighborhoods": 3})
    scene = space.random()
    coordinate, direction = search.saliency(scene)
    assert coordinate == 1 and direction < 0
    neighbor = space.mutate(scene, coordinate=coordinate, direction=direction)
    assert search.logits([neighbor])[0] >= search.logits([scene])[0]
    search.propose(5)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_actual_continuous_run_reconstructs_paid_metrics_and_reuses_completed_run(tmp_path):
    from benchmark.continuous_runs import run_task
    arguments = {"method": "random", "algorithm_seed": 11, "budget": 4, "initial_count": 2,
                 "destination": tmp_path / "run.json", "metadata": {"target_id": "evaluator_only"}}
    result = run_task("history_idm_mobil", "front_brake", (), **arguments)
    assert result["metrics"]["queries"] == result["cost"]["actual_physical_calls"] == 4
    assert len(result["candidates"]) == 4
    assert "evaluator_only" not in json.dumps(result["result"])
    assert run_task("history_idm_mobil", "front_brake", (), **arguments) == result
    assert len(list((tmp_path / "run").glob("attempt_*.json"))) == 4


@pytest.mark.parametrize("kind", DOMAINS)
def test_continuous_development_excludes_held_difficult_combinations(kind):
    space = SearchSpace(kind, 11, ordinary_only=True)
    points = [space.random() for _ in range(100)]
    initial = initial_scenarios(kind, 23, ordinary_only=True)
    assert not any(difficult(scene) for scene in points + initial)
    held = sample(kind, 1, "hard")[0]
    assert not space.valid(held)
    assert SearchSpace(kind, 11).valid(held)
