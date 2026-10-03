from pathlib import Path

import numpy as np
import torch
import yaml

from replications.scenariofuzz_highway_env.scenariofuzz.corpus import ScenarioSpec, build_default_corpus
from replications.scenariofuzz_highway_env.scenariofuzz.filter import select_candidates
from replications.scenariofuzz_highway_env.scenariofuzz.graph_builder import build_graph, stack_graphs
from replications.scenariofuzz_highway_env.scenariofuzz.mutators import generate_candidates
from replications.scenariofuzz_highway_env.scenariofuzz.sem_model import ScenarioEvaluationModel
from replications.scenariofuzz_highway_env.scenariofuzz.sem_training import _train_model


CONFIG_PATH = Path("replications/scenariofuzz_highway_env/configs/online.yaml")


def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_scenario_id_is_parameter_based():
    a = ScenarioSpec.create(10.0, -2.0, "fast_intrusion")
    b = ScenarioSpec.create(10, -2, "fast_intrusion")
    assert a.scenario_id == b.scenario_id


def test_random_and_neighbor_mutation_are_bounded_and_unique():
    seed = build_default_corpus(config())[0]
    rng = np.random.default_rng(7)
    broad = generate_candidates(seed, 100, rng)
    reference = broad.candidates[0]
    local = generate_candidates(seed, 100, rng, reference=reference, gap_step=.5, speed_step=.25)
    assert len({candidate.scenario_id for candidate in broad.candidates}) == len(broad.candidates)
    assert all(seed.validate(candidate)[0] for candidate in local.candidates)
    assert all(abs(candidate.initial_gap - reference.initial_gap) <= 2.5 + 1e-6 or candidate.initial_gap in seed.gap_bounds for candidate in local.candidates)


def test_graph_has_distinct_edge_entity_line_graph_and_no_outcomes():
    seed = build_default_corpus(config())[0]
    graph = build_graph(seed, ScenarioSpec.create(18, -4, "cutin_braking"))
    graph.validate()
    assert graph.edge_features.shape[0] == graph.edge_index.shape[1]
    assert graph.line_adjacency.shape[0] == graph.edge_features.shape[0]
    feature_names = " ".join((*__import__("replications.scenariofuzz_highway_env.scenariofuzz.graph_builder", fromlist=["NODE_FEATURE_NAMES"]).NODE_FEATURE_NAMES,))
    assert "collision" not in feature_names and "ttc" not in feature_names


def test_sem_returns_logits_and_backpropagates():
    seed = build_default_corpus(config())[0]
    graphs = [build_graph(seed, ScenarioSpec.create(10 + i, -4, "fast_intrusion")) for i in range(3)]
    batch = stack_graphs(graphs)
    model = ScenarioEvaluationModel()
    logits = model(batch)
    assert logits.shape == (3,)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_threshold_policy_does_not_execute_rejected_candidates():
    candidates = [ScenarioSpec.create(10 + i, -2, "fast_intrusion") for i in range(4)]
    indices, reason = select_candidates(candidates, np.asarray([.1, .51, .9, .3]), 3, .5, np.random.default_rng(1), "resample")
    assert indices == [2, 1]
    assert reason == "threshold_descending"
    indices, reason = select_candidates(candidates, np.asarray([.1, .2, .3, .4]), 3, .5, np.random.default_rng(1), "resample")
    assert indices == [] and reason == "empty_filter_resample"


def test_sem_initialization_is_reproducible_for_history_size():
    seed = build_default_corpus(config())[0]
    graphs = [build_graph(seed, ScenarioSpec.create(10 + i, -4 + i / 10, "fast_intrusion")) for i in range(6)]
    tensors = stack_graphs(graphs)
    labels = torch.tensor([0., 1., 0., 1., 0., 1.])
    training_config = {
        "random_seed": 17, "hidden": 8, "heads": 2, "dropout": .1,
        "learning_rate": 1e-3, "weight_decay": 1e-4,
        "epochs": 3, "early_stopping_patience": 3,
    }
    first, _, _ = _train_model(
        training_config, tensors, labels, np.asarray([0, 1, 2, 3]), np.asarray([4, 5]), "cpu", "4",
    )
    second, _, _ = _train_model(
        training_config, tensors, labels, np.asarray([0, 1, 2, 3]), np.asarray([4, 5]), "cpu", "4",
    )
    assert all(torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in first.state_dict())
