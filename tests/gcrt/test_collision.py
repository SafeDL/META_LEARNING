import copy
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm
import torch

from benchmark.query import PaidExecution, QueryLedger, ReplayExecutor, run_isolated
from benchmark.scenarios import DOMAINS, difficult, features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.common.classification import logistic_moments, torch_condition
from methods.common.response import ResponseArchive, ResponseEncoder
from methods.gcrt.config import ModelConfig
from methods.gcrt.posterior import CollisionPosterior
from methods.gcrt.prior import posterior
from methods.gcrt.selection import acquisition_values, choose, select
from methods.gcrt.train import episode, episode_loss, load_encoder, train
from methods.common.model_contract import validate_model_contract


def model(size=12):
    x = np.linspace(0, 1, size)
    covariance = np.exp(-((x[:, None] - x[None]) / 0.25)**2)
    means = np.array([2 * x - 1, 1 - 2 * x, np.zeros(size)])
    return CollisionPosterior(means, np.broadcast_to(covariance, (3, size, size)),
                              [0.375, 0.375, 0.25])


def archive(source, *, role="history", family="rule", seed=0):
    scenes = sample("front_brake", 56, "history" if role == "history" else "development",
                    replicate=seed)
    y = np.arange(len(scenes)) % 3 == 0
    return ResponseArchive(source, family, source, role, scenes, y,
                           np.column_stack((np.linspace(-1, 60, len(y)), np.ones(len(y)))))


def test_quadrature_matches_direct_bernoulli_integration():
    mean, variance = 0.8, 1.7
    expected = quad(lambda z: expit(z) * norm.pdf(z, mean, np.sqrt(variance)),
                    -20, 20)[0]
    actual = logistic_moments(mean, variance)[0]
    assert actual == pytest.approx(expected, abs=2e-9)


def test_virtual_and_paid_collision_updates_match_without_mutation():
    state = model()
    state.observe(0, True)
    before = copy.deepcopy(state)
    future = np.array([3, 5, 8])
    for label in (False, True):
        prediction = state.conditional_probabilities(2, label, future)
        clone = copy.deepcopy(state)
        clone.observe(2, label)
        np.testing.assert_allclose(prediction, clone.probabilities(future), atol=1e-13)
    np.testing.assert_array_equal(state.covariance, before.covariance)
    np.testing.assert_array_equal(state.mean, before.mean)
    np.testing.assert_array_equal(state.weights, before.weights)
    assert state.observed == {0}


def test_weights_use_prequery_collision_likelihood():
    state = model()
    expected = state.weights * state.component_probabilities([0])[:, 0]
    expected /= expected.sum()
    state.observe(0, True)
    np.testing.assert_allclose(state.weights, expected)
    with pytest.raises(TypeError):
        state.observe(1, 0.9)
    with pytest.raises(ValueError):
        state.observe(0, True)


def test_numpy_and_differentiable_updates_agree_and_stay_positive():
    state = model(140)
    mean = torch.tensor(state.mean, requires_grad=True)
    covariance = torch.tensor(state.covariance)
    log_weight = torch.tensor(state.log_weight)
    for index in range(128):
        label = index % 4 == 0
        state.observe(index, label)
        if index < 3:
            mean, covariance, evidence = torch_condition(mean, covariance, index, label)
            log_weight += evidence
            log_weight -= torch.logsumexp(log_weight, 0)
            np.testing.assert_allclose(mean.detach(), state.mean, atol=1e-11)
            np.testing.assert_allclose(covariance.detach(), state.covariance, atol=1e-11)
    assert np.linalg.eigvalsh(state.covariance).min() > -1e-9
    assert np.isfinite(state.probabilities()).all()
    mean.sum().backward()
    assert mean.grad_fn is not None


def test_binary_lookahead_and_identifier_tie_break():
    state = model(8)
    available = np.arange(8)
    q = state.probabilities()
    np.testing.assert_allclose(acquisition_values(state, available, 1), q)
    actual = acquisition_values(state, available, 3)
    expected = []
    for index in available:
        future = available[available != index]
        value = q[index]
        for label, mass in ((False, 1 - q[index]), (True, q[index])):
            clone = copy.deepcopy(state)
            clone.observe(int(index), label)
            value += mass * np.sort(clone.probabilities(future))[-2:].sum()
        expected.append(value)
    np.testing.assert_allclose(actual, expected)
    tied = CollisionPosterior(np.zeros((1, 2)), np.eye(2)[None], [1])
    assert choose(tied, [0, 1], ["z", "a"], 1)[0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA comparison needs a GPU")
@pytest.mark.parametrize("fixed", [False, True])
def test_gpu_all_candidate_binary_lookahead_matches_scalar_cpu_updates(fixed):
    state = model(28)
    state.fixed_weights = fixed
    state.observe(1, True)
    state.observe(4, False)
    before = copy.deepcopy(state)
    available = np.array([i for i in range(28) if i not in state.observed])
    cpu = acquisition_values(state, available, 5, device="cpu")
    gpu = acquisition_values(state, available, 5, device="cuda")
    np.testing.assert_allclose(gpu, cpu, atol=1e-12)
    np.testing.assert_array_equal(before.mean, state.mean)
    np.testing.assert_array_equal(before.covariance, state.covariance)


def test_initialization_repeats_and_failed_execution_are_metered():
    scenes = sample("front_brake", 8, "development")
    def execute(s):
        return PaidExecution(QueryFeedback(s.scenario_id, False,
                                           SafetyMeasurements(2, 3), 0.01), [])
    ledger = QueryLedger(scenes, 3, execute)
    ledger.query(0, purpose="initial")
    with pytest.raises(ValueError):
        ledger.query(0)
    ledger.query(0, purpose="repeat")
    ledger.query(1)
    with pytest.raises(ValueError):
        ledger.query(2)
    assert len(ledger.records) == 3
    historical = QueryLedger(scenes, 2, execute, account="history_certification")
    historical.query(0, purpose="history_certification")
    historical.query(0, purpose="repeat")
    assert len(historical.records) == 2
    assert all(r["account"] == "history_certification" for r in historical.records)
    def fail(s):
        raise RuntimeError("simulator failure")
    failed = QueryLedger(scenes, 1, fail)
    with pytest.raises(RuntimeError):
        failed.query(0)
    assert failed.records[0]["status"] == "failed"
    with pytest.raises(ValueError):
        failed.query(1)


def test_process_isolation_and_collision_not_safety_threshold():
    scenes = sample("front_brake", 12, "development")
    # Deliberately contradict risk: huge safety margin but a measured collision.
    executions = {s.scenario_id: PaidExecution(
        QueryFeedback(s.scenario_id, i % 2 == 0, SafetyMeasurements(90, 10), 0.02),
        [{"private_evaluator_truth": "must not be sent"}]) for i, s in enumerate(scenes)}
    ledger = QueryLedger(scenes, 8, ReplayExecutor(executions))
    result = run_isolated(select, {"model": model(), "candidates": scenes,
                                  "initial_indices": [0, 1], "budget": 8}, ledger)
    assert result["actual_queries"] == 8
    assert result["queries"][0]["feedback"]["collision"] is True
    assert result["selected_indices"][:2] == [0, 1]
    assert result["queries"][-1]["horizon"] == 1
    assert "private_evaluator_truth" not in repr(result)
    assert sum(r["feedback"]["execution_s"] for r in ledger.records) == pytest.approx(0.16)


@pytest.mark.parametrize("kind", DOMAINS)
def test_named_domains_and_coordinate_holdouts(kind):
    history = sample(kind, 32, "history")
    development = sample(kind, 32, "development")
    new = sample(kind, 32, "new")
    hard = sample(kind, 16, "hard")
    assert not any(difficult(s) for s in history + development + new)
    assert all(difficult(s) for s in hard)
    assert not {s.coordinate() for s in history} & {s.coordinate() for s in new + development}
    assert features(history).shape == (32, DOMAINS[kind].dimension)
    assert history == sample(kind, 32, "history")


def test_episodic_context_never_uses_simulated_targets_query_labels():
    archives = [archive("a"), archive("b", family="search", seed=1)]
    contexts, indices = episode(archives, archives[0], np.random.default_rng(3), 4, 8)
    assert [a.source_id for a in contexts] == ["b"]
    encoder = ResponseEncoder(6)
    x = archives[0].x[indices]
    before = encoder.covariance(x, contexts).detach().clone()
    archives[0].collision[indices[4:]] = ~archives[0].collision[indices[4:]]
    torch.testing.assert_close(before, encoder.covariance(x, contexts))
    loss, metrics = episode_loss(encoder, contexts, archives[0], indices, 4)
    loss.backward()
    assert encoder.context_input.weight.grad.abs().sum() > 0
    assert np.isfinite(metrics["collision_nll"])
    with pytest.raises(ValueError, match="Held family"):
        train(archives, [archive("dev", role="development")], "unused.pt", held_family="search")


def test_training_builds_independent_collision_checkpoint(tmp_path):
    history = [archive("a"), archive("b", seed=1)]
    development = [archive("dev", role="development", seed=2)]
    config = replace(ModelConfig(), steps=2, validation_interval=1, support_counts=(4,),
                     query_count=8)
    destination = tmp_path / "collision.pt"
    report = train(history, development, destination, config=config)
    encoder, checkpoint = load_encoder(destination)
    state = posterior(encoder, history[0].x[:10], history)
    assert checkpoint["format"] == "collision-bernoulli"
    assert checkpoint["history_ids"] == ["a", "b"]
    assert report["status"] == "complete"
    assert state.mean.shape == (3, 10)
    assert not any(p.requires_grad for p in encoder.parameters())
    assert not list(tmp_path.glob("**/*stop*"))
    validate_model_contract(checkpoint, "history_response", configuration=checkpoint["config"])
    with pytest.raises(ValueError, match="configuration changed"):
        validate_model_contract(checkpoint, "history_response", configuration={"steps": 99})
    checkpoint["model_contract"]["source_files"]["methods/common/response.py"] = "retired_code"
    stale = tmp_path / "stale.pt"
    torch.save(checkpoint, stale)
    with pytest.raises(ValueError, match="construction code changed"):
        load_encoder(stale)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA numerical equivalence needs a GPU"))])
def test_batched_response_sources_match_independent_outputs_and_gradients(device):
    torch.manual_seed(11)
    contexts = [archive("a"), archive("b", seed=1), archive("c", seed=2)]
    contexts[1] = ResponseArchive("b", "rule", "b", "history", contexts[1].scenarios[:24],
                                   contexts[1].collision[:24], contexts[1].safety[:24])
    encoder = ResponseEncoder(6).to(device)
    reference = copy.deepcopy(encoder)
    x = contexts[0].x[:12]
    query = reference.query_input(reference.tensor(x))[None]
    scalar = []
    for context in contexts:
        indices = np.linspace(0, len(context.scenarios) - 1, min(64, len(context.scenarios)), dtype=int)
        tokens = np.column_stack((context.x[indices], context.collision[indices], context.auxiliary[indices]))
        values = reference.context_encoder(reference.context_input(reference.tensor(tokens))[None])
        attended, _ = reference.attention(query, values, values, need_weights=False)
        scalar.append(torch.nn.functional.normalize(reference.output((query + attended)[0]), dim=-1))
    expected = torch.stack(scalar)
    actual = encoder.encode(x, contexts)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for name, parameter in encoder.named_parameters():
        previous = dict(reference.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, previous.grad, rtol=1e-4, atol=1e-5)
