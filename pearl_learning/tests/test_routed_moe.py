from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from pearl_learning.src.checkpoint import CHECKPOINT_SCHEMA, METHOD_CONTRACT, load_checkpoint, save_checkpoint
from pearl_learning.src.collector import collect_episode
from pearl_learning.src.io import read_config
from pearl_learning.src.moe import (
    DESCRIPTOR_FIELDS,
    DESCRIPTOR_SCHEMA,
    PosteriorRoutedMoEActor,
    PosteriorRouter,
    physical_task_descriptor,
)
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.replay import Transition


def small_config(*, architecture: str = "posterior_routed_moe", routing: str = "soft",
                 top_k: int = 2) -> dict:
    config = read_config("pearl_learning/configs/posterior_routed_moe.yaml")
    config = copy.deepcopy(config)
    config["experiment"]["device"] = "cpu"
    config["networks"]["actor_architecture"] = architecture
    config["networks"]["context_hidden_sizes"] = [16]
    config["networks"]["actor_hidden_sizes"] = [32, 32]
    config["networks"]["critic_hidden_sizes"] = [32, 32]
    if architecture == "posterior_routed_moe":
        config["networks"]["moe"]["routing"] = routing
        config["networks"]["moe"]["top_k"] = top_k
        config["networks"]["moe"]["router_hidden_sizes"] = [16]
        config["networks"]["moe"]["expert_hidden_size"] = 16
    return config


def descriptor(config: dict, *, curvature: float = 0.01):
    topology = {
        "adversary_lane_count": 4.0,
        "sut_lane_count": 4.0,
        "merge_length_m": 32.0,
        "conflict_radius_m": 4.0,
        "adversary_route_curvature": curvature,
        "sut_route_curvature": curvature / 2.0,
    }
    moe = config["networks"]["moe"]
    return physical_task_descriptor(
        topology,
        schema=moe["descriptor_schema"],
        normalization=moe["descriptor_normalization"],
    )


def transition(episode: str, *, final: bool = False) -> Transition:
    return Transition(
        np.zeros(37, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        0.1,
        np.ones(37, dtype=np.float32) * 0.1,
        False,
        final,
        "horizon" if final else "running",
        "task",
        episode,
        "case",
        "prior_support",
        0,
    )


def update_once(agent: PEARLAgent, config: dict) -> dict[str, float]:
    contexts = []
    batches = []
    descriptors = []
    for index in range(2):
        rows = [transition(f"context-{index}") for _ in range(4)]
        contexts.append([rows])
        batches.append([transition(f"rl-{index}") for _ in range(8)])
        descriptors.append(descriptor(config, curvature=0.01 + index * 0.01))
    return agent.update(contexts, batches, task_descriptors=descriptors, posterior_versions=[1, 1])


def test_router_soft_and_top_k_numerical_contracts():
    descriptor_tensor = torch.randn(3, len(DESCRIPTOR_FIELDS))
    mu = torch.randn(3, 5, requires_grad=True)
    log_var = torch.zeros(3, 5, requires_grad=True)
    soft = PosteriorRouter(len(DESCRIPTOR_FIELDS), 5, 3, 3, "soft", [12])
    soft_output = soft(descriptor_tensor, mu, log_var)
    assert torch.all(soft_output.weights >= 0)
    assert torch.allclose(soft_output.weights.sum(-1), torch.ones(3))
    assert torch.all(soft_output.weights > 0)
    sparse = PosteriorRouter(len(DESCRIPTOR_FIELDS), 5, 3, 2, "top_k", [12])
    sparse_output = sparse(descriptor_tensor, mu, log_var)
    assert torch.equal((sparse_output.weights == 0).sum(-1), torch.ones(3, dtype=torch.long))
    assert torch.allclose(sparse_output.weights.sum(-1), torch.ones(3))
    sparse_output.weights.square().sum().backward()
    assert mu.grad is None and log_var.grad is None


@pytest.mark.parametrize("routing,top_k", [("soft", 1), ("top_k", 0), ("top_k", 4)])
def test_router_rejects_illegal_top_k(routing: str, top_k: int):
    with pytest.raises(ValueError):
        PosteriorRouter(len(DESCRIPTOR_FIELDS), 5, 3, top_k, routing, [8])


def test_router_and_descriptor_fail_fast_on_nonfinite_or_schema_mismatch():
    config = small_config()
    values = torch.zeros(1, len(DESCRIPTOR_FIELDS))
    values[0, 0] = torch.nan
    router = PosteriorRouter(len(DESCRIPTOR_FIELDS), 5, 2, 2, "soft", [8])
    with pytest.raises(ValueError, match="finite"):
        router(values, torch.zeros(1, 5), torch.zeros(1, 5))
    with pytest.raises(ValueError, match="schema"):
        physical_task_descriptor(
            {field: 1.0 for field in DESCRIPTOR_FIELDS},
            schema="wrong",
            normalization=config["networks"]["moe"]["descriptor_normalization"],
        )


def test_descriptor_is_allowlisted_hashed_and_rule_independent():
    config = small_config()
    first = descriptor(config)
    second = descriptor(config)
    assert first == second
    assert first.schema == DESCRIPTOR_SCHEMA
    assert first.fields == DESCRIPTOR_FIELDS
    assert len(first.content_hash) == 64
    serialized = str(first.audit_dict()).lower()
    for forbidden in ("task_id", "geometry_id", "logical_type", "split", "priority", "rule", "route_remaining"):
        assert forbidden not in serialized


def test_moe_actor_matches_dense_action_contract():
    actor = PosteriorRoutedMoEActor(37, 5, 2, [32, 32], 2, 16)
    observation = torch.randn(7, 37)
    latent = torch.randn(7, 5)
    weights = torch.full((7, 2), 0.5)
    deterministic, deterministic_logp = actor.sample(observation, latent, weights, True)
    stochastic, stochastic_logp = actor.sample(observation, latent, weights, False)
    assert deterministic.shape == stochastic.shape == (7, 2)
    assert deterministic_logp.shape == stochastic_logp.shape == (7, 1)
    assert torch.all(deterministic.abs() <= 1.0)
    assert torch.all(stochastic.abs() <= 1.0)
    assert torch.isfinite(deterministic_logp).all() and torch.isfinite(stochastic_logp).all()


def test_route_is_reproducible_versioned_and_query_independent():
    config = small_config()
    torch.manual_seed(9)
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    mu, log_var = agent.prior()
    physical = descriptor(config)
    first = agent.compute_route(physical, mu, log_var, 0)
    repeated = agent.compute_route(physical, mu, log_var, 0)
    next_version = agent.compute_route(physical, mu, log_var, 1)
    assert first == repeated
    assert first.weights == repeated.weights
    assert first.route_hash == repeated.route_hash
    assert first.route_hash != next_version.route_hash
    assert first.query_free and not first.gradient_enabled


def test_actor_update_gradient_boundaries_and_optimizer_coverage():
    config = small_config()
    torch.manual_seed(3)
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    metrics = update_once(agent, config)
    assert metrics["router_gradient_norm"] > 0.0
    assert metrics["actor_shared_gradient_norm"] > 0.0
    assert metrics["actor_head_gradient_norm"] > 0.0
    assert metrics["expert_0_gradient_norm"] > 0.0
    assert metrics["expert_1_gradient_norm"] > 0.0
    assert metrics["context_encoder_actor_gradient_norm"] == 0.0
    assert metrics["critic_actor_gradient_norm"] == 0.0
    assert metrics["critic_phase_actor_unchanged"] == 1.0
    assert metrics["actor_phase_critic_unchanged"] == 1.0
    assert np.isfinite(list(metrics.values())).all()
    assert len(agent.last_router_audits) == 2
    assert all(row["gradient_enabled"] and row["query_free"] for row in agent.last_router_audits)


def test_top_k_inactive_expert_has_exactly_zero_gradient():
    actor = PosteriorRoutedMoEActor(37, 5, 2, [16], 2, 8)
    observation = torch.randn(4, 37)
    latent = torch.randn(4, 5)
    weights = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    actor.sample(observation, latent, weights)[0].sum().backward()
    active = sum(float(parameter.grad.abs().sum()) for parameter in actor.residual_experts[0].parameters())
    inactive = sum(float(parameter.grad.abs().sum()) for parameter in actor.residual_experts[1].parameters())
    assert active > 0.0
    assert inactive == 0.0


class TinyEnv:
    def __init__(self):
        self.steps = 0
        self.case_id = ""

    def reset(self, *, options):
        self.steps = 0
        self.case_id = str(options["case"]["case_id"])
        return np.zeros(37, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        truncated = self.steps == 3
        return np.zeros(37, dtype=np.float32), 0.0, False, truncated, {
            "termination_reason": "horizon" if truncated else "running"
        }

    def episode_record(self):
        return {"case_id": self.case_id, "episode_length": self.steps}


def test_collector_keeps_one_route_for_whole_episode_and_records_hash():
    config = small_config()
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    mu, log_var = agent.prior()
    route = agent.compute_route(descriptor(config), mu, log_var, 2)
    task = type("Task", (), {"task_id": "task"})()
    rollout = collect_episode(
        TinyEnv(), task, {"case_id": "query-a"}, agent, mu, "deterministic_query",
        torch.device("cpu"), episode_id="episode", posterior_version=2, route_context=route,
    )
    assert {row.posterior_version for row in rollout.transitions} == {2}
    assert rollout.record["route_hash"] == route.route_hash
    second = collect_episode(
        TinyEnv(), task, {"case_id": "query-b"}, agent, mu, "deterministic_query",
        torch.device("cpu"), episode_id="episode-2", posterior_version=2, route_context=route,
    )
    assert second.record["route_hash"] == rollout.record["route_hash"]


def test_posterior_sampled_query_uses_deterministic_conditional_policy():
    config = small_config(architecture="dense")
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    mu, log_var = agent.prior()
    first_latent = agent.sample_latent_seeded(mu, log_var, 91)
    repeated_latent = agent.sample_latent_seeded(mu, log_var, 91)
    other_latent = agent.sample_latent_seeded(mu, log_var, 92)
    assert torch.equal(first_latent, repeated_latent)
    assert not torch.equal(first_latent, other_latent)
    task = type("Task", (), {"task_id": "task"})()
    rollout = collect_episode(
        TinyEnv(), task, {"case_id": "query"}, agent, first_latent,
        "posterior_sampled_query", torch.device("cpu"), episode_id="episode",
        posterior_version=0,
    )
    assert rollout.record["collection_mode"] == "posterior_sampled_query"


def checkpoint_roundtrip(architecture: str):
    config = small_config(architecture=architecture)
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    if architecture == "posterior_routed_moe":
        update_once(agent, config)
    before_hash = agent.parameter_hash()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        rng_state = {
            "numpy_generator": np.random.default_rng(4).bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
        save_checkpoint(
            path, agent, config, "taskbook", 12,
            casebook_hashes={"task": "casebook"}, training_seed=4,
            rng_state=rng_state, trainer_state={"marker": 7},
        )
        restored = PEARLAgent(37, 2, config, torch.device("cpu"))
        payload = load_checkpoint(path, restored, torch.device("cpu"))
        assert payload["schema"] == CHECKPOINT_SCHEMA
        assert payload["method_contract"] == METHOD_CONTRACT
        assert restored.parameter_hash() == before_hash
        assert payload["trainer_state"] == {"marker": 7}
        assert payload["architecture"] == agent.architecture_metadata()
        if architecture == "posterior_routed_moe":
            mu, log_var = agent.prior()
            route = agent.compute_route(descriptor(config), mu, log_var, 0)
            restored_route = restored.compute_route(descriptor(config), mu, log_var, 0)
            observation = torch.zeros(1, 37)
            assert route == restored_route
            assert torch.equal(agent.act(observation, mu, True, route), restored.act(observation, mu, True, restored_route))
        return path


@pytest.mark.parametrize("architecture", ["dense", "posterior_routed_moe"])
def test_dense_and_moe_checkpoint_roundtrip(architecture: str):
    checkpoint_roundtrip(architecture)


def test_checkpoint_rejects_architecture_mismatch():
    moe_config = small_config()
    agent = PEARLAgent(37, 2, moe_config, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        save_checkpoint(
            path, agent, moe_config, "taskbook", 0,
            casebook_hashes={}, training_seed=1,
            rng_state={"numpy_generator": np.random.default_rng().bit_generator.state,
                       "torch": torch.get_rng_state(), "cuda": None},
            trainer_state={},
        )
        dense = PEARLAgent(37, 2, small_config(architecture="dense"), torch.device("cpu"))
        with pytest.raises(ValueError, match="architecture"):
            load_checkpoint(path, dense, torch.device("cpu"))


def test_checkpoint_rejects_retired_method_contract_without_fallback():
    config = small_config(architecture="dense")
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "incompatible.pt"
        save_checkpoint(
            path, agent, config, "taskbook", 0,
            casebook_hashes={}, training_seed=1,
            rng_state={"numpy_generator": np.random.default_rng().bit_generator.state,
                       "torch": torch.get_rng_state(), "cuda": None},
            trainer_state={},
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("method_contract")
        torch.save(payload, path)
        with pytest.raises(ValueError, match="incompatible checkpoints"):
            load_checkpoint(path, agent, torch.device("cpu"))
