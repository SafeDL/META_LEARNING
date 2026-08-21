from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from pearl_learning.src.evaluator import evaluate_fewshot
from pearl_learning.src.moe import DESCRIPTOR_FIELDS, PosteriorRouter
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.pearl_trainer import _training_context_episode_count
from pearl_learning.src.replay import TaskReplayBuffers
from pearl_learning.tests.test_routed_moe import descriptor, small_config, transition


@pytest.mark.parametrize(
    ("mode", "expected_input_dim"),
    [
        ("static", len(DESCRIPTOR_FIELDS)),
        ("posterior_mean", 5),
        ("static_posterior_mean", len(DESCRIPTOR_FIELDS) + 5),
        ("static_posterior_mean_logvar", len(DESCRIPTOR_FIELDS) + 10),
    ],
)
def test_router_input_ablations_have_explicit_dimensions(mode: str, expected_input_dim: int):
    router = PosteriorRouter(len(DESCRIPTOR_FIELDS), 5, 2, 2, "soft", [11], mode)
    first_linear = next(module for module in router.model if isinstance(module, torch.nn.Linear))
    assert first_linear.in_features == expected_input_dim
    assert tuple(router.input_fields)
    output = router(torch.zeros(3, len(DESCRIPTOR_FIELDS)), torch.zeros(3, 5), torch.zeros(3, 5))
    assert output.weights.shape == (3, 2)
    assert torch.allclose(output.weights.sum(-1), torch.ones(3))


def test_router_rejects_unknown_input_ablation():
    with pytest.raises(ValueError, match="input_mode"):
        PosteriorRouter(len(DESCRIPTOR_FIELDS), 5, 2, 2, "soft", [8], "task_id")


def test_route_interventions_are_public_versioned_and_normalized():
    config = small_config()
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    mu, log_var = agent.prior()
    source = agent.compute_route(descriptor(config), mu, log_var, 0)

    frozen = agent.intervene_route(source, posterior_version=4, mode="frozen_prior")
    uniform = agent.intervene_route(source, posterior_version=4, mode="uniform")
    knockout = agent.intervene_route(
        source, posterior_version=4, mode="expert_knockout", expert_index=0,
    )

    assert frozen.weights == pytest.approx(source.weights)
    assert frozen.posterior_version == 4
    assert frozen.source_route_hash == source.route_hash
    assert uniform.weights == pytest.approx([0.5, 0.5])
    assert knockout.weights == pytest.approx([0.0, 1.0])
    assert sum(knockout.weights) == pytest.approx(1.0)
    assert len({source.route_hash, frozen.route_hash, uniform.route_hash, knockout.route_hash}) == 4


def test_expert_action_audit_exposes_anonymous_expert_outputs():
    config = small_config()
    torch.manual_seed(19)
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    observations = torch.randn(6, 37)
    latent = torch.randn(6, 5)
    actions = agent.expert_action_means(observations, latent)
    assert actions.shape == (6, 2, 2)
    assert torch.isfinite(actions).all()
    assert torch.linalg.vector_norm(actions[:, 0] - actions[:, 1], dim=-1).mean() > 0


def test_internal_sac_path_freezes_context_encoder_and_keeps_it_unchanged():
    config = small_config(architecture="dense")
    config["ablation"] = {**config.get("ablation", {}), "no_context_training": True}
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in agent.context_encoder.parameters())
    before = [parameter.detach().clone() for parameter in agent.context_encoder.parameters()]
    contexts = [[[transition("context") for _ in range(4)]] for _ in range(2)]
    batches = [[transition(f"rl-{index}") for _ in range(8)] for index in range(2)]
    metrics = agent.update(contexts, batches)
    after = list(agent.context_encoder.parameters())
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    assert np.isfinite(list(metrics.values())).all()


def test_moe_sac_uses_static_router_and_frozen_context_encoder():
    config = small_config()
    config["ablation"] = {**config.get("ablation", {}), "no_context_training": True}
    config["networks"]["moe"]["input_mode"] = "static"
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    metadata = agent.architecture_metadata()
    assert metadata["moe"]["router_input_fields"] == list(DESCRIPTOR_FIELDS)
    assert not any(parameter.requires_grad for parameter in agent.context_encoder.parameters())


def test_dense_evaluator_rejects_route_intervention_before_environment_creation():
    config = small_config(architecture="dense")
    agent = PEARLAgent(37, 2, config, torch.device("cpu"))
    with pytest.raises(ValueError, match="MoE actor"):
        evaluate_fewshot(agent, config, [], {}, "meta_validation", query_route_mode="uniform")


def test_training_context_count_is_shape_safe_across_uneven_task_replay():
    buffers = TaskReplayBuffers(["a", "b"])
    for index in range(3):
        rows = [
            replace(transition(f"a-{index}"), task_id="a"),
            replace(transition(f"a-{index}", final=True), task_id="a"),
        ]
        buffers.add_episode("a", rows)
    for index in range(2):
        rows = [
            replace(transition(f"b-{index}"), task_id="b"),
            replace(transition(f"b-{index}", final=True), task_id="b"),
        ]
        buffers.add_episode("b", rows)
    count = _training_context_episode_count(
        buffers, ["a", "b"], 1, 8, np.random.default_rng(3),
    )
    assert count == 1
    contexts = buffers.context_per_task(["a", "b"], count * 4, 4, np.random.default_rng(4))
    assert [np.asarray(task).shape for task in contexts] == [(1, 4), (1, 4)]
