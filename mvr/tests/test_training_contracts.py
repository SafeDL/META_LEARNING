from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mvr.model import TransferableScenarioMiner
from mvr.state import PhysicalStateExtractor
from mvr.policy.adversarial_sac import AdversarialSAC, _Actor
from mvr.scripts.plot_inner_sac_training import _recorded_stages
from mvr.training.replay import InnerReplay, OuterRolloutBuffer, OuterRolloutStep
from mvr.training.stages import TrainingStage, trainable_components
from mvr.training.trainers import _training_signal_metrics
from mvr.training.updates import update_outer_ppo
from mvr.training.workflow import StagedWorkflow


def test_stage_ownership_and_universal_on_policy_ppo() -> None:
    model = TransferableScenarioMiner(state_dim=PhysicalStateExtractor.dimension, map_dim=8)
    workflow = StagedWorkflow(model.training_components())
    workflow.activate(TrainingStage.OUTER)
    assert trainable_components(TrainingStage.OUTER) == {"task_structure_encoder", "universal_scene_policy"}
    assert all(parameter.requires_grad for parameter in model.universal_scene_policy.parameters())
    assert not any(parameter.requires_grad for parameter in model.inner_sac.parameters())
    policy = model.universal_scene_policy
    action = policy.sample(torch.zeros(8), torch.zeros(2, 8), torch.ones(2, dtype=torch.bool), torch.zeros(1, 16))
    buffer = OuterRolloutBuffer()
    buffer.add(OuterRolloutStep(
        torch.zeros(8), torch.zeros(2, 8), torch.ones(2, dtype=torch.bool), torch.ones(5, dtype=torch.bool), torch.tensor([[-1.0, 1.0]] * 5), torch.zeros(16),
        action.expert_index.squeeze(0).detach(), action.candidate_index.squeeze(0).detach(),
        action.continuous.squeeze(0).detach(),
        action.log_prob.squeeze(0).detach(), action.value.squeeze(0).detach(), 1.0, True,
    ))
    buffer.finish()
    loss = update_outer_ppo(policy, buffer, torch.optim.Adam(policy.parameters(), lr=1e-3), epochs=1, batch_size=1)
    assert torch.isfinite(torch.tensor(loss))


def test_outer_action_masks_inactive_logical_dimensions() -> None:
    policy = TransferableScenarioMiner(state_dim=PhysicalStateExtractor.dimension, map_dim=8).universal_scene_policy
    action = policy.sample(
        torch.zeros(8), torch.zeros(2, 8), torch.ones(2, dtype=torch.bool), torch.zeros(1, 16),
        continuous_mask=torch.tensor([True, True, True, True, False]),
    )
    assert action.continuous.shape == (1, 5)
    assert action.continuous[0, -1].item() == 0.0


def test_sac_actor_objective_does_not_backpropagate_into_critics() -> None:
    sac = AdversarialSAC(feature_dim=4, action_dim=2)
    losses = sac.losses(
        torch.randn(3, 4), torch.randn(3, 2).tanh(), torch.randn(3),
        torch.randn(3, 4), torch.zeros(3, dtype=torch.bool),
    )
    losses.actor.backward()
    assert all(parameter.grad is None for parameter in sac.critic1.parameters())
    assert all(parameter.grad is None for parameter in sac.critic2.parameters())


def test_sac_actor_uses_standard_tanh_change_of_variables() -> None:
    actor = _Actor(feature_dim=3, action_dim=2)
    features = torch.zeros(4, 3)
    torch.manual_seed(17)
    action, log_prob = actor.sample(features)
    torch.manual_seed(17)
    normal = actor.distribution(features)
    raw = normal.rsample()
    expected_action = raw.tanh()
    expected_log_prob = (
        normal.log_prob(raw).sum(-1)
        - torch.log(1.0 - expected_action.square() + 1e-6).sum(-1)
    )

    torch.testing.assert_close(action, expected_action)
    torch.testing.assert_close(log_prob, expected_log_prob)


def test_sac_bellman_target_is_not_clamped() -> None:
    sac = AdversarialSAC(feature_dim=4, action_dim=2)
    target = sac.critic_target(
        torch.tensor([50.0]),
        torch.zeros(1, 4),
        torch.ones(1, dtype=torch.bool),
    )

    torch.testing.assert_close(target, torch.tensor([50.0]))


def test_sac_bellman_target_uses_per_transition_smdp_discount() -> None:
    class ConstantCritic(torch.nn.Module):
        def forward(self, features, _action):
            return features[:, 0] * 0.0 + 4.0

    sac = AdversarialSAC(feature_dim=4, action_dim=2)
    sac.target1 = ConstantCritic()
    sac.target2 = ConstantCritic()
    with torch.no_grad():
        sac.log_alpha.fill_(-50.0)

    target = sac.critic_target(
        torch.tensor([1.0, 2.0]),
        torch.zeros(2, 4),
        torch.tensor([False, True]),
        bootstrap_discount=torch.tensor([0.99 ** 5, 0.99 ** 2]),
    )

    torch.testing.assert_close(
        target, torch.tensor([1.0 + (0.99 ** 5) * 4.0, 2.0]),
        atol=1e-6, rtol=1e-6,
    )


def test_interaction_prior_freeze_is_opt_in_and_cutin_scoped() -> None:
    model = TransferableScenarioMiner(state_dim=PhysicalStateExtractor.dimension, map_dim=8)
    workflow = StagedWorkflow(model.training_components())

    frozen = workflow.activate(
        TrainingStage.INTERACTION_PRIOR, freeze_static_representation=True
    )

    assert frozen == {"shared_feature_encoder", "inner_sac"}
    for name in ("map_encoder", "interaction_encoder", "task_structure_encoder"):
        assert not any(parameter.requires_grad for parameter in model.training_components()[name].parameters())
    for name in frozen:
        assert all(parameter.requires_grad for parameter in model.training_components()[name].parameters())

    default = workflow.activate(TrainingStage.INTERACTION_PRIOR)
    assert {"map_encoder", "interaction_encoder", "task_structure_encoder"} <= default


def test_sac_actor_uses_unclamped_critic_value() -> None:
    class ConstantCritic(torch.nn.Module):
        def forward(self, _features, action):
            return action.sum(dim=-1) * 0.0 + 50.0

    sac = AdversarialSAC(feature_dim=4, action_dim=2)
    sac.critic1 = ConstantCritic()
    sac.critic2 = ConstantCritic()
    with torch.no_grad():
        sac.log_alpha.fill_(-20.0)

    actor, _ = sac.actor_alpha_losses(torch.zeros(8, 4))

    assert actor.item() < -40.0


def test_sac_event_action_anchor_is_finite_and_keeps_critics_frozen() -> None:
    sac = AdversarialSAC(feature_dim=4, action_dim=2)
    actor, _ = sac.actor_alpha_losses(
        torch.randn(4, 4),
        actions=torch.tensor([[-0.75, -0.75]] * 4),
        rewards=torch.tensor([4.0, -0.1, -0.1, -0.1]),
        event_action_weight=2.0,
    )
    actor.backward()
    assert torch.isfinite(actor)
    assert all(parameter.grad is None for parameter in sac.critic1.parameters())
    assert all(parameter.grad is None for parameter in sac.critic2.parameters())


def test_inner_replay_reserves_positive_event_transitions() -> None:
    replay = InnerReplay()
    for index in range(12):
        replay.add(SimpleNamespace(episode_id=str(index), reward=1.0 if index < 4 else -0.1))

    rows = replay.sample(8, positive_fraction=0.5)

    assert len(rows) == 8
    assert sum(float(row.reward) > 0.0 for row in rows) == 4


def test_inner_replay_does_not_treat_dense_risk_shaping_as_an_event() -> None:
    replay = InnerReplay()
    for index in range(12):
        replay.add(SimpleNamespace(
            episode_id=str(index), reward=2.0 if index < 2 else 0.5
        ))

    rows = replay.sample(8, positive_fraction=0.5)

    assert sum(float(row.reward) >= 1.0 for row in rows) == 2


def test_training_signal_metrics_report_event_and_reward_density() -> None:
    task = SimpleNamespace(functional_scenario="cutin")
    episode = SimpleNamespace(
        concrete_scenario=SimpleNamespace(candidate_id="main_conflict"),
        rollout=SimpleNamespace(
            outcome={"valid_target_collision": False, "valid_critical_near_miss": True},
            transitions=(
                {"reward_inner": -0.1, "info": {"event_just_captured": False}},
                {"reward_inner": 3.0, "info": {"event_just_captured": True}},
            ),
        ),
    )

    report = _training_signal_metrics([(task, episode)])

    assert report["overall"]["valid_event_episodes"] == 1
    assert report["overall"]["positive_reward_transitions"] == 1
    assert report["overall"]["positive_reward_transition_fraction"] == 0.5
    assert report["family:cutin"]["event_capture_transitions"] == 1


def test_training_curve_loader_recovers_prior_stage_after_resume(tmp_path: Path) -> None:
    metrics = {"episode_return_curve": [{"inner_return": 1.0}]}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "stages": [{"stage": "context_meta", "metrics": metrics}],
    }), encoding="utf-8")
    (tmp_path / "interaction_prior.json").write_text(json.dumps({
        "metrics": metrics,
    }), encoding="utf-8")

    stages = _recorded_stages(str(tmp_path / "manifest.json"))

    assert set(stages) == {"interaction_prior", "context_meta"}
