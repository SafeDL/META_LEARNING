from __future__ import annotations

from types import SimpleNamespace

import torch

from mvr.model import TransferableScenarioMiner
from mvr.state import PhysicalStateExtractor
from mvr.policy.adversarial_sac import AdversarialSAC
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
