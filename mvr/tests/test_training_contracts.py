from __future__ import annotations

import torch

from mvr.model import TransferableScenarioMiner
from mvr.training.replay import OuterRolloutBuffer, OuterRolloutStep
from mvr.training.stages import TrainingStage, trainable_components
from mvr.training.updates import update_outer_ppo
from mvr.training.workflow import StagedWorkflow


def test_stage_ownership_and_universal_on_policy_ppo() -> None:
    model = TransferableScenarioMiner(state_dim=10, map_dim=8)
    workflow = StagedWorkflow(model.training_components())
    workflow.activate(TrainingStage.OUTER)
    assert trainable_components(TrainingStage.OUTER) == {"universal_scene_policy"}
    assert all(parameter.requires_grad for parameter in model.universal_scene_policy.parameters())
    assert not any(parameter.requires_grad for parameter in model.inner_sac.parameters())
    policy = model.universal_scene_policy
    action = policy.sample(torch.zeros(8), torch.zeros(2, 8), torch.ones(2, dtype=torch.bool), torch.zeros(1, 16))
    buffer = OuterRolloutBuffer()
    buffer.add(OuterRolloutStep(
        torch.zeros(8), torch.zeros(2, 8), torch.ones(2, dtype=torch.bool), torch.zeros(16),
        action.expert_index.squeeze(0).detach(), action.candidate_index.squeeze(0).detach(),
        action.continuous.squeeze(0).detach(), action.option_index.squeeze(0).detach(),
        action.log_prob.squeeze(0).detach(), action.value.squeeze(0).detach(), 1.0, True,
    ))
    buffer.finish()
    loss = update_outer_ppo(policy, buffer, torch.optim.Adam(policy.parameters(), lr=1e-3), epochs=1, batch_size=1)
    assert torch.isfinite(torch.tensor(loss))
