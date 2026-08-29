from __future__ import annotations

import torch

from mvr.policy.universal_scene_policy import UniversalScenePolicy


def test_universal_policy_handles_variable_candidate_counts_and_trains_experts() -> None:
    policy = UniversalScenePolicy(8, 4, 4, 3)
    for count in (2, 3, 4):
        action = policy.sample(torch.randn(8), torch.randn(count, 8), torch.ones(count, dtype=torch.bool), torch.randn(1, 4))
        assert 0 <= int(action.candidate_index) < count
        assert 0 <= int(action.expert_index) < 3
    logits = policy.router(torch.randn(5, 8), torch.randn(5, 4))
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(5))
    loss = policy.router.load_balance_loss(logits)
    for expert in policy.experts:
        loss = loss + sum(parameter.square().mean() for parameter in expert.parameters())
    loss.backward()
    assert all(any(parameter.grad is not None for parameter in expert.parameters()) for expert in policy.experts)
