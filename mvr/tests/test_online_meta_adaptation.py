from __future__ import annotations

import torch

from mvr.failure.criteria import DEFAULT_FAILURE_CRITERIA
from mvr.model import TransferableScenarioMiner
from mvr.state import PhysicalStateExtractor
from mvr.scenario.taskbook import load_taskbook
from mvr.training.trainers import build_online


def test_online_adaptation_obeys_k_shot_budget_and_freezes_after_support() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    model = TransferableScenarioMiner(state_dim=PhysicalStateExtractor.dimension, map_dim=16)
    zero = build_online(model, task, 1, DEFAULT_FAILURE_CRITERIA).run(
        task, 2, posterior_support_limit=0
    )
    assert all(torch.allclose(row.latent_before, row.latent_after) for row in zero.episodes)
    adapted = build_online(model, task, 1, DEFAULT_FAILURE_CRITERIA).run(
        task, 3, posterior_support_limit=2
    )
    assert len(adapted.episodes) == 3
    assert not torch.allclose(adapted.episodes[0].latent_before, adapted.episodes[0].latent_after)
    assert torch.allclose(adapted.episodes[2].latent_before, adapted.episodes[2].latent_after)
