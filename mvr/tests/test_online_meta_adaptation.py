from __future__ import annotations

import numpy as np
import torch

from mvr.failure.criteria import DEFAULT_FAILURE_CRITERIA
from mvr.model import TransferableScenarioMiner
from mvr.state import PhysicalStateExtractor
from mvr.scenario.taskbook import load_taskbook
from mvr.training.online_meta_test import _inner_learning_blocks
from mvr.training.trainers import build_online


def test_inner_learning_blocks_preserve_macro_boundaries_and_terminal_tail() -> None:
    transitions = [
        {
            "state": np.asarray((index,), dtype=np.float32),
            "raw_policy_action": np.asarray((decision,) * 4, dtype=np.float32),
            "reward_inner": float(index + 1),
            "next_state": np.asarray((index + 1,), dtype=np.float32),
            "done": index == 6,
            "info": {"inner_policy_decision": index in {0, 3, 5}},
        }
        for index, decision in enumerate((1, 1, 1, 2, 2, 3, 3))
    ]

    blocks = _inner_learning_blocks(transitions)

    assert [len(block) for block in blocks] == [3, 2, 2]
    assert [sum(row["reward_inner"] for row in block) for block in blocks] == [6.0, 9.0, 13.0]
    np.testing.assert_allclose(blocks[1][0]["raw_policy_action"], (2.0,) * 4)
    np.testing.assert_allclose(blocks[-1][-1]["next_state"], (7.0,))
    assert blocks[-1][-1]["done"] is True


def test_online_adaptation_obeys_k_shot_budget_and_freezes_after_support() -> None:
    task = load_taskbook("mvr/configs/taskbook.json")[0]
    model = TransferableScenarioMiner(state_dim=PhysicalStateExtractor.dimension, map_dim=16)
    zero = build_online(model, task, 1, DEFAULT_FAILURE_CRITERIA).run(
        task, 2, posterior_support_limit=0
    )
    assert all(torch.allclose(row.latent_before, row.latent_after) for row in zero.episodes)
    first_episode = zero.episodes[0]
    blocks = _inner_learning_blocks(first_episode.rollout.transitions)
    replay_rows = [
        row
        for row in zero.inner_transitions
        if row.episode_id == first_episode.episode_id
    ]
    assert len(replay_rows) == len(blocks) < len(first_episode.rollout.transitions)
    for replay_row, block in zip(replay_rows, blocks):
        np.testing.assert_allclose(
            replay_row.action, block[0]["raw_policy_action"]
        )
        assert replay_row.duration_steps == len(block)
        assert replay_row.reward == sum(
            0.99 ** index * row["reward_inner"]
            for index, row in enumerate(block)
        )
        np.testing.assert_allclose(replay_row.next_state, block[-1]["next_state"])
        assert replay_row.done is block[-1]["done"]
    adapted = build_online(model, task, 1, DEFAULT_FAILURE_CRITERIA).run(
        task, 3, posterior_support_limit=2
    )
    assert len(adapted.episodes) == 3
    assert not torch.allclose(adapted.episodes[0].latent_before, adapted.episodes[0].latent_after)
    assert torch.allclose(adapted.episodes[2].latent_before, adapted.episodes[2].latent_after)
