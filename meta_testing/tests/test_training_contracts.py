from __future__ import annotations

import pytest

from meta_testing.training.replay import InnerReplay, InnerTransition
from meta_testing.training.stages import TrainingStage, trainable_components


def test_inner_replay_excludes_context_episodes() -> None:
    replay = InnerReplay()
    replay.add(InnerTransition("support", 1, 1, 1.0, 2, False))
    replay.add(InnerTransition("query", 1, 1, 1.0, 2, False))
    assert replay.sample(1, excluded_episode_ids={"support"})[0].episode_id == "query"
    with pytest.raises(ValueError):
        replay.sample(1, excluded_episode_ids={"support", "query"})


def test_stages_do_not_start_outer_and_inner_together() -> None:
    assert trainable_components(TrainingStage.INNER_PRETRAIN) == {"inner"}
    assert "outer" not in trainable_components(TrainingStage.POSTERIOR)
