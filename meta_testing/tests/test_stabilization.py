from __future__ import annotations

import numpy as np
import torch

from meta_testing.audits import gate_failure_landscape
from meta_testing.failure.inner_reward import InnerRiskReward
from meta_testing.failure.signature import FailureSignatureBuilder
from meta_testing.training.replay import OuterRolloutBuffer, OuterRolloutStep
from meta_testing.training.updates import update_outer_ppo
from meta_testing.policy.scene_policy import HybridScenePolicy
from meta_testing.training.online_meta_test import OnlineMetaTest
from meta_testing.training.runner import Rollout
from meta_testing.model import HierarchicalMetaTester
from meta_testing.map.schema import MapPolyline, MapTokens
from meta_testing.scenario.applied import ExecutableEpisode
from meta_testing.scenario.catalog import mvr_parameter_spaces
from meta_testing.scenario.task_spec import MetaTestTaskSpec
from meta_testing.failure.signature import FailureSignature


def test_safe_valid_episode_is_not_a_failure() -> None:
    signature = FailureSignatureBuilder().from_outcome({"min_ttc": 15.0, "min_distance": 100.0}, "merge", None)
    assert signature.is_valid_episode and not signature.is_failure


def test_inner_reward_uses_risk_not_environment_reward() -> None:
    reward = InnerRiskReward()
    safe = np.zeros(12, dtype=np.float32)
    safe[8], safe[10] = 1.0, 1.0
    dangerous = safe.copy()
    dangerous[8], dangerous[10] = 0.05, 0.02
    assert reward(dangerous, {}, "gap_close", 10, 100) > reward(safe, {}, "gap_close", 10, 100)
    assert reward(dangerous, {"wrong_route": True}, "gap_close", 10, 100) < reward(dangerous, {}, "gap_close", 10, 100)


def test_outer_rollout_computes_normalized_gae() -> None:
    buffer = OuterRolloutBuffer()
    for reward, done in ((1.0, False), (2.0, True)):
        buffer.add(OuterRolloutStep(torch.zeros(1), torch.zeros(1), torch.empty(0), torch.zeros((), dtype=torch.long), torch.zeros(4), torch.zeros((), dtype=torch.long), torch.zeros(()), torch.zeros(()), reward, done))
    buffer.finish(gamma=1.0, gae_lambda=1.0)
    assert buffer.returns is not None and torch.allclose(buffer.returns, torch.tensor([3.0, 2.0]))
    assert buffer.advantages is not None and abs(float(buffer.advantages.mean())) < 1e-6


def test_outer_ppo_updates_only_finished_on_policy_rollout() -> None:
    policy = HybridScenePolicy(3, 2, 1, 2)
    buffer = OuterRolloutBuffer()
    inputs = torch.zeros(1, 3)
    action = policy.sample(inputs)
    buffer.add(OuterRolloutStep(torch.zeros(1), torch.zeros(1), torch.zeros(1), action.candidate_index.squeeze(0).detach(), action.continuous.squeeze(0).detach(), action.option_index.squeeze(0).detach(), action.log_prob.squeeze(0).detach(), action.value.squeeze(0).detach(), 1.0, True))
    buffer.finish()
    assert np.isfinite(update_outer_ppo(policy, buffer, torch.optim.Adam(policy.parameters(), lr=1e-3), epochs=1, batch_size=1))


def test_gate_a_requires_non_degenerate_cross_profile_difference() -> None:
    assert gate_failure_landscape(0.1, 0.2, 0.9, 0.2)["pass"]
    assert not gate_failure_landscape(0.1, 0.2, 1.0, 0.0)["pass"]


def test_online_loop_updates_posterior_and_respects_budget() -> None:
    polyline = MapPolyline("lane", "lane", np.asarray(((0.0, 0.0), (10.0, 0.0))), np.zeros(2), np.zeros(2), 3.5, 10.0, {})
    tokens = MapTokens("a" * 64, (polyline,), {})

    class Env:
        def close(self) -> None:
            pass

    class Executor:
        def reset(self, task, action):
            return ExecutableEpisode(Env(), np.zeros(3), {}, None, None, None, None, None, tokens, None)

    class Runner:
        def rollout(self, episode, config, option, inner_action, analyze):
            signature = FailureSignature("valid_critical_near_miss", "merge", None, (1, 1, 1), True, True)
            return Rollout(config, option, [], {"min_ttc": 2.0, "min_distance": 4.0, "max_closing_speed": 5.0}, signature, torch.zeros(2, 12))

    task = MetaTestTaskSpec("task", "meta_train", "idm_cautious", "merge", "map", "a" * 64, "template", "merge_v1", 1)
    result = OnlineMetaTest(HierarchicalMetaTester({"merge_v1": mvr_parameter_spaces()["merge_v1"]}, state_dim=3, map_dim=16), Executor(), Runner(), lambda _: None).run(task, 2)
    assert len(result.episodes) == len(result.inner_transitions) + 2
    assert len(result.outer_rollout.rows) == 2 and result.outer_rollout.rows[-1].done
    assert not torch.allclose(result.episodes[0].latent_before, result.episodes[0].latent_after)
