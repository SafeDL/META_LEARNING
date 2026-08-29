from types import SimpleNamespace

import pytest
import torch

from mvr.evaluation.fewshot_inner import (
    AdaptationQualityProtocol,
    BudgetEfficiencyProtocol,
    summarize_outcomes,
    valid_critical_score,
)
from mvr.model import TransferableScenarioMiner
from mvr.training.headroom_casebook import is_base_safe_headroom
from mvr.training.replay import ContextReplay, SupportGroup


def test_headroom_requires_a_lawful_challenge_without_a_base_failure() -> None:
    assert is_base_safe_headroom({"is_valid_episode": True, "is_failure": False}, 1)
    assert not is_base_safe_headroom({"is_valid_episode": True, "is_failure": True}, 1)
    assert not is_base_safe_headroom({"is_valid_episode": False, "is_failure": False}, 1)
    assert not is_base_safe_headroom({"is_valid_episode": True, "is_failure": False}, 0)


def test_support_groups_are_episode_level_and_disjoint() -> None:
    support = SimpleNamespace(episode_id="support")
    query = SimpleNamespace(episode_id="query")
    replay = ContextReplay()
    replay.add(SupportGroup("group", "task", (support,), {"query": query}))
    assert replay.get("group").task_id == "task"
    with pytest.raises(ValueError, match="disjoint"):
        SupportGroup("bad", "task", (support,), {"support": support}).validate()


def test_actor_stops_context_gradient_but_critic_keeps_it() -> None:
    model = TransferableScenarioMiner(state_dim=11, map_dim=8, latent_dim=4)
    support = torch.randn(1, 2, 128)
    mask = torch.ones(1, 2, dtype=torch.bool)
    latent, _ = model.infer_posterior(support, mask)
    state = torch.randn(1, 11)
    scene = torch.randn(1, 8)
    concrete = torch.randn(1, 13)

    actor_features = model.inner_features(state, scene, latent.detach(), concrete)
    actor, _ = model.inner_sac.actor_alpha_losses(actor_features)
    actor.backward()
    assert all(parameter.grad is None for parameter in model.context_encoder.parameters())

    model.zero_grad(set_to_none=True)
    critic_features = model.inner_features(state, scene, latent, concrete)
    critic = model.inner_sac.critic_loss(
        critic_features, torch.zeros(1, 2), torch.zeros(1), critic_features.detach(),
        torch.ones(1, dtype=torch.bool),
    )
    critic.backward()
    assert any(parameter.grad is not None for parameter in model.context_encoder.parameters())


def test_concrete_candidate_changes_inner_features() -> None:
    model = TransferableScenarioMiner(state_dim=11, map_dim=8, latent_dim=4)
    state, scene, latent = torch.zeros(1, 11), torch.zeros(1, 8), torch.zeros(1, 4)
    left = torch.cat((torch.tensor([[1.0] + [0.0] * 7]), torch.zeros(1, 5)), dim=-1)
    right = torch.cat((torch.tensor([[0.0, 1.0] + [0.0] * 6]), torch.zeros(1, 5)), dim=-1)
    assert not torch.allclose(
        model.inner_features(state, scene, latent, left),
        model.inner_features(state, scene, latent, right),
    )


def test_logical_domain_bounds_are_part_of_observable_task_structure() -> None:
    model = TransferableScenarioMiner(state_dim=11, map_dim=8, latent_dim=4)
    scene = torch.zeros(8)
    narrow = {name: (-0.2, 0.2) for name in (
        "adversary_distance_to_conflict_m", "sut_distance_to_conflict_m",
        "adversary_initial_speed_mps", "sut_initial_speed_mps", "maneuver_onset_progress",
    )}
    wide = {name: (-0.8, 0.8) for name in narrow}
    assert not torch.allclose(
        model.encode_task_structure(scene, narrow),
        model.encode_task_structure(scene, wide),
    )


def test_adaptation_quality_and_budget_efficiency_are_separate_protocols() -> None:
    quality = AdaptationQualityProtocol()
    budget = BudgetEfficiencyProtocol()
    assert quality.total_episodes(4) == 12
    assert budget.query_cases(4) == 16
    assert valid_critical_score({"is_valid_episode": True, "valid_target_collision": True}) == 1.0
    assert valid_critical_score({"is_valid_episode": True, "valid_critical_near_miss": True}) == 0.5
    assert summarize_outcomes([
        {"is_valid_episode": True, "valid_target_collision": True},
        {"is_valid_episode": False, "valid_target_collision": True},
    ]) == {
        "episodes": 2.0,
        "valid_critical_score_mean": 0.5,
        "cumulative_valid_critical_score": 1.0,
        "invalid_rate": 0.5,
    }
