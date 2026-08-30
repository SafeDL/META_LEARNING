from types import SimpleNamespace

import pytest
import torch

from mvr.evaluation.fewshot_inner import (
    AdaptationQualityProtocol,
    BudgetEfficiencyProtocol,
    paired_bootstrap,
    paired_policy_deltas,
    summarize_outcomes,
    valid_critical_score,
)
from mvr.model import TransferableScenarioMiner
from mvr.training.calibration_casebook import (
    CalibrationCase,
    CalibrationCasebook,
    is_calibration_headroom,
)
from mvr.scenario.parameter_space import NormalizedScenarioAction
from mvr.training.replay import ContextReplay, SupportGroup
from mvr.scripts.evaluate_inner_fewshot import _query_records


def test_calibration_casebook_keeps_validation_provenance_without_test_safety_claim(tmp_path) -> None:
    assert is_calibration_headroom({"is_valid_episode": True, "is_failure": False}, 1)
    assert not is_calibration_headroom({"is_valid_episode": True, "is_failure": True}, 1)
    casebook = CalibrationCasebook({"test-task": (CalibrationCase(
        NormalizedScenarioAction(0, (0.0,) * 5), "validation-task", "idm_fast_small_gap", 3,
    ),)}, {"test_sut_base_safe_claim": False})
    path = tmp_path / "casebook.json"
    casebook.save(path)
    loaded = CalibrationCasebook.load(path)
    assert loaded.case_for("test-task", 0).provenance()["calibration_task_id"] == "validation-task"
    assert loaded.metadata["test_sut_base_safe_claim"] is False


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
    concrete = torch.randn(1, 18)

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
    left = torch.cat((torch.tensor([[1.0] + [0.0] * 7]), torch.zeros(1, 10)), dim=-1)
    right = torch.cat((torch.tensor([[0.0, 1.0] + [0.0] * 6]), torch.zeros(1, 10)), dim=-1)
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
        model.encode_task_structure(scene, narrow, (True,) * 5),
        model.encode_task_structure(scene, wide, (True,) * 5),
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


def test_paired_bootstrap_requires_complete_query_pairs() -> None:
    records = []
    for policy, score in (("adapted_h_z", 1.0), ("shared_prior", 0.5)):
        records.append({
            "task_id": "task", "logical_domain_id": "test", "support_shots": 2,
            "seed": 11, "query_case_id": 0, "policy": policy, "score": score, "invalid": False,
        })
    pairs = paired_policy_deltas(records, "adapted_h_z", "shared_prior")
    assert pairs == [{"score_delta": 0.5, "invalid_delta": 0.0}]
    report = paired_bootstrap(pairs, samples=50, seed=7)
    assert report["pairs"] == 1.0
    assert report["mean_delta"] == pytest.approx(0.5)


def test_query_record_contains_pairing_seed_latent_and_calibration_provenance() -> None:
    casebook = CalibrationCasebook({"task": (CalibrationCase(
        NormalizedScenarioAction(0, (0.0,) * 5), "validation-task", "idm_fast_small_gap", 7,
    ),)}, {"test_sut_base_safe_claim": False})
    task = SimpleNamespace(
        task_id="task", functional_scenario="merge", geometry_id="g01", logical_domain_id="tight_gap",
    )
    episode = SimpleNamespace(
        outcome={"is_valid_episode": True, "valid_target_collision": True},
        latent_before=torch.tensor([[0.1, 0.2]]),
        concrete_scenario=SimpleNamespace(to_dict=lambda: {"candidate_id": "main_conflict"}),
    )
    record = _query_records(
        [episode], task=task, casebook=casebook, support_shots=2,
        seed=22, policy="adapted_h_z", max_support=0,
    )[0]
    assert record["seed"] == 22
    assert record["query_case_id"] == 0
    assert record["z"] == pytest.approx([0.1, 0.2])
    assert record["concrete_provenance"]["casebook"]["calibration_case_id"] == 7
