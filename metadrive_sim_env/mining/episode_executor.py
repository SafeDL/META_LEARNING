"""One fully auditable Mining simulator call."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

from ..evaluation.fewshot_inner import valid_critical_score
from ..physical_limits import CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2
from ..provenance import content_hash
from ..scenario.catalog import CUTIN_POST_MANEUVER_FOLLOW_THROUGH_M
from ..scenario.concrete import ConcreteScenario
from ..scenario.executor import ScenarioExecutor
from ..scenario.task_spec import ScenarioMiningTaskSpec
from ..training.runner import HierarchicalRunner
from .behavior import CutInBehavior
from .response import VulnerabilityResponseConfig, compute_vulnerability_response
from .types import CutInDesign, MiningObservation

BEHAVIOR_CONTRACT = "mining_cutin_step_behavior_constant_speed_hold"


class EpisodeExecutor:
    """Execute one design without reimplementing scenario or failure semantics."""
    def __init__(
        self,
        executor: ScenarioExecutor,
        runner: HierarchicalRunner,
        environment_horizon: int | None = None,
        response_config: VulnerabilityResponseConfig | None = None,
    ) -> None:
        self.executor = executor
        self.runner = runner
        self.environment_horizon = environment_horizon
        self.response_config = response_config or VulnerabilityResponseConfig(
            ttc_scale_s=5.0,
            distance_scale_m=10.0,
            noncritical_cap=0.74,
            near_miss_floor=0.75,
            proxy_event_threshold=0.75,
        )

    @staticmethod
    def _status(outcome: Mapping[str, Any], transitions: list[Mapping[str,
                                                                      Any]]) -> tuple[str, bool]:
        score = valid_critical_score(outcome)
        if score > 0.0:
            return "valid_event", True
        if not bool(outcome.get("is_valid_episode", False)):
            return "invalid", False
        onset_seen = outcome.get("cutin_actual_onset") is not None
        completed = any(
            bool(row["info"].get("semantic_maneuver_completed", False)) for row in transitions)
        route_completed = bool(outcome.get("test_process_completed", False))
        if onset_seen and completed and route_completed:
            return "completed_noncritical", True
        return "censored", False

    def run(
        self,
        task: ScenarioMiningTaskSpec,
        design: CutInDesign,
        episode_seed: int,
        *,
        logical_domain_id: str | None = None,
    ) -> MiningObservation:
        if task.functional_scenario != "cutin":
            raise ValueError("Mining only executes Cut-in tasks")
        if task.geometry_id != "cutin-g01":
            raise ValueError("Mining requires retained geometry cutin-g01")
        overrides = ({
            "horizon": int(self.environment_horizon)
        } if self.environment_horizon is not None else None)
        episode = self.executor.reset(
            task,
            design.scenario_action(),
            episode_seed=episode_seed,
            environment_overrides=overrides,
        )
        behavior = CutInBehavior()
        behavior_hash = content_hash({
            "contract":
            BEHAVIOR_CONTRACT,
            "speed_control_gain":
            behavior.speed_control_gain,
            "maximum_longitudinal_action":
            behavior.maximum_longitudinal_action,
            "lateral_acceleration_limit_mps2": (CUTIN_LATERAL_ACCELERATION_LIMIT_MPS2),
            "completion_condition":
            "sut_cutin_follow_through",
            "post_maneuver_follow_through_m": (CUTIN_POST_MANEUVER_FOLLOW_THROUGH_M),
        })
        started = perf_counter()
        try:
            rollout = self.runner.rollout(episode, "cutin", step_action=behavior)
            concrete = ConcreteScenario.from_applied(
                task,
                episode.applied_scenario,
                behavior_hash,
                episode_seed=episode_seed,
            )
        finally:
            episode.env.close()
        score = valid_critical_score(rollout.outcome)
        status, posterior_eligible = self._status(rollout.outcome, rollout.transitions)
        response = compute_vulnerability_response(rollout.outcome, status, self.response_config)
        audit = {
            "cutin_actual_onset":
            rollout.outcome.get("cutin_actual_onset"),
            "prescribed_adversary_speed_mps":
            behavior.prescribed_speed_mps,
            "termination_reason":
            rollout.outcome.get("termination_reason"),
            "maneuver_completed":
            any(
                bool(row["info"].get("semantic_maneuver_completed", False))
                for row in rollout.transitions),
            "sut_route_completed":
            bool(rollout.outcome.get("test_process_completed", False)),
            "test_process_completed":
            bool(rollout.outcome.get("test_process_completed", False)),
            "test_completion_condition":
            rollout.outcome.get("test_completion_condition"),
            "control_telemetry":
            rollout.outcome.get("control_telemetry", {}),
            "failure_signature":
            rollout.signature.signature_id,
            "severity_vector":
            rollout.signature.severity_vector,
        }
        return MiningObservation(
            design=design,
            task_id=task.task_id,
            sut_ref=task.sut_ref,
            geometry_id=task.geometry_id,
            logical_domain_id=logical_domain_id or task.logical_domain_id,
            episode_seed=int(episode_seed),
            score=score,
            is_valid_episode=bool(rollout.outcome.get("is_valid_episode", False)),
            status=status,
            posterior_eligible=posterior_eligible,
            vulnerability_response=response,
            outcome=audit,
            concrete_scenario=concrete.to_dict(),
            behavior_contract_hash=behavior_hash,
            elapsed_seconds=perf_counter() - started,
        )
