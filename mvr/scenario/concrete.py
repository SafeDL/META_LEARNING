"""Reproducible manifest for one concrete adversarial scenario."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .applied import AppliedScenario
from .option import AdversarialOption
from .parameter_space import NormalizedScenarioAction, ParameterSpace
from .task_spec import ScenarioMiningTaskSpec


@dataclass(frozen=True)
class ConcreteScenario:
    geometry_id: str
    geometry_hash: str
    geometry_seed: int
    candidate_id: str
    conflict_zone_id: str
    option: str
    initial_state: Mapping[str, float]
    inner_policy_hash: str
    normalized_continuous: tuple[float, ...]
    latent: tuple[float, ...] = ()
    episode_seed: int | None = None

    @classmethod
    def from_applied(
        cls,
        task: ScenarioMiningTaskSpec,
        applied: AppliedScenario,
        inner_policy_hash: str,
        *,
        latent: Any = (),
        episode_seed: int | None = None,
    ) -> "ConcreteScenario":
        latent_values = tuple(float(value) for value in latent)
        return cls(
            task.geometry_id, task.geometry_hash, task.geometry_seed, applied.selected_candidate,
            applied.conflict_zone_id, applied.selected_option,
            {
                "adversary_distance_to_conflict_m": applied.adversary_distance_to_conflict_m,
                "sut_distance_to_conflict_m": applied.sut_distance_to_conflict_m,
                "adversary_initial_speed_mps": applied.adversary_speed_mps,
                "sut_initial_speed_mps": applied.sut_speed_mps,
                "maneuver_onset_progress": applied.maneuver_onset_progress,
            },
            inner_policy_hash, applied.normalized_continuous,
            latent_values,
            episode_seed,
        )

    def replay_action(self, space: ParameterSpace) -> NormalizedScenarioAction:
        if self.candidate_id not in space.candidates:
            raise ValueError("concrete scenario candidate is absent from the parameter space")
        if len(self.normalized_continuous) != space.continuous_dim:
            raise ValueError("concrete scenario lacks its normalized Outer action")
        return NormalizedScenarioAction(
            space.candidates.index(self.candidate_id), self.normalized_continuous, AdversarialOption(self.option)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
