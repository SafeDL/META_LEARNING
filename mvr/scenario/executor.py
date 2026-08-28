"""One outer action maps to one verified physical simulator episode."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ..map.metadrive_tokenizer import tokenize_road_network
from ..sut.registry import SUTRegistry, default_registry
from .applied import AppliedScenario, ExecutableEpisode
from .layout import ScenarioLayout
from .interaction import InteractionCandidate
from .parameter_space import NormalizedScenarioAction, ParameterSpace
from .roles import spawn_sut
from .route_geometry import RoutePolyline
from .task_spec import ScenarioMiningTaskSpec


class ScenarioAdapter(Protocol):
    family: str
    def build_env(
        self,
        task: ScenarioMiningTaskSpec,
        config: Mapping[str, float | str],
        layout: ScenarioLayout | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> Any: ...
    def reset(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], seed: int) -> tuple[Any, Mapping[str, Any]]: ...
    def resolve_layout(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str], candidates: tuple[str, ...]) -> ScenarioLayout: ...
    def validate_runtime(self, env: Any, task: ScenarioMiningTaskSpec, config: Mapping[str, float | str]) -> None: ...
    def spawn_from_conflict_distance(self, route: RoutePolyline, conflict_xy: tuple[float, float], distance_to_conflict_m: float) -> float: ...


@dataclass(frozen=True)
class _CachedLayout:
    layout: ScenarioLayout
    adversary_route: RoutePolyline
    sut_route: RoutePolyline


@dataclass(frozen=True)
class _StaticScene:
    map_tokens: Any
    candidates: tuple[InteractionCandidate, ...]
    layouts: Mapping[str, _CachedLayout]


@dataclass
class ScenarioExecutor:
    adapters: Mapping[str, ScenarioAdapter]
    spaces: Mapping[str, ParameterSpace]
    sut_registry: SUTRegistry = field(default_factory=default_registry)
    _static_scenes: dict[tuple[str, str], _StaticScene] = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def _adversary(env: Any) -> Any:
        agents = env.engine.agent_manager.active_agents
        if "default_agent" not in agents:
            raise RuntimeError("MetaDrive episode has no default adversary agent")
        return agents["default_agent"]

    @staticmethod
    def _speed_mps(vehicle: Any) -> float:
        return float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6

    @staticmethod
    def _assert_vehicle_applied(vehicle: Any, lane_index: tuple[Any, Any, int], spawn_m: float, speed_mps: float, destination: Any) -> None:
        config = vehicle.config
        if tuple(config["spawn_lane_index"]) != tuple(lane_index):
            raise RuntimeError("runtime vehicle spawn lane differs from resolved scenario layout")
        if not np.isclose(float(config["spawn_longitude"]), float(spawn_m), atol=1e-5):
            raise RuntimeError("runtime vehicle spawn longitudinal position differs from outer output")
        if config.get("destination") != destination:
            raise RuntimeError("runtime vehicle destination differs from resolved candidate route")
        if not np.isclose(ScenarioExecutor._speed_mps(vehicle), float(speed_mps), atol=0.25):
            raise RuntimeError("runtime vehicle initial speed differs from outer output")

    @staticmethod
    def _assert_vehicle_route(
        vehicle: Any,
        expected: tuple[Any, ...],
        *,
        lane_stable: bool = False,
    ) -> None:
        actual = tuple(vehicle.navigation.checkpoints)
        if actual != expected:
            raise RuntimeError("native navigation checkpoints differ from scenario contract")
        if lane_stable and int(vehicle.lane.index[2]) != int(vehicle.config["spawn_lane_index"][2]):
            raise RuntimeError("lane-stable SUT route changed lanes during native initialization")

    @staticmethod
    def sut_lane_status(episode: ExecutableEpisode, *, require_routing_target: bool) -> dict[str, Any]:
        """Validate the SUT's physical and controller lane targets at this step."""
        contract = episode.layout.native_navigation
        if contract is None:
            raise RuntimeError("scenario layout has no native navigation contract")
        navigation_lane = episode.sut.navigation.current_lane
        actual_lane = episode.sut.lane
        expected_number = contract.expected_sut_lane_number(navigation_lane.index[:2])
        policy = episode.env.engine.get_policy(episode.sut.id)
        routing_lane = getattr(policy, "routing_target_lane", None)
        status = {
            "sut_current_lane": tuple(actual_lane.index),
            "sut_navigation_lane": tuple(navigation_lane.index),
            "sut_current_ref_lanes": tuple(tuple(lane.index) for lane in episode.sut.navigation.current_ref_lanes),
            "sut_routing_target_lane": None if routing_lane is None else tuple(routing_lane.index),
            "sut_expected_lane_number": int(expected_number),
        }
        if int(actual_lane.index[2]) != expected_number or int(navigation_lane.index[2]) != expected_number:
            raise RuntimeError(f"SUT lane number violates lane-stable route: {status!r}")
        if require_routing_target:
            # Policies act before navigation localization is advanced by the
            # same physics tick.  At a road boundary the concrete road index
            # can therefore lag by one tick, but the lane number must never
            # deviate from the declared lane-stable sequence.
            if routing_lane is None or int(routing_lane.index[2]) != expected_number:
                raise RuntimeError(f"SUT routing target violates lane-stable route: {status!r}")
        return status

    @staticmethod
    def _static_key(task: ScenarioMiningTaskSpec) -> tuple[str, str]:
        return task.adapter_id, task.geometry_hash

    @staticmethod
    def _resolve_spawn_config(
        adapter: ScenarioAdapter,
        cached: _CachedLayout,
        candidate: InteractionCandidate,
        config: Mapping[str, float | str],
        action: NormalizedScenarioAction,
    ) -> dict[str, float | str]:
        resolved = dict(config)
        for index, (name, route) in enumerate((("adversary", cached.adversary_route), ("sut", cached.sut_route))):
            distance_key = f"{name}_distance_to_conflict_m"
            lower = float(getattr(candidate, f"{name}_distance_min_m"))
            upper = float(getattr(candidate, f"{name}_distance_available_m"))
            if not 0.0 <= lower <= upper:
                raise RuntimeError(f"{candidate.candidate_id} has no executable {name} spawn interval")
            applied_distance = float(lower + 0.5 * (float(action.continuous[index]) + 1.0) * (upper - lower))
            resolved[distance_key] = applied_distance
            resolved[f"{name}_spawn_m"] = adapter.spawn_from_conflict_distance(
                route, cached.layout.conflict_xy, applied_distance
            )
        return resolved

    def _static_scene(self, task: ScenarioMiningTaskSpec) -> _StaticScene:
        key = self._static_key(task)
        if key not in self._static_scenes:
            self.enumerate_interactions(task)
        return self._static_scenes[key]

    def enumerate_interactions(
        self, task: ScenarioMiningTaskSpec
    ) -> tuple[Any, tuple[InteractionCandidate, ...]]:
        """Inspect the runtime map before policy selection without exposing labels."""
        task.validate()
        key = self._static_key(task)
        cached = self._static_scenes.get(key)
        if cached is not None:
            return cached.map_tokens, cached.candidates
        try:
            adapter, space = self.adapters[task.adapter_id], self.spaces[task.functional_scenario]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        base = space.decode(NormalizedScenarioAction(0, (0.0,) * space.continuous_dim, space.options[0]))
        env = adapter.build_env(task, base)
        try:
            adapter.reset(env, task, base, task.geometry_seed)
            tokens = tokenize_road_network(env.current_map.road_network)
            if tokens.map_hash != task.geometry_hash:
                raise RuntimeError(f"runtime map hash mismatch for {task.task_id}")
            candidates = []
            layouts: dict[str, _CachedLayout] = {}
            for index in range(len(space.candidates)):
                config = space.decode(
                    NormalizedScenarioAction(index, (0.0,) * space.continuous_dim, space.options[0])
                )
                layout = adapter.resolve_layout(env, task, config, space.candidates)
                adversary_route = RoutePolyline.from_env(
                    env, {"route_id": "adversary", "lane_sequence": layout.adversary_route}
                )
                sut_route = RoutePolyline.from_env(
                    env, {"route_id": "sut", "lane_sequence": layout.sut_route}
                )
                layouts[layout.candidate] = _CachedLayout(layout, adversary_route, sut_route)
                candidates.append(InteractionCandidate.from_routes(layout, adversary_route, sut_route))
            static = _StaticScene(tokens, tuple(candidates), layouts)
            self._static_scenes[key] = static
            return static.map_tokens, static.candidates
        finally:
            env.close()

    def reset(
        self,
        task: ScenarioMiningTaskSpec,
        action: NormalizedScenarioAction,
        *,
        episode_seed: int | None = None,
        environment_overrides: Mapping[str, Any] | None = None,
    ) -> ExecutableEpisode:
        task.validate()
        run_seed = task.geometry_seed if episode_seed is None else int(episode_seed)
        try:
            adapter, space = self.adapters[task.adapter_id], self.spaces[task.functional_scenario]
        except KeyError as error:
            raise ValueError(f"no executable contract for task {task.task_id}") from error
        config = space.decode(action)
        static = self._static_scene(task)
        candidate = str(config["route_or_conflict_candidate"])
        try:
            cached = static.layouts[candidate]
        except KeyError as error:
            raise RuntimeError(f"static layout is missing candidate {candidate!r}") from error
        interaction = next(row for row in static.candidates if row.candidate_id == candidate)
        config = self._resolve_spawn_config(adapter, cached, interaction, config, action)
        if environment_overrides:
            env = adapter.build_env(task, config, cached.layout, environment_overrides)
        else:
            env = adapter.build_env(task, config, cached.layout)
        try:
            observation, _ = adapter.reset(env, task, config, task.geometry_seed)
            adapter.validate_runtime(env, task, config)
            runtime_tokens = tokenize_road_network(env.current_map.road_network)
            if runtime_tokens.map_hash != task.map_hash:
                raise RuntimeError(f"runtime map hash changed between layout and execution: expected {task.map_hash}, got {runtime_tokens.map_hash}")
            map_tokens = static.map_tokens
            adversary = self._adversary(env)
            sut_adapter, sut_profile = self.sut_registry.create(task.sut_ref)
            sut_adapter.reset(env, task, config, run_seed)
            layout = cached.layout
            sut = spawn_sut(
                env,
                lane_index=layout.sut_lane,
                longitudinal_m=float(config["sut_spawn_m"]),
                speed_mps=float(config["sut_initial_speed_mps"]),
                destination=layout.sut_destination,
                adapter=sut_adapter,
                profile=sut_profile,
                seed=run_seed,
                speed_limit_mps=layout.traffic_contract.speed_limit_mps,
                nominal_speed_mps=layout.traffic_contract.sut_nominal_speed_mps,
            )
            self._assert_vehicle_applied(adversary, layout.adversary_lane, float(config["adversary_spawn_m"]), float(config["adversary_initial_speed_mps"]), layout.adversary_destination)
            self._assert_vehicle_applied(sut, layout.sut_lane, float(config["sut_spawn_m"]), float(config["sut_initial_speed_mps"]), layout.sut_destination)
            navigation = layout.native_navigation
            if navigation is None:
                raise RuntimeError("scenario layout has no native navigation contract")
            self._assert_vehicle_route(
                adversary, navigation.adversary_checkpoints
            )
            self._assert_vehicle_route(
                sut, navigation.sut_checkpoints, lane_stable=navigation.sut_lane_stable
            )
            applied = AppliedScenario(
                str(adversary.id), str(sut.id), layout.adversary_lane, layout.sut_lane,
                float(config["adversary_spawn_m"]), float(config["sut_spawn_m"]),
                float(config["adversary_distance_to_conflict_m"]),
                float(config["sut_distance_to_conflict_m"]),
                float(config["adversary_initial_speed_mps"]), float(config["sut_initial_speed_mps"]),
                float(config["maneuver_onset_progress"]),
                layout.candidate, layout.conflict_zone_id, str(config["option"]),
                layout.adversary_route, layout.sut_route, tuple(float(value) for value in action.continuous),
            )
            setattr(env, "_mvr_episode", applied)
            episode = ExecutableEpisode(
                env, observation, adversary, sut, sut_adapter, sut_profile,
                applied, map_tokens, layout, cached.adversary_route, cached.sut_route, run_seed,
            )
            self.sut_lane_status(episode, require_routing_target=False)
            return episode
        except Exception:
            env.close()
            raise
