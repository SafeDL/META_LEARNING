"""Gymnasium facade for frozen, route-audited logical merge tasks."""
from __future__ import annotations

from typing import Any, Mapping
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .adapters import adapter_for
from .critical import (
    CRITICAL_METRIC_SCHEMA,
    LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA,
    collision_risk_barrier,
    conflict_entry_order_satisfied,
    critical_measurements,
    is_strict_near_miss_schema,
    strict_near_miss_potential,
)
from .benchmark_calibration import thresholds_for_task
from .metrics import EpisodeMetrics
from .moe import PhysicalTaskDescriptor, physical_task_descriptor
from .observation import build_observation, observation_dim, observation_fields
from .reward import compute_reward, validate_reward_contract
from .task_spec import LogicalScenarioTaskSpec


def compose_route_tracking_action(
    policy_action: np.ndarray,
    heading_error_to_lookahead: float,
    lateral_m: float,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Compose a fixed-route tracker with a bounded learned residual.

    The route is part of the frozen physical task, not a hidden reward label.
    Delegating lane keeping to a deterministic low-level controller lets the
    learned policy focus on interaction timing while retaining a small
    steering residual for robustness.
    """
    action = np.asarray(policy_action, dtype=np.float32)
    tracking = config.get("control", {}).get("route_tracking", {})
    if not bool(tracking.get("enabled", False)):
        return action.copy()
    steering = (
        float(tracking.get("heading_gain", 1.5)) * float(heading_error_to_lookahead)
        - float(tracking.get("lateral_gain", 0.12)) * float(lateral_m)
        + float(tracking.get("residual_scale", 0.25)) * float(action[0])
    )
    return np.asarray([np.clip(steering, -1.0, 1.0), action[1]], dtype=np.float32)


def _ttc(a: Any, b: Any, cap: float) -> tuple[float, float, float]:
    position = np.asarray(b.position, dtype=float) - np.asarray(a.position, dtype=float)
    velocity = np.asarray(b.velocity, dtype=float) - np.asarray(a.velocity, dtype=float)
    distance = float(np.linalg.norm(position))
    closing = float(np.dot(position, velocity) / max(distance, 1e-6))
    return (min(cap, distance / -closing) if closing < 0.0 else cap), distance, max(0.0, -closing)


def target_contact_matches_rule(priority_spec: Mapping[str, Any], adversary_speed_mps: float, sut_speed_mps: float,
                                first_conflict_entry_role: str | None = None) -> bool:
    """Check the frozen, interaction-only target-contact qualification rule.

    The rule is intentionally absent from ``logical_merge_obs``.  It can only
    be inferred from a support transition's observed motion and reward.
    """
    entry_order = str(priority_spec.get("target_contact_entry_order", "any"))
    if entry_order != "any":
        return first_conflict_entry_role == entry_order.removesuffix("_first")
    relation = str(priority_spec.get("target_contact_speed_relation", "any"))
    margin = float(priority_spec.get("target_contact_speed_margin_mps", 0.0))
    if relation == "any":
        return True
    if relation == "adversary_faster":
        return bool(adversary_speed_mps >= sut_speed_mps + margin)
    if relation == "sut_faster":
        return bool(sut_speed_mps >= adversary_speed_mps + margin)
    raise ValueError(f"unsupported target-contact speed relation: {relation}")


def interpolated_conflict_entry_role(
    previous_offsets: Mapping[str, float], current_offsets: Mapping[str, float],
) -> str | None:
    """Resolve first zone entry by interpolating route progress within a step.

    Comparing only post-step progress can label the vehicle farther downstream
    as first even when its crossing occurred later in the same physics step.
    A true tie is left unresolved instead of assigned to a Task label.
    """
    candidates: list[tuple[float, str]] = []
    for role in ("adversary", "sut"):
        before = float(previous_offsets[role])
        after = float(current_offsets[role])
        if before < 0.0 <= after:
            delta = after - before
            if delta <= 0.0:
                raise ValueError("conflict-entry offset must advance across a crossing")
            candidates.append((float(-before / delta), role))
    if not candidates:
        return None
    candidates.sort()
    if len(candidates) > 1 and math.isclose(candidates[0][0], candidates[1][0], abs_tol=1e-7):
        return None
    return candidates[0][1]


class LogicalMergeEnv(gym.Env):
    """RL controls the explicit adversary route; the SUT remains fixed IDM."""
    metadata = {"render_modes": ["human", "topdown"]}

    def __init__(self, task: LogicalScenarioTaskSpec, config: Mapping[str, Any], cases: list[Mapping[str, Any]], *, verify_geometry_hash: bool = True):
        super().__init__()
        task.validate()
        if not cases:
            raise ValueError("a task environment needs at least one frozen case")
        schema = str(config["environment"]["observation_schema"])
        expected_dim = observation_dim(schema)
        if int(config["environment"]["observation_dim"]) != expected_dim:
            raise ValueError("environment configuration and observation implementation use different dimensions")
        validate_reward_contract(config["reward"], int(config["environment"]["horizon"]))
        self.task, self.config, self.cases = task, dict(config), [dict(x) for x in cases]
        self.verify_geometry_hash = bool(verify_geometry_hash)
        self.adapter = adapter_for(task.logical_type)
        # The mechanism gates intentionally reduce the policy decision to the
        # longitudinal residual.  Steering remains the deterministic route
        # tracker output, so a policy difference has a direct physical
        # interpretation.  Normal training keeps the historical 2-D action.
        self._mechanism_longitudinal_only = bool(
            config.get("control", {}).get("mechanism_longitudinal_only", False)
        )
        self._mechanism_action_mode = str(
            config.get("control", {}).get("mechanism_action_mode", "longitudinal_residual")
        )
        if self._mechanism_action_mode not in {"longitudinal_residual", "target_arrival_gap"}:
            raise ValueError("unsupported control.mechanism_action_mode")
        if self._mechanism_action_mode != "longitudinal_residual" and not self._mechanism_longitudinal_only:
            raise ValueError("target_arrival_gap mechanism action requires 1-D longitudinal-only control")
        configured_action_dim = int(config["environment"]["action_dim"])
        expected_action_dim = 1 if self._mechanism_longitudinal_only else 2
        if configured_action_dim != expected_action_dim:
            raise ValueError(
                "environment.action_dim is incompatible with "
                "control.mechanism_longitudinal_only"
            )
        self.observation_space = spaces.Box(-1.0, 1.0, (expected_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (expected_action_dim,), np.float32)
        self._rng = np.random.default_rng(task.case_seed)
        self._env: Any | None = None; self._case: dict[str, Any] | None = None
        self.adversary: Any | None = None; self.sut: Any | None = None; self._frame: dict[str, Any] | None = None
        self._previous_action = np.zeros(2, dtype=np.float32); self._metrics: EpisodeMetrics | None = None
        self._route_progress: dict[str, float | None] = {"adversary": None, "sut": None}
        self._first_conflict_entry_role: str | None = None
        self._conflict_entry_resolved = False
        self._previous_conflict_entry_offsets: dict[str, float] = {"adversary": -math.inf, "sut": -math.inf}
        self._runtime_map_hash: str | None = None
        self._mask_topology_for_episode = False

    def _choose_case(self, seed: int | None, options: Mapping[str, Any]) -> dict[str, Any]:
        if "case" in options:
            return dict(options["case"])
        if "case_id" in options:
            return next(dict(case) for case in self.cases if case["case_id"] == options["case_id"])
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return dict(self.cases[int(self._rng.integers(len(self.cases)))])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.close()
        case = self._choose_case(seed, options or {})
        env = self.adapter.build_env(self.task, case, self.config)
        try:
            map_seed = int(self.task.map_config.get("start_seed", case["case_seed"]))
            try:
                env.reset(seed=map_seed)
            except TypeError:
                env.reset(force_seed=map_seed)
            adversary, sut = self.adapter.establish_roles(env, self.task, case, self.config)
            frame = self.adapter.conflict_frame(env, self.task, adversary, sut)
            frame["priority_spec"] = dict(self.task.priority_spec)
            runtime_hash = self.adapter.map_hash(env)
            if self.verify_geometry_hash and runtime_hash != self.task.map_hash:
                raise RuntimeError(f"task {self.task.task_id} map hash mismatch: expected {self.task.map_hash}, got {runtime_hash}")
        except Exception:
            env.close()
            raise
        self._env, self._case, self.adversary, self.sut, self._frame = env, case, adversary, sut, frame
        self._runtime_map_hash = runtime_hash
        self._previous_action.fill(0.0); self._route_progress = {"adversary": None, "sut": None}
        self._first_conflict_entry_role = None
        self._conflict_entry_resolved = False
        self._previous_conflict_entry_offsets = self._conflict_entry_offsets()
        dropout = float(self.config.get("regularization", {}).get("topology_dropout_probability", 0.0))
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("topology_dropout_probability must lie in [0, 1]")
        # Keep the descriptor mask constant for an entire rollout.  The seed is
        # derived only from frozen task/case data, making replay reproducible
        # while exposing training to both topology-present and topology-masked
        # episodes across the case pool.
        dropout_seed = (int(self.task.case_seed) * 1_000_003 + int(case["case_seed"])) & 0xFFFFFFFF
        self._mask_topology_for_episode = bool(np.random.default_rng(dropout_seed).random() < dropout)
        metric_schema = str(self.config.get("critical_metric", {}).get("schema", "legacy_logical_merge_critical"))
        self._metrics = EpisodeMetrics(self.task.task_id, str(case["case_id"]), metric_schema=metric_schema)
        return self._observation(), self._info("reset")

    def geometry_provenance(self) -> dict[str, str]:
        if self._runtime_map_hash is None or self._frame is None:
            raise RuntimeError("reset before requesting geometry provenance")
        from .io import content_hash
        route_payload = lambda route: {"lane_indices": [list(index) for index in route.lane_indices], "points": route.points.round(6).tolist()}
        return {
            "map_hash": self._runtime_map_hash,
            "adversary_route_hash": content_hash(route_payload(self._frame["adversary_route"])),
            "sut_route_hash": content_hash(route_payload(self._frame["sut_route"])),
            "conflict_hash": content_hash({"origin": np.asarray(self._frame["origin"]).round(6).tolist(), "radius_m": self._frame["radius_m"]}),
        }

    def physical_task_descriptor(self) -> PhysicalTaskDescriptor:
        """Return the frozen allowlisted descriptor for the initialized map."""
        if self._env is None or self._frame is None:
            raise RuntimeError("reset before requesting the physical task descriptor")
        moe = self.config.get("networks", {}).get("moe", {})
        return physical_task_descriptor(
            self.adapter.topology_features(self._env, self.task),
            schema=str(moe.get("descriptor_schema", "")),
            normalization=dict(moe.get("descriptor_normalization", {})),
        )

    def _observation(self) -> np.ndarray:
        config = self.config
        if self._mask_topology_for_episode:
            config = {**config, "ablation": {**config.get("ablation", {}), "no_topology": True}}
        return build_observation(self.adversary, self.sut, self._frame, self.adapter.topology_features(self._env, self.task), config)

    def _out_of_road(self, vehicle: Any) -> bool:
        checker = getattr(self._env, "_is_out_of_road", None)
        return bool(checker(vehicle)) if checker else bool(getattr(vehicle, "out_of_road", False))

    def _conflict_entry_offsets(self) -> dict[str, float]:
        """Signed route progress beyond each role's conflict-zone entrance."""
        result: dict[str, float] = {}
        for role, vehicle in (("adversary", self.adversary), ("sut", self.sut)):
            route = self._frame[f"{role}_route"]
            conflict_s = float(self._frame[f"{role}_conflict_s_m"])
            projection = route.projection(
                vehicle.position,
                float(getattr(vehicle, "heading_theta", 0.0)),
                float(getattr(getattr(vehicle, "lane", None), "width", 3.8)),
            )
            result[role] = float(projection.s_m - (conflict_s - float(self._frame["radius_m"])))
        return result

    def _update_conflict_entry_order(self) -> None:
        if self._conflict_entry_resolved:
            return
        current_offsets = self._conflict_entry_offsets()
        role = interpolated_conflict_entry_role(self._previous_conflict_entry_offsets, current_offsets)
        crossed = any(
            self._previous_conflict_entry_offsets[key] < 0.0 <= current_offsets[key]
            for key in current_offsets
        )
        self._previous_conflict_entry_offsets = current_offsets
        if crossed:
            self._first_conflict_entry_role = role
            self._conflict_entry_resolved = True

    def _arrival_state(self) -> dict[str, float | str]:
        """Return route-based arrival times and distances for both roles.

        Unlike a historical zone-entry marker, these quantities remain
        controllable by the adversary at the contact decision point.
        """
        result: dict[str, float | str] = {}
        arrivals: list[tuple[float, str]] = []
        for role, vehicle in (("adversary", self.adversary), ("sut", self.sut)):
            route = self._frame[f"{role}_route"]
            conflict_s = float(self._frame[f"{role}_conflict_s_m"])
            projection = route.projection(vehicle.position, float(getattr(vehicle, "heading_theta", 0.0)), float(getattr(getattr(vehicle, "lane", None), "width", 3.8)))
            speed = float(np.dot(np.asarray(vehicle.velocity, dtype=float), projection.tangent))
            signed_distance = conflict_s - projection.s_m
            distance = max(0.0, signed_distance)
            arrival_time = distance / max(speed, 0.1)
            signed_arrival_time = signed_distance / max(speed, 0.1)
            result[f"{role}_time_s"] = arrival_time
            result[f"{role}_signed_time_s"] = signed_arrival_time
            result[f"{role}_distance_m"] = distance
            result[f"{role}_signed_distance_m"] = signed_distance
            arrivals.append((arrival_time, role))
        result["order"] = min(arrivals)[1]
        return result

    def initial_case_measurements(self) -> dict[str, float | str | bool]:
        """Expose geometry-derived initial-condition facts for case building.

        These values are provenance only. They are never appended to the
        policy observation or the frozen static task descriptor.
        """
        if self._env is None or self._frame is None:
            raise RuntimeError("reset before requesting initial case measurements")
        arrival = self._arrival_state()
        pair_distance = float(np.linalg.norm(
            np.asarray(self.adversary.position, dtype=float) - np.asarray(self.sut.position, dtype=float)
        ))
        initial_target_overlap, _ = self.adapter.target_contact(
            self._env, self.adversary, self.sut,
        )
        return {
            **arrival,
            "initial_pair_distance_m": pair_distance,
            "initial_target_overlap": bool(initial_target_overlap),
            "adversary_initial_speed_mps": float(np.linalg.norm(self.adversary.velocity)),
            "sut_initial_speed_mps": float(np.linalg.norm(self.sut.velocity)),
            "initial_relative_speed_mps": float(np.linalg.norm(self.adversary.velocity) - np.linalg.norm(self.sut.velocity)),
            "conflict_radius_m": float(self._frame["radius_m"]),
        }

    def _reward_shaping(
        self,
        previous_adversary_s_m: float | None,
        adversary_s_m: float,
        arrival: Mapping[str, float | str],
    ) -> dict[str, float]:
        cfg = self.config["reward"]
        conflict_s = float(self._frame["adversary_conflict_s_m"])
        progress_clip = max(float(cfg.get("route_progress_clip_m", 1.0)), 1e-6)
        if previous_adversary_s_m is None:
            route_progress = 0.0
        else:
            before = max(0.0, conflict_s - float(previous_adversary_s_m))
            after = max(0.0, conflict_s - float(adversary_s_m))
            route_progress = float(np.clip((before - after) / progress_clip, -1.0, 1.0))

        desired = str(self.task.priority_spec["target_contact_entry_order"])
        time_difference = float(arrival["sut_time_s"]) - float(arrival["adversary_time_s"])
        desired_sign = 1.0 if desired == "adversary_first" else -1.0
        time_scale = max(float(cfg.get("priority_alignment_time_scale_s", 2.0)), 1e-6)
        distance_scale = max(float(cfg.get("priority_alignment_distance_scale_m", 20.0)), 1e-6)
        adversary_signed_distance = float(arrival["adversary_signed_distance_m"])
        sut_signed_distance = float(arrival["sut_signed_distance_m"])
        # Reward the desired ordering only while both vehicles are still
        # approaching the conflict and both are nearby.  Using the nearest
        # vehicle or clamped post-conflict distances lets a policy collect a
        # large reward forever after sending only the SUT through.
        if adversary_signed_distance <= 0.0 or sut_signed_distance <= 0.0:
            conflict_gate = 0.0
        else:
            farthest_conflict_distance = max(adversary_signed_distance, sut_signed_distance)
            conflict_gate = float(np.exp(-farthest_conflict_distance / distance_scale))
        priority_alignment = (
            0.0
            if desired == "any"
            else conflict_gate * float(np.tanh(desired_sign * time_difference / time_scale))
        )

        route = self._frame["adversary_route"]
        lane_width = float(getattr(getattr(self.adversary, "lane", None), "width", 3.8))
        projection = route.projection(
            self.adversary.position,
            float(getattr(self.adversary, "heading_theta", 0.0)),
            lane_width,
        )
        lateral = min(abs(projection.lateral_m) / max(lane_width, 1e-6), 1.0)
        heading = min(abs(projection.heading_error) / (np.pi / 2.0), 1.0)
        return {
            "route_progress": route_progress,
            "priority_alignment": priority_alignment,
            "route_deviation": lateral + heading,
        }

    def _info(self, reason: str, **extra: Any) -> dict[str, Any]:
        return {"task_id": self.task.task_id, "geometry_id": self.task.geometry_id, "logical_type": self.task.logical_type, "case_id": self._case["case_id"], "observation_schema": str(self.config["environment"]["observation_schema"]), "sut_controller": "IDMPolicy fixed_parameters", "sut_target_speed_mps": float(self.config["sut"]["target_speed_mps"]), "termination_reason": reason, "map_hash": self._runtime_map_hash, **extra}

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("reset must be called before step")
        policy_action = np.asarray(action, dtype=np.float32)
        expected_action_shape = (1,) if self._mechanism_longitudinal_only else (2,)
        if policy_action.shape != expected_action_shape or not np.all(np.isfinite(policy_action)):
            raise ValueError(f"action must be finite shape {expected_action_shape}")
        policy_action = np.clip(policy_action, -1.0, 1.0)
        pre_step_arrival = self._arrival_state()
        # Reward shaping and the MetaDrive adapter retain the canonical
        # [steering_residual, longitudinal] representation internally.  The
        # target-arrival mode is a mechanism-only hierarchical contract: the
        # policy chooses a signed target ETA offset, while a frozen feedback
        # controller performs the low-level throttle tracking.
        longitudinal = float(policy_action[0])
        if self._mechanism_action_mode == "target_arrival_gap":
            control = dict(self.config.get("control", {}).get("arrival_gap_controller", {}))
            target_scale = float(control.get("target_scale_s", 0.08))
            gain = float(control.get("gain", 2.0))
            time_scale = float(self.config.get("normalization", {}).get("time_s", 10.0))
            if target_scale <= 0.0 or gain <= 0.0 or time_scale <= 0.0:
                raise ValueError("arrival-gap controller scales and gain must be positive")
            current_gap = float(pre_step_arrival["adversary_time_s"]) - float(pre_step_arrival["sut_time_s"])
            target_gap = float(policy_action[0]) * target_scale
            longitudinal = float(np.clip(gain * (current_gap - target_gap) / time_scale, -1.0, 1.0))
        action = (
            np.asarray([0.0, longitudinal], dtype=np.float32)
            if self._mechanism_longitudinal_only else policy_action
        )
        tracking = self.config.get("control", {}).get("route_tracking", {})
        route = self._frame["adversary_route"]
        lane_width = float(getattr(getattr(self.adversary, "lane", None), "width", 3.8))
        projection = route.projection(
            self.adversary.position, float(self.adversary.heading_theta), lane_width,
        )
        lookahead = max(0.0, float(tracking.get("lookahead_m", 10.0)))
        tangent = route.tangent_at_s(min(route.length_m, projection.s_m + lookahead))
        desired_heading = math.atan2(float(tangent[1]), float(tangent[0]))
        heading_error_to_lookahead = float(
            (desired_heading - float(self.adversary.heading_theta) + math.pi) % (2.0 * math.pi) - math.pi
        )
        applied_action = compose_route_tracking_action(
            action, heading_error_to_lookahead, projection.lateral_m, self.config,
        )
        adversary_speed = float(np.linalg.norm(np.asarray(self.adversary.velocity, dtype=float)))
        sut_speed = float(np.linalg.norm(np.asarray(self.sut.velocity, dtype=float)))
        pre_step_arrival_order = str(pre_step_arrival["order"])
        previous_adversary_progress = self._route_progress["adversary"]
        _, _, _, _, upstream_info = self._env.step(applied_action)
        self._update_conflict_entry_order()
        physical_target_contact, method = self.adapter.target_contact(self._env, self.adversary, self.sut)
        target_contact_rule_satisfied = bool(
            physical_target_contact and target_contact_matches_rule(
                self.task.priority_spec, adversary_speed, sut_speed, self._first_conflict_entry_role,
            )
        )
        target = bool(physical_target_contact and target_contact_rule_satisfied)
        adv_progress, adv_wrong, adv_complete = self.adapter.route_status(
            self._env, self.adversary, "adversary", self._route_progress["adversary"]
        )
        sut_progress, sut_wrong, sut_complete = self.adapter.route_status(
            self._env, self.sut, "sut", self._route_progress["sut"]
        )
        self._route_progress.update({"adversary": adv_progress, "sut": sut_progress})
        post_step_arrival = self._arrival_state()
        ttc, distance, closing_speed = _ttc(self.adversary, self.sut, float(self.config["reward"]["ttc_cap"]))
        evaluation = self.config.get("evaluation", {})
        critical_threshold = float(evaluation.get("critical_ttc_threshold", 1.5))
        conflict_gap_threshold = float(evaluation.get("conflict_arrival_gap_threshold_s", 1.5))
        conflict_horizon = float(evaluation.get("conflict_lookahead_horizon_s", 3.0))
        both_approaching_conflict = bool(
            float(post_step_arrival["adversary_signed_distance_m"]) > 0.0
            and float(post_step_arrival["sut_signed_distance_m"]) > 0.0
        )
        route_conflict_proximity = bool(
            both_approaching_conflict
            and max(
                float(post_step_arrival["adversary_time_s"]),
                float(post_step_arrival["sut_time_s"]),
            ) <= conflict_horizon
            and abs(
                float(post_step_arrival["adversary_time_s"])
                - float(post_step_arrival["sut_time_s"])
            ) <= conflict_gap_threshold
        )
        proximity_rule_satisfied = target_contact_matches_rule(
            self.task.priority_spec, adversary_speed, sut_speed, str(post_step_arrival["order"]),
        )
        physical_critical_proximity = bool(
            not physical_target_contact
            and (ttc <= critical_threshold or route_conflict_proximity)
        )
        any_crash = bool(getattr(self.adversary, "crash_vehicle", False) or getattr(self.sut, "crash_vehicle", False))
        invalid_flags = {
            "adversary_out_of_road": self._out_of_road(self.adversary) and not adv_complete,
            "sut_out_of_road": self._out_of_road(self.sut) and not sut_complete,
            "wrong_route": bool(adv_wrong or sut_wrong),
        }
        metric_cfg = dict(self.config.get("critical_metric", {}))
        measurements: dict[str, float | bool] | None = None
        valid_near_miss = False
        near_miss_order_satisfied = True
        metric_schema = str(metric_cfg.get("schema", ""))
        if is_strict_near_miss_schema(metric_schema):
            effective_metric_cfg = thresholds_for_task(self.config, self.task)
            measurements = critical_measurements(
                post_step_arrival,
                pair_distance_m=distance,
                ttc_s=ttc,
                closing_speed_mps=closing_speed,
                thresholds=effective_metric_cfg,
            )
            order_required = metric_schema == LOGICAL_ORDER_CRITICAL_METRIC_SCHEMA
            required_order = str(self.task.priority_spec.get("target_contact_entry_order", "any"))
            near_miss_order_satisfied = bool(
                not order_required
                or conflict_entry_order_satisfied(required_order, self._first_conflict_entry_role)
            )
            valid_near_miss = bool(
                measurements["spatiotemporal_near_miss_candidate"]
                and not physical_target_contact
                and not any_crash
                and not any(invalid_flags.values())
                and near_miss_order_satisfied
            )
            if bool(metric_cfg.get("calibration_trace_mode", False)):
                valid_near_miss = False
        strict_metric = is_strict_near_miss_schema(metric_schema)
        events = {
            "target_collision": target,
            # In a collision-free near-miss experiment, a target collision is
            # an observed safety failure rather than an alternate success.
            # Historical logical-contact schemas retain their existing bonus.
            "target_collision_disqualifying": strict_metric,
            "physical_critical_proximity": physical_critical_proximity,
            "route_conflict_proximity": route_conflict_proximity,
            "rule_satisfied_critical_proximity": bool(
                physical_critical_proximity
                and proximity_rule_satisfied
            ),
            "valid_critical_near_miss": valid_near_miss,
            "near_miss_order_satisfied": near_miss_order_satisfied,
            "non_target_collision": (any_crash or physical_target_contact) and not target,
            "adversary_out_of_road": invalid_flags["adversary_out_of_road"],
            "sut_out_of_road": invalid_flags["sut_out_of_road"],
            "lane_marking_violation": bool(getattr(self.adversary, "on_yellow_continuous_line", False)),
            "wrong_route": invalid_flags["wrong_route"],
            "adversary_route_complete": bool(adv_complete),
            "sut_route_complete": bool(sut_complete),
        }
        if not strict_metric:
            events.pop("valid_critical_near_miss")
        shaping = self._reward_shaping(previous_adversary_progress, adv_progress, post_step_arrival)
        if strict_metric and measurements is not None and not any_crash and not any(invalid_flags.values()):
            required_order = str(self.task.priority_spec.get("target_contact_entry_order", "any"))
            prospective_order_satisfied = conflict_entry_order_satisfied(
                required_order, str(post_step_arrival["order"]),
            )
            thresholds = thresholds_for_task(self.config, self.task)
            shaping["strict_near_miss_potential"] = strict_near_miss_potential(
                measurements, thresholds,
                prospective_order_satisfied=prospective_order_satisfied,
            )
            shaping["collision_risk_barrier"] = collision_risk_barrier(
                measurements,
                thresholds,
                safe_pair_distance_ratio=float(self.config["reward"].get(
                    "collision_risk_barrier_safe_pair_ratio", 0.85,
                )),
            )
        else:
            shaping["strict_near_miss_potential"] = 0.0
            shaping["collision_risk_barrier"] = 0.0
        reward = compute_reward(
            ttc, distance, action, self._previous_action, events, self.config["reward"], shaping,
        )
        self._metrics.update(reward.total, ttc, distance, events, method, measurements)
        self._previous_action = action.copy()
        critical_terminal = "valid_critical_near_miss" if strict_metric else "physical_critical_proximity"
        terminal_events = (
            "target_collision", critical_terminal, "non_target_collision", "adversary_out_of_road",
            "sut_out_of_road", "wrong_route", "adversary_route_complete", "sut_route_complete",
        )
        terminated = bool(any(events[key] for key in terminal_events))
        truncated = bool(self._metrics.episode_length >= int(self.config["environment"]["horizon"])) and not terminated
        reason = "target_collision" if target else (
            critical_terminal
            if events[critical_terminal]
            else next(
            (key for key in terminal_events[1:] if events[key]),
            "horizon" if truncated else "running",
            )
        )
        if terminated or truncated:
            self._metrics.termination_reason = reason
        info = self._info(reason, critical_metric_schema=str(metric_cfg.get("schema", "legacy_logical_merge_critical")), mechanism_action_mode=self._mechanism_action_mode, ttc=ttc, distance=distance, min_ttc=self._metrics.min_ttc, min_distance=self._metrics.min_distance, target_contact_method=method, physical_target_contact=bool(physical_target_contact), target_contact_rule_satisfied=target_contact_rule_satisfied, pre_step_arrival_order=pre_step_arrival_order, conflict_arrival_gap_s=abs(float(post_step_arrival["adversary_time_s"]) - float(post_step_arrival["sut_time_s"])), first_conflict_entry_role=self._first_conflict_entry_role, policy_action=policy_action.tolist(), applied_action=applied_action.tolist(), reward_components=reward.as_dict(), reward_shaping=shaping, adversary_route_progress_m=adv_progress, sut_route_progress_m=sut_progress, adversary_lane_index=list(getattr(self.adversary, "lane_index", ())), sut_lane_index=list(getattr(self.sut, "lane_index", ())), **(measurements or {}), **events)
        info.update(dict(upstream_info or {}))
        return self._observation(), float(reward.total), terminated, truncated, info

    def episode_record(self) -> dict[str, object]:
        if self._metrics is None:
            raise RuntimeError("no completed/reset episode")
        return self._metrics.record(float(self.config.get("evaluation", {}).get("critical_ttc_threshold", 1.5)))

    def close(self) -> None:
        if self._env is not None:
            self._env.close(); self._env = None


def freeze_physical_task_descriptor(
    task: LogicalScenarioTaskSpec,
    config: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
) -> PhysicalTaskDescriptor:
    """Initialize one trusted case, freeze its static descriptor, then close it."""
    env = LogicalMergeEnv(task, config, cases)
    try:
        env.reset(options={"case": cases[0]})
        return env.physical_task_descriptor()
    finally:
        env.close()
