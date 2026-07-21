"""Gymnasium facade for frozen, route-audited logical merge tasks."""
from __future__ import annotations

from typing import Any, Mapping
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .adapters import adapter_for
from .metrics import EpisodeMetrics
from .observation import OBSERVATION_DIM, OBSERVATION_SCHEMA, build_observation
from .reward import compute_reward
from .task_spec import LogicalScenarioTaskSpec


def _ttc(a: Any, b: Any, cap: float) -> tuple[float, float]:
    position = np.asarray(b.position, dtype=float) - np.asarray(a.position, dtype=float)
    velocity = np.asarray(b.velocity, dtype=float) - np.asarray(a.velocity, dtype=float)
    distance = float(np.linalg.norm(position))
    closing = float(np.dot(position, velocity) / max(distance, 1e-6))
    return (min(cap, distance / -closing) if closing < 0.0 else cap), distance


class LogicalMergeEnv(gym.Env):
    """RL controls the explicit adversary route; the SUT remains fixed IDM."""
    metadata = {"render_modes": ["human", "topdown"]}

    def __init__(self, task: LogicalScenarioTaskSpec, config: Mapping[str, Any], cases: list[Mapping[str, Any]], *, verify_geometry_hash: bool = True):
        super().__init__()
        task.validate()
        if not cases:
            raise ValueError("a task environment needs at least one frozen case")
        if config["environment"]["observation_schema"] != OBSERVATION_SCHEMA:
            raise ValueError("environment configuration and observation implementation use different schemas")
        if int(config["environment"]["observation_dim"]) != OBSERVATION_DIM:
            raise ValueError("environment configuration and observation implementation use different dimensions")
        self.task, self.config, self.cases = task, dict(config), [dict(x) for x in cases]
        self.verify_geometry_hash = bool(verify_geometry_hash)
        self.adapter = adapter_for(task.logical_type)
        self.observation_space = spaces.Box(-1.0, 1.0, (OBSERVATION_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._rng = np.random.default_rng(task.case_seed)
        self._env: Any | None = None; self._case: dict[str, Any] | None = None
        self.adversary: Any | None = None; self.sut: Any | None = None; self._frame: dict[str, Any] | None = None
        self._previous_action = np.zeros(2, dtype=np.float32); self._metrics: EpisodeMetrics | None = None
        self._route_progress: dict[str, float | None] = {"adversary": None, "sut": None}
        self._runtime_map_hash: str | None = None

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
        self._metrics = EpisodeMetrics(self.task.task_id, str(case["case_id"]))
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

    def _observation(self) -> np.ndarray:
        return build_observation(self.adversary, self.sut, self._frame, self.adapter.topology_features(self._env, self.task), self.config)

    def _out_of_road(self, vehicle: Any) -> bool:
        checker = getattr(self._env, "_is_out_of_road", None)
        return bool(checker(vehicle)) if checker else bool(getattr(vehicle, "out_of_road", False))

    def _info(self, reason: str, **extra: Any) -> dict[str, Any]:
        return {"task_id": self.task.task_id, "geometry_id": self.task.geometry_id, "logical_type": self.task.logical_type, "case_id": self._case["case_id"], "observation_schema": OBSERVATION_SCHEMA, "sut_controller": "IDMPolicy fixed_parameters", "sut_target_speed_mps": float(self.config["sut"]["target_speed_mps"]), "termination_reason": reason, "map_hash": self._runtime_map_hash, **extra}

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("reset must be called before step")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be finite shape (2,)")
        action = np.clip(action, -1.0, 1.0)
        _, _, _, _, upstream_info = self._env.step(action)
        target, method = self.adapter.target_contact(self._env, self.adversary, self.sut)
        adv_progress, adv_wrong = self.adapter.route_status(self._env, self.adversary, "adversary", self._route_progress["adversary"])
        sut_progress, sut_wrong = self.adapter.route_status(self._env, self.sut, "sut", self._route_progress["sut"])
        self._route_progress.update({"adversary": adv_progress, "sut": sut_progress})
        any_crash = bool(getattr(self.adversary, "crash_vehicle", False) or getattr(self.sut, "crash_vehicle", False))
        events = {
            "target_collision": target,
            "non_target_collision": any_crash and not target,
            "adversary_out_of_road": self._out_of_road(self.adversary),
            "sut_out_of_road": self._out_of_road(self.sut),
            "lane_marking_violation": bool(getattr(self.adversary, "on_yellow_continuous_line", False)),
            "wrong_route": bool(adv_wrong or sut_wrong),
        }
        ttc, distance = _ttc(self.adversary, self.sut, float(self.config["reward"]["ttc_cap"]))
        reward = compute_reward(ttc, distance, action, self._previous_action, events, self.config["reward"])
        self._metrics.update(reward.total, ttc, distance, events, method)
        self._previous_action = action.copy()
        terminal_events = ("target_collision", "non_target_collision", "adversary_out_of_road", "sut_out_of_road", "wrong_route")
        terminated = bool(any(events[key] for key in terminal_events))
        truncated = bool(self._metrics.episode_length >= int(self.config["environment"]["horizon"])) and not terminated
        reason = "target_collision" if target else next((key for key in terminal_events[1:] if events[key]), "horizon" if truncated else "running")
        if terminated or truncated:
            self._metrics.termination_reason = reason
        info = self._info(reason, ttc=ttc, distance=distance, min_ttc=self._metrics.min_ttc, min_distance=self._metrics.min_distance, target_contact_method=method, reward_components=reward.as_dict(), adversary_route_progress_m=adv_progress, sut_route_progress_m=sut_progress, **events)
        info.update(dict(upstream_info or {}))
        return self._observation(), float(reward.total), terminated, truncated, info

    def episode_record(self) -> dict[str, object]:
        if self._metrics is None:
            raise RuntimeError("no completed/reset episode")
        return self._metrics.record(float(self.config.get("evaluation", {}).get("critical_ttc_threshold", 1.5)))

    def close(self) -> None:
        if self._env is not None:
            self._env.close(); self._env = None
