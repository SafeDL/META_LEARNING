"""Gymnasium environment for SAC-controlled ``on_ramp_merge`` cases."""
from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import metadrive_compat as md
from .casebook import build_case_table, validate_case
from .metrics import EpisodeMetrics
from .observation import OBSERVATION_SCHEMA, build_observation
from .reward import compute_reward, compute_ttc


class Stage1AdversarialMergeEnv(gym.Env):
    """Fixed-role merge environment; SAC controls only the ramp adversary."""

    metadata = {"render_modes": ["human", "topdown"]}

    def __init__(self, config: Mapping[str, Any], split: str = "train", seed: int = 0):
        super().__init__()
        self.cfg = dict(config)
        self.env_cfg = self.cfg["environment"]
        self.norm = self.cfg["normalization"]
        self.reward_cfg = self.cfg["reward"]
        if self.env_cfg.get("topology") != "on_ramp_merge":
            raise ValueError("this environment supports topology='on_ramp_merge' only")
        self.split = split
        self._rng = np.random.default_rng(seed)
        self._cases = build_case_table(self.cfg, split)
        self.observation_space = spaces.Box(-1.0, 1.0, (38,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._env: Any | None = None
        self._case: dict[str, Any] | None = None
        self.adversary_vehicle = self.sut_vehicle = None
        self._spawn = (0.0, 0.0)
        self._previous_action = np.zeros(2, np.float32)
        self._metrics: EpisodeMetrics | None = None

    @property
    def active_case(self) -> dict[str, Any]:
        if self._case is None:
            raise RuntimeError("reset must be called before reading active_case")
        return dict(self._case)

    def case_table(self) -> list[dict[str, Any]]:
        return [dict(case) for case in self._cases]

    def _choose_case(self, reset_seed: int | None, options: Mapping[str, Any]) -> dict[str, Any]:
        if "case" in options:
            return validate_case(options["case"], self.cfg)
        if "case_id" in options:
            wanted = str(options["case_id"])
            for case in self._cases:
                if case["case_id"] == wanted:
                    return dict(case)
            raise ValueError(f"case_id {wanted!r} is not in the {self.split} case table")
        if reset_seed is not None:
            self._rng = np.random.default_rng(reset_seed)
        return dict(self._cases[int(self._rng.integers(len(self._cases)))])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        case = self._choose_case(seed, options or {})
        if self._env is None:
            self._env = md.make_merge_env(self.env_cfg, case)
            # With fixed roads and zero built-in traffic, all case randomness
            # is explicit.  Reusing one engine preserves that determinism and
            # avoids rebuilding Panda3D/physics for every episode.
        else:
            md.configure_adversary_spawn(self._env, case)
        try:
            md.reset_env(self._env, int(self.env_cfg.get("engine_seed", 0)))
            adv, sut = md.establish_case_roles(self._env, case)
            if not md.has_expected_roles(self._env, adv, sut):
                raise RuntimeError("MetaDrive reset did not establish ramp adversary and mainline SUT")
            distance = float(np.linalg.norm(np.asarray(adv.position) - np.asarray(sut.position)))
            if distance < float(self.cfg["logical_scenario"]["constraints"]["min_initial_distance_m"]):
                raise RuntimeError(f"case {case['case_id']} begins with unsafe role separation {distance:.3f} m")
        except Exception:
            self._env.close()
            self._env = None
            raise
        self._case = case
        self.adversary_vehicle, self.sut_vehicle = adv, sut
        self._spawn = tuple(np.asarray(adv.position, dtype=float))
        self._previous_action.fill(0.0)
        self._metrics = EpisodeMetrics(str(case["case_id"]), int(case["background_seed"]))
        return self._observation(), self._reset_info()

    def _reset_info(self) -> dict[str, Any]:
        return {
            "case_id": self._case["case_id"], "background_seed": self._case["background_seed"],
            "theta": dict(self._case["theta"]),
            "adversary_id": md.vehicle_id(self.adversary_vehicle),
            "sut_id": md.vehicle_id(self.sut_vehicle),
            "adversary_policy": "SAC continuous [steering, throttle_or_brake]",
            "sut_policy": "IDM baseline", "topology": "on_ramp_merge",
            "observation_schema": OBSERVATION_SCHEMA,
        }

    def _observation(self) -> np.ndarray:
        nearby = [v for v in md.traffic_vehicles(self._env) if md.vehicle_id(v) != md.vehicle_id(self.sut_vehicle)]
        nearby.sort(key=lambda v: np.linalg.norm(np.asarray(v.position) - np.asarray(self.adversary_vehicle.position)))
        return build_observation(md.state(self.adversary_vehicle), md.state(self.sut_vehicle),
                                 [md.state(v) for v in nearby], self._spawn,
                                 md.route_features(self.adversary_vehicle), self.norm)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector shaped (2,)")
        action = np.clip(action, -1.0, 1.0)
        _, _, _, _, env_info = self._env.step(action)
        adv, sut = self.adversary_vehicle, self.sut_vehicle
        adv_s, sut_s = md.state(adv), md.state(sut)
        rel_p = np.array([sut_s["x"] - adv_s["x"], sut_s["y"] - adv_s["y"]])
        rel_v = np.array([sut_s["vx"] - adv_s["vx"], sut_s["vy"] - adv_s["vy"]])
        ttc = compute_ttc(rel_p, rel_v, float(self.reward_cfg["ttc_cap"]))
        distance = float(np.linalg.norm(rel_p))
        target = md.target_contact(adv, sut)
        any_crash = bool(getattr(adv, "crash_vehicle", False))
        non_target = any_crash and not target
        events = {
            "target_collision": target, "any_vehicle_collision": any_crash,
            "non_target_collision": non_target,
            "object_collision": bool(getattr(adv, "crash_object", False)),
            "adversary_out_of_road": md.out_of_road(self._env, adv),
            "sut_out_of_road": md.out_of_road(self._env, sut),
            "wrong_way": bool(getattr(adv, "on_yellow_continuous_line", False)),
        }
        critical_before = self._metrics.target_collision or self._metrics.min_ttc <= float(self.cfg["evaluation"]["critical_ttc_threshold"])
        events["invalid_before_critical"] = not critical_before and (
            events["non_target_collision"] or events["adversary_out_of_road"] or
            events["sut_out_of_road"] or events["wrong_way"])
        breakdown = compute_reward(ttc, distance, action, self._previous_action, events, self.reward_cfg)
        delta = action - self._previous_action
        self._metrics.update(breakdown.total, ttc, distance, action, delta, events)
        self._previous_action = action.copy()
        terminated = bool(target or events["adversary_out_of_road"] or events["sut_out_of_road"] or
                          events["non_target_collision"] or events["object_collision"])
        truncated = bool(self._metrics.episode_length >= int(self.env_cfg["horizon"]))
        reason = ("target_collision" if target else "adversary_out_of_road" if events["adversary_out_of_road"]
                  else "sut_out_of_road" if events["sut_out_of_road"] else "non_target_collision" if non_target
                  else "object_collision" if events["object_collision"] else "horizon" if truncated else "running")
        if terminated or truncated:
            self._metrics.termination_reason = reason
        info = dict(env_info or {})
        info.update(self._reset_info())
        info.update({"ttc": ttc, "min_ttc": self._metrics.min_ttc, "distance": distance,
                     "min_distance": self._metrics.min_distance, "reward_components": breakdown.as_dict(),
                     "termination_reason": reason, **events})
        return self._observation(), float(breakdown.total), terminated, truncated, info

    def episode_record(self) -> dict[str, Any]:
        record = self._metrics.record(float(self.cfg["evaluation"]["critical_ttc_threshold"]))
        record["theta"] = self._case["theta"]
        return record

    def set_camera_target(self, role: str) -> None:
        """Follow the selected semantic role in MetaDrive's live chase view."""
        if self._env is None or self.adversary_vehicle is None or self.sut_vehicle is None:
            raise RuntimeError("reset must be called before selecting a camera target")
        targets = {"sut": self.sut_vehicle, "adversary": self.adversary_vehicle}
        if role not in targets:
            raise ValueError("camera role must be 'sut' or 'adversary'")
        md.track_vehicle(self._env, targets[role])

    def camera_frame(self) -> np.ndarray:
        """Return the SUT-following off-screen RGB frame for dual-view visualization."""
        if self._env is None:
            raise RuntimeError("reset must be called before reading a camera frame")
        return md.camera_frame(self._env)

    def render(self, view: str = "chase", text: Mapping[str, Any] | None = None, **kwargs: Any):
        """Render the interactive chase view or return an optional top-down frame."""
        if self._env is None:
            return None
        if view == "chase":
            return self._env.render(text=dict(text or {}), **kwargs)
        if view == "topdown":
            return self._env.render(text=dict(text or {}), mode="top_down", **kwargs)
        raise ValueError("render view must be 'chase' or 'topdown'")

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
