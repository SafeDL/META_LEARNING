"""Gymnasium environment for SAC-controlled adversarial merge interactions."""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from . import metadrive_compat as md
from .metrics import EpisodeMetrics
from .observation import OBSERVATION_SCHEMA, build_observation
from .reward import compute_reward, compute_ttc


class Stage1AdversarialMergeEnv(gym.Env):
    metadata = {"render_modes": ["human", "topdown"]}

    def __init__(self,
                 config: Mapping[str, Any],
                 split: str = "train",
                 seed: int = 0):
        super().__init__()
        self.cfg = dict(config)
        self.env_cfg = self.cfg["environment"]
        self.norm = self.cfg["normalization"]
        self.reward_cfg = self.cfg["reward"]
        if self.env_cfg.get("topology") != "merge":
            raise ValueError("Stage 1 supports topology='merge' only")
        if self.env_cfg.get("scenario_source", "procedural") != "procedural":
            raise ValueError(
                "dataset scenario source is not part of the Stage 1 acceptance path"
            )
        self.split = split
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-1.0, 1.0, (38, ), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2, ), np.float32)
        self._env = None
        self._active_scenario_seed = None
        self.adversary_vehicle = self.sut_vehicle = None
        self._spawn = (0.0, 0.0)
        self._previous_action = np.zeros(2, np.float32)
        self._metrics = None
        split_cfg = self.cfg["scenario_split"].get(split)
        if split_cfg is None:
            raise ValueError(f"unknown scenario split: {split}")
        self._seeds = list(
            range(
                int(split_cfg["start_seed"]),
                int(split_cfg["start_seed"]) +
                int(split_cfg["num_scenarios"])))

    @property
    def unwrapped_metadrive_env(self):
        return self._env

    def _scenario_seed(self, reset_seed: int | None) -> int:
        if reset_seed is not None:
            self._rng = np.random.default_rng(reset_seed)
        return int(self._seeds[int(self._rng.integers(len(self._seeds)))])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario_seed = int((options or {}).get("scenario_seed",
                                                self._scenario_seed(seed)))
        if scenario_seed not in self._seeds:
            raise ValueError(
                f"scenario seed {scenario_seed} is outside {self.split} split")
        # MetaDrive 0.4.x constrains reset seeds to its construction range when
        # num_scenarios=1. Recreate only when moving to another split seed.
        if self._env is None or self._active_scenario_seed != scenario_seed:
            if self._env is not None: self._env.close()
            self._env = md.make_merge_env(self.env_cfg, scenario_seed)
            self._active_scenario_seed = scenario_seed
        last_error = None
        for _ in range(int(self.env_cfg["max_reset_retries"])):
            try:
                md.reset_env(self._env, scenario_seed)
                adv = md.adversary(self._env)
                candidates = [
                    v for v in md.traffic_vehicles(self._env)
                    if md.vehicle_id(v) != md.vehicle_id(adv)
                ]
                if candidates:
                    self.adversary_vehicle, self.sut_vehicle = adv, min(
                        candidates,
                        key=lambda v: np.linalg.norm(
                            np.asarray(v.position) - np.asarray(adv.position)))
                    self._spawn = tuple(np.asarray(adv.position, dtype=float))
                    self._previous_action.fill(0.0)
                    self._metrics = EpisodeMetrics(scenario_seed, 0)
                    return self._observation(), self._reset_info(scenario_seed)
                last_error = RuntimeError(
                    "no traffic vehicle available for fixed IDM SUT")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"could not establish adversary/SUT roles after {self.env_cfg['max_reset_retries']} resets: {last_error}"
        )

    def _reset_info(self, scenario_seed: int) -> dict[str, Any]:
        return {
            "scenario_seed": scenario_seed,
            "adversary_id": md.vehicle_id(self.adversary_vehicle),
            "sut_id": md.vehicle_id(self.sut_vehicle),
            "adversary_policy": "EnvInputPolicy",
            "sut_policy": "IDMPolicy",
            "topology": "merge",
            "observation_schema": OBSERVATION_SCHEMA
        }

    def _observation(self):
        nearby = [
            v for v in md.traffic_vehicles(self._env)
            if md.vehicle_id(v) != md.vehicle_id(self.sut_vehicle)
        ]
        nearby.sort(key=lambda v: np.linalg.norm(
            np.asarray(v.position) - np.asarray(self.adversary_vehicle.position
                                                )))
        return build_observation(md.state(self.adversary_vehicle),
                                 md.state(self.sut_vehicle),
                                 [md.state(v) for v in nearby], self._spawn,
                                 md.route_features(self.adversary_vehicle),
                                 self.norm)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2, ):
            raise ValueError(
                f"action must have shape (2,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or Inf")
        action = np.clip(action, -1.0, 1.0)
        _, _, _, _, env_info = self._env.step(
            action)  # MetaDrive action interface: [steering, throttle/brake]
        adv, sut = self.adversary_vehicle, self.sut_vehicle
        adv_s, sut_s = md.state(adv), md.state(sut)
        rel_p = np.array([sut_s["x"] - adv_s["x"], sut_s["y"] - adv_s["y"]])
        rel_v = np.array(
            [sut_s["vx"] - adv_s["vx"], sut_s["vy"] - adv_s["vy"]])
        ttc, distance = compute_ttc(rel_p, rel_v,
                                    float(self.reward_cfg["ttc_cap"])), float(
                                        np.linalg.norm(rel_p))
        target = md.target_contact(adv, sut)
        any_crash = bool(getattr(adv, "crash_vehicle", False))
        non_target = any_crash and not target
        events = {
            "target_collision": target,
            "any_vehicle_collision": any_crash,
            "non_target_collision": non_target,
            "object_collision": bool(getattr(adv, "crash_object", False)),
            "adversary_out_of_road": md.out_of_road(self._env, adv),
            "sut_out_of_road": md.out_of_road(self._env, sut),
            "wrong_way": bool(getattr(adv, "on_yellow_continuous_line", False))
        }
        critical_before = self._metrics.target_collision or self._metrics.min_ttc <= float(
            self.cfg["evaluation"]["critical_ttc_threshold"])
        events["invalid_before_critical"] = not critical_before and (
            events["non_target_collision"] or events["adversary_out_of_road"]
            or events["wrong_way"])
        breakdown = compute_reward(ttc, distance, action,
                                   self._previous_action, events,
                                   self.reward_cfg)
        delta = action - self._previous_action
        self._metrics.update(breakdown.total, ttc, distance, action, delta,
                             events)
        self._previous_action = action.copy()
        terminated = bool(target or events["adversary_out_of_road"]
                          or events["non_target_collision"]
                          or events["object_collision"])
        truncated = bool(
            self._metrics.episode_length >= int(self.env_cfg["horizon"]))
        reason = "target_collision" if target else "adversary_out_of_road" if events[
            "adversary_out_of_road"] else "non_target_collision" if non_target else "object_collision" if events[
                "object_collision"] else "horizon" if truncated else "running"
        if terminated or truncated: self._metrics.termination_reason = reason
        info = dict(env_info or {})
        info.update(self._reset_info(self._metrics.scenario_seed))
        info.update({
            "ttc": ttc,
            "min_ttc": self._metrics.min_ttc,
            "distance": distance,
            "min_distance": self._metrics.min_distance,
            "reward_components": breakdown.as_dict(),
            "termination_reason": reason,
            **events
        })
        return self._observation(), float(
            breakdown.total), terminated, truncated, info

    def episode_record(self) -> dict:
        return self._metrics.record(
            float(self.cfg["evaluation"]["critical_ttc_threshold"]))

    def render(self):
        return self._env.render(
            mode="top_down") if self._env is not None else None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
