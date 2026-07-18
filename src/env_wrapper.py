"""
MetaDrive adversarial NPC environment wrapper.

The wrapper exposes a Gymnasium-compatible interface for training an
adversarial NPC agent. The ego vehicle runs IDM; our agent controls one
NPC vehicle and receives an adversarial reward based on TTC and collision.

Observation (38-dim, normalised to [-1, 1]):
  [0:6]   own kinematics   : v_x, v_y, heading, yaw_rate, pos_x, pos_y
  [6:30]  4 nearest others : dx, dy, dv_x, dv_y, heading_rel, dist  (per vehicle)
  [30:38] road geometry    : 4 waypoint_dirs (2D each = 8 dims)
               + lane_width, curvature, num_lanes, merge_flag
              → re-encoded as [wp_dir_0..3 (4 floats), lane_width, curvature, num_lanes, merge_flag]

Action (2-dim, continuous, clipped to [-1, 1]):
  [0] normalised longitudinal acceleration  (mapped to [-4, 4] m/s^2)
  [1] normalised steering                   (mapped to [-0.5, 0.5] rad)

Reward:
  r = -min(TTC, 5) / 5  +  3 * collision  -  0.01 * ||a||^2
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.component.road_network.node_road_network import NodeRoadNetwork
    METADRIVE_AVAILABLE = True
except ImportError:
    METADRIVE_AVAILABLE = False

# ─── topology family → MetaDrive map string ──────────────────────────────────
TOPOLOGY_MAP = {
    "highway":      {"map": "SSSSSS",   "num_lanes": 3, "speed_limit": 20},
    "merge":        {"map": "SrS",      "num_lanes": 2, "speed_limit": 20},
    "t_junction":   {"map": "STSS",     "num_lanes": 2, "speed_limit": 15},
    "intersection": {"map": "SXSS",     "num_lanes": 2, "speed_limit": 15},
    "roundabout":   {"map": "OSSO",     "num_lanes": 2, "speed_limit": 12},
    "y_junction":   {"map": "STSS",     "num_lanes": 2, "speed_limit": 15},
    # y_junction uses same block type as t_junction but different num_lanes config
}

OBS_DIM = 38
ACT_DIM = 2
N_NPC_OBS = 4

# running normalisation statistics (populated from training topologies)
_OBS_MEAN = np.zeros(OBS_DIM, dtype=np.float32)
_OBS_STD  = np.ones(OBS_DIM, dtype=np.float32)


def set_normalisation_stats(mean: np.ndarray, std: np.ndarray):
    global _OBS_MEAN, _OBS_STD
    _OBS_MEAN = mean.astype(np.float32)
    _OBS_STD  = np.maximum(std, 1e-8).astype(np.float32)


class AdversarialNPCEnv(gym.Env):
    """
    Adversarial NPC environment wrapping MetaDrive.

    If MetaDrive is not installed, a lightweight toy environment is used
    so unit tests and the code-review step can run without the simulator.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        topology: str = "highway",
        num_scenarios: int = 20,
        horizon: int = 500,
        seed: int = 0,
        ttc_scale: float = 5.0,
        collision_bonus: float = 3.0,
        action_reg: float = 0.01,
        use_toy_env: bool = False,
    ):
        super().__init__()
        self.topology = topology
        self.horizon = horizon
        self.ttc_scale = ttc_scale
        self.collision_bonus = collision_bonus
        self.action_reg = action_reg
        self.use_toy_env = use_toy_env or not METADRIVE_AVAILABLE
        self._num_scenarios = num_scenarios
        self._init_seed = seed

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        if not self.use_toy_env:
            topo_cfg = TOPOLOGY_MAP[topology]
            self._env = MetaDriveEnv({
                "map": topo_cfg["map"],
                "num_scenarios": num_scenarios,
                "start_seed": seed,
                "traffic_density": 0.2,        # ensure at least one traffic vehicle
                "use_render": False,
                "random_traffic": True,
                "crash_vehicle_done": False,   # don't terminate on crash
                "crash_object_done": False,
                "out_of_road_done": True,
                "vehicle_config": {"enable_reverse": False},
            })
            # default_agent is the vehicle we control (adversarial NPC).
            # Ego (target) is identified at reset time from traffic vehicles.
            self._ego_vehicle = None
        else:
            self._env = None

        self._step_count = 0
        self._last_ego_speed = 0.0
        self._rng = np.random.default_rng(seed)

    # ── Gymnasium interface ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        if self.use_toy_env:
            return self._toy_reset(), {}

        # MetaDrive occasionally fails to spawn agents on reset; retry once.
        try:
            obs_raw, info = self._env.reset()
        except Exception:
            self._env.close()
            topo_cfg = TOPOLOGY_MAP[self.topology]
            self._env = MetaDriveEnv({
                "map": topo_cfg["map"],
                "num_scenarios": self._num_scenarios,
                "start_seed": self._init_seed,
                "traffic_density": 0.2,
                "use_render": False,
                "random_traffic": True,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "out_of_road_done": True,
                "vehicle_config": {"enable_reverse": False},
            })
            obs_raw, info = self._env.reset()

        self._ego_vehicle = self._pick_ego_vehicle()
        return self._build_obs(), info

    def step(self, action: np.ndarray):
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)

        if self.use_toy_env:
            return self._toy_step(action)

        # Map normalised action → MetaDrive action
        accel  = float(action[0]) * 4.0   # [-4, 4] m/s^2
        steer  = float(action[1]) * 0.5   # [-0.5, 0.5] rad

        # Step the environment — ego uses IDM (default MetaDrive policy)
        # We inject our NPC action by overriding the first NPC vehicle
        npc_action = np.array([accel, steer], dtype=np.float32)
        _, reward_env, terminated, truncated, info = self._env.step(npc_action)

        obs = self._build_obs()
        reward = self._compute_reward(action, info)
        done = terminated or truncated or self._step_count >= self.horizon

        # Record trajectory fingerprint
        info["ttc"]             = self._compute_ttc()
        info["collision"]       = info.get("crash_vehicle", False)
        info["npc_action"]      = action.copy()
        info["npc_pos"]         = self._get_npc_pos()
        info["ego_speed"]       = self._get_ego_speed()

        return obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()

    # ── Observation builder ──────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Build the 38-dim observation from MetaDrive state."""
        if self.use_toy_env:
            return self._toy_obs()

        npc = self._get_npc_vehicle()
        ego = self._get_ego_vehicle()

        # [0:6] Own kinematics
        v = npc.velocity        # (vx, vy)
        pos = npc.position      # (px, py)
        heading = npc.heading_theta
        yaw_rate = getattr(npc, "angular_velocity", 0.0)
        own_kin = np.array([v[0], v[1], heading, yaw_rate, pos[0], pos[1]], dtype=np.float32)

        # [6:30] 4 nearest other vehicles (ego + 3 random NPCs)
        others = self._get_nearby_vehicles(npc, k=N_NPC_OBS)
        npc_feats = []
        for other in others:
            rel_pos = other.position - npc.position
            rel_vel = other.velocity - npc.velocity
            h_rel   = other.heading_theta - heading
            dist    = float(np.linalg.norm(rel_pos))
            npc_feats.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], h_rel, dist])
        while len(npc_feats) < N_NPC_OBS * 6:
            npc_feats.extend([0.0] * 6)
        npc_arr = np.array(npc_feats[:N_NPC_OBS * 6], dtype=np.float32)

        # [30:38] Road geometry (8 dims)
        road_arr = self._get_road_geometry(npc)

        obs_raw = np.concatenate([own_kin, npc_arr, road_arr])

        # Normalise
        obs_norm = (obs_raw - _OBS_MEAN) / _OBS_STD
        return np.clip(obs_norm, -1.0, 1.0).astype(np.float32)

    def _get_road_geometry(self, vehicle) -> np.ndarray:
        """
        Extract exactly 8-dim road geometry:
          [0:4]  heading angle (radians) to each of 4 upcoming waypoints
          [4]    lane_width  (m)
          [5]    curvature   (1/m, or 0)
          [6]    num_lanes   (float)
          [7]    merge_flag  (0/1)
        """
        try:
            navi = vehicle.navigation
            checkpoints = navi.checkpoints if hasattr(navi, "checkpoints") else []
            wp_headings = []
            for i in range(4):
                if i < len(checkpoints) - 1:
                    d    = np.array(checkpoints[i + 1]) - np.array(checkpoints[i])
                    heading = float(np.arctan2(d[1], d[0]))
                else:
                    heading = 0.0
                wp_headings.append(heading)
            lane_width = float(getattr(vehicle.lane, "width", 3.5)) if vehicle.lane else 3.5
            curvature  = float(getattr(vehicle.lane, "curvature", 0.0)) if vehicle.lane else 0.0
            num_lanes  = float(TOPOLOGY_MAP[self.topology].get("num_lanes", 2))
            merge_flag = 1.0 if "merge" in self.topology or "r" in TOPOLOGY_MAP[self.topology]["map"] else 0.0
            result = np.array(wp_headings + [lane_width, curvature, num_lanes, merge_flag],
                              dtype=np.float32)
            assert result.shape == (8,), f"road_geometry dim={result.shape}"
            return result
        except Exception:
            return np.zeros(8, dtype=np.float32)

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray, info: dict) -> float:
        ttc       = self._compute_ttc()
        collision = float(info.get("crash_vehicle", False))
        r_ttc     = -min(ttc, self.ttc_scale) / self.ttc_scale
        r_col     = self.collision_bonus * collision
        r_reg     = -self.action_reg * float(np.dot(action, action))
        return r_ttc + r_col + r_reg

    def _compute_ttc(self) -> float:
        """
        Estimate time-to-collision between NPC and ego.

        d(dist)/dt = dot(rel_pos_unit, rel_vel)
        Vehicles converge when d(dist)/dt < 0 (distance shrinking).
        TTC = dist / closing_speed  where closing_speed = -d(dist)/dt > 0
        """
        try:
            npc = self._get_npc_vehicle()
            ego = self._get_ego_vehicle()
            if npc is None or ego is None:
                return self.ttc_scale
            rel_pos = np.array(ego.position) - np.array(npc.position)
            rel_vel = np.array(ego.velocity)  - np.array(npc.velocity)
            dist    = float(np.linalg.norm(rel_pos))
            if dist < 1e-3:
                return 0.0   # already overlapping
            # rate = d(dist)/dt; negative = converging
            rate = float(np.dot(rel_pos, rel_vel)) / dist
            if rate < -0.1:                 # vehicles are converging
                return min(dist / (-rate), self.ttc_scale)
            return self.ttc_scale           # diverging or parallel → no imminent collision
        except Exception:
            return self.ttc_scale

    # ── Vehicle accessors ────────────────────────────────────────────────────

    def _get_npc_vehicle(self):
        """Return the vehicle we control = MetaDrive default_agent."""
        try:
            return self._env.agents.get("default_agent") or self._env.vehicle
        except Exception:
            return None

    def _pick_ego_vehicle(self):
        """
        At reset time: pick one IDM traffic vehicle as the 'ego' target.
        Returns the closest traffic vehicle to the controlled NPC, or None.
        """
        npc = self._get_npc_vehicle()
        try:
            traffic = list(self._env.engine.traffic_manager.vehicles)
            traffic = [v for v in traffic if v is not npc]
            if not traffic:
                return None
            if npc is not None:
                traffic.sort(key=lambda v: float(np.linalg.norm(
                    np.array(v.position) - np.array(npc.position))))
            return traffic[0]
        except Exception:
            return None

    def _get_ego_vehicle(self):
        """Return the stably-identified ego (target) vehicle for this episode."""
        return self._ego_vehicle

    def _get_nearby_vehicles(self, ref_vehicle, k: int):
        """Return k nearest vehicles (any type) excluding ref_vehicle."""
        try:
            npc = self._get_npc_vehicle()
            traffic = list(self._env.engine.traffic_manager.vehicles)
            # Include the stored ego + other traffic; exclude self
            candidates = [v for v in traffic if v is not ref_vehicle]
            candidates.sort(key=lambda v: float(np.linalg.norm(
                np.array(v.position) - np.array(ref_vehicle.position))))
            return candidates[:k]
        except Exception:
            return []

    def _get_ego_speed(self) -> float:
        try:
            ego = self._get_ego_vehicle()
            return float(np.linalg.norm(ego.velocity)) if ego else 0.0
        except Exception:
            return 0.0

    def _get_npc_pos(self) -> np.ndarray:
        try:
            npc = self._get_npc_vehicle()
            return np.array(npc.position, dtype=np.float32) if npc else np.zeros(2)
        except Exception:
            return np.zeros(2, dtype=np.float32)

    # ── Toy environment (fallback when MetaDrive not installed) ──────────────

    def _toy_reset(self) -> np.ndarray:
        self._toy_state = self._rng.uniform(-0.5, 0.5, OBS_DIM).astype(np.float32)
        return self._toy_state

    def _toy_obs(self) -> np.ndarray:
        return self._toy_state.copy()

    def _toy_step(self, action: np.ndarray):
        noise = self._rng.normal(0, 0.1, OBS_DIM).astype(np.float32)
        self._toy_state = np.clip(self._toy_state + noise, -1.0, 1.0)
        ttc = max(0.1, float(self._rng.exponential(2.0)))
        collision = ttc < 0.5
        reward = -min(ttc, self.ttc_scale) / self.ttc_scale + self.collision_bonus * collision
        done = self._step_count >= self.horizon
        info = {"ttc": ttc, "collision": collision, "npc_action": action.copy(),
                "npc_pos": np.zeros(2), "ego_speed": 10.0}
        return self._toy_state.copy(), reward, done, False, info


def make_env(topology: str, seed: int = 0, **kwargs) -> AdversarialNPCEnv:
    """Factory function for a single topology environment."""
    return AdversarialNPCEnv(topology=topology, seed=seed, **kwargs)


def collect_normalisation_stats(topologies: list, n_episodes: int = 50) -> tuple:
    """Collect running mean/std from training topologies for observation normalisation."""
    all_obs = []
    for topo in topologies:
        env = make_env(topo, seed=0)
        obs, _ = env.reset()
        all_obs.append(obs)
        for ep in range(n_episodes):
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, done, _, _ = env.step(action)
                all_obs.append(obs)
            env.reset()
        env.close()
    arr = np.stack(all_obs, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)
