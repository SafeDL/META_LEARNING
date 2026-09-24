"""Paper-aligned DenseRL protocol adapted to the highway-env Cut-in harness.

This module deliberately keeps the original AdaTE separation between (1)
source surrogate challenge tables, (2) target-side expected-policy DenseRL,
(3) simplex coefficient fitting, and (4) an independent importance-sampling
evaluation.  It is an adaptation, not a reimplementation of the upstream
overtaking simulator.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from highway_env_benchmark.envs.cutin_env import CutInScenario
from sut_algorithms.highway_env.idm_profiles import get_profile

from .adaptive_policy import select_gap_action
from .dense_env import DenseCutInEnv, DenseSnapshot
from .dense_value import SparseExpectedTD, StateKey
from .importance import (
    defensive_mixture,
    log_importance_weight,
    mixture_of_policies,
    weighted_event_estimate,
)
from .mixture import QPDiagnostics, simplex_least_squares, uniform_alpha
from .state_encoder import StateEncoder


@dataclass(frozen=True)
class SnapshotSpec:
    """Replayable source-derived start state with no target outcome attached."""

    snapshot_id: str
    scenario: CutInScenario
    seed: int
    action_history: tuple[int, ...]
    source_profile: str
    source_state: StateKey
    source_criticality: float


@dataclass
class VariantResult:
    name: str
    alpha: np.ndarray
    learner: SparseExpectedTD
    alpha_rows: list[dict]
    transition_rows: list[dict]
    q_rows: list[dict]
    propensity_rows: list[dict]
    stopped_early: bool
    restore_steps: int


@dataclass(frozen=True)
class SourceStage:
    """Frozen source-side artefacts shared by several held-out targets."""

    values: list[dict[tuple[StateKey, int], float]]
    snapshots: list[SnapshotSpec]
    episodes: int
    transitions: int


def _scenario_pool(config: dict) -> list[CutInScenario]:
    design = config.get("scenario_design", {})
    gaps = [float(value) for value in design.get("gaps", [5, 10, 18, 30, 45])]
    relative_speeds = [float(value) for value in design.get("relative_speeds", [-10, -6, -2, 1])]
    modes = [
        str(value) for value in design.get("modes", [
            "fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go", "slow_lead_following"
        ])
    ]
    return [
        CutInScenario(gap, relative_speed, mode) for mode in modes for gap in gaps
        for relative_speed in relative_speeds
    ]


def _scenario_probabilities(config: dict, scenarios: list[CutInScenario]) -> np.ndarray:
    """Return the declared natural initial-state distribution over the grid."""
    design = config.get("scenario_design", {})
    gaps = [float(value) for value in design.get("gaps", [5, 10, 18, 30, 45])]
    speeds = [float(value) for value in design.get("relative_speeds", [-10, -6, -2, 1])]
    default_modes = list(dict.fromkeys(scenario.mode for scenario in scenarios))
    modes = [str(value) for value in design.get("modes", default_modes)]
    gap_weights = np.asarray(design.get("gap_probabilities", np.ones(len(gaps))), dtype=float)
    speed_weights = np.asarray(
        design.get("relative_speed_probabilities", np.ones(len(speeds))), dtype=float
    )
    mode_weights = np.asarray(design.get("mode_probabilities", np.ones(len(modes))), dtype=float)
    for name, values, expected in (
        ("gap_probabilities", gap_weights, len(gaps)),
        ("relative_speed_probabilities", speed_weights, len(speeds)),
        ("mode_probabilities", mode_weights, len(modes)),
    ):
        if values.shape != (expected,) or np.any(values < 0) or values.sum() <= 0:
            raise ValueError(f"{name} must be non-negative and match its scenario axis")
    gap_weight = dict(zip(gaps, gap_weights / gap_weights.sum(), strict=True))
    speed_weight = dict(zip(speeds, speed_weights / speed_weights.sum(), strict=True))
    mode_weight = dict(zip(modes, mode_weights / mode_weights.sum(), strict=True))
    probabilities = np.asarray([
        gap_weight[scenario.initial_gap]
        * speed_weight[scenario.relative_speed]
        * mode_weight[scenario.mode]
        for scenario in scenarios
    ])
    return probabilities / probabilities.sum()


def _scenario_id(scenario: CutInScenario) -> str:
    return f"{scenario.mode}_g{scenario.initial_gap:g}_dv{scenario.relative_speed:g}".replace(
        "-", "m")


def _source_matrix(source_values: list[dict[tuple[StateKey, int], float]], state: StateKey,
                   actions: int) -> np.ndarray | None:
    columns = []
    for model in source_values:
        values = [model.get((state, action)) for action in range(actions)]
        if any(value is None for value in values):
            return None
        columns.append(values)
    return np.asarray(columns, dtype=float).T


def _snapshot_from_env(env: DenseCutInEnv) -> DenseSnapshot:
    return env.snapshot()


def _collect_source_models(
    profiles: list[str],
    scenarios: list[CutInScenario],
    config: dict,
    encoder: StateEncoder,
    phi: np.ndarray,
    seed: int,
) -> tuple[list[dict[tuple[StateKey, int], float]], list[SnapshotSpec], int, int]:
    """Learn source Q_j tables off-policy under phi and build replayable S_c."""
    actions = phi.size
    horizon = int(config["horizon"])
    rollouts = int(config.get("source_rollouts_per_scenario", 3))
    epochs = int(config.get("source_td_epochs", 3))
    source_learners = [
        SparseExpectedTD(actions,
                         phi,
                         learning_rate=float(config["learning_rate"]),
                         gamma=float(config["gamma"])) for _ in profiles
    ]
    transitions: list[list[tuple[StateKey, int, StateKey | None, bool]]] = [[] for _ in profiles]
    candidates: list[tuple[str, CutInScenario, int, DenseSnapshot, StateKey]] = []
    source_episodes = 0
    for profile_index, profile_name in enumerate(profiles):
        for scenario_index, scenario in enumerate(scenarios):
            for repeat in range(rollouts):
                # Paired source rollouts use the same background action plan.
                # Controller differences, rather than unrelated random actions,
                # should determine whether surrogate state tables overlap.
                execution_seed = seed + 100 * scenario_index + repeat
                rng = np.random.default_rng(execution_seed)
                env = DenseCutInEnv(get_profile(profile_name), scenario)
                try:
                    env.reset(seed=execution_seed)
                    for time_step in range(horizon):
                        state = encoder.encode(env.state_features())
                        snapshot = _snapshot_from_env(env)
                        candidates.append(
                            (profile_name, scenario, execution_seed, snapshot, state))
                        # Cycling the first action makes each source Q(s0,a)
                        # observable; later actions use uniform coverage.
                        action = repeat % actions if time_step == 0 else int(rng.integers(actions))
                        _, _, terminal, truncated, _ = env.step(action)
                        next_state = None if terminal or truncated else encoder.encode(
                            env.state_features())
                        transitions[profile_index].append(
                            (state, action, next_state, bool(terminal)))
                        if terminal or truncated:
                            break
                    source_episodes += 1
                finally:
                    env.close()
    for learner, records in zip(source_learners, transitions):
        for _ in range(epochs):
            for state, action, next_state, terminal in records:
                learner.update(state, action, float(terminal), next_state, terminal, critical=True)
    source_values = [{(state, action): float(values[action])
                      for state, values in learner.values.items() for action in range(actions)}
                     for learner in source_learners]
    alpha = uniform_alpha(len(source_values))
    snapshots: list[SnapshotSpec] = []
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    for profile_name, scenario, execution_seed, snapshot, state in candidates:
        key = (_scenario_id(scenario), execution_seed, snapshot.action_history)
        if key in seen:
            continue
        matrix = _source_matrix(source_values, state, actions)
        if matrix is None:
            continue
        criticality = float(phi @ (matrix @ alpha))
        if criticality <= float(config.get("criticality_threshold", 1e-8)):
            continue
        seen.add(key)
        snapshots.append(
            SnapshotSpec(
                snapshot_id=f"sc_{len(snapshots):05d}",
                scenario=scenario,
                seed=execution_seed,
                action_history=snapshot.action_history,
                source_profile=profile_name,
                source_state=state,
                source_criticality=criticality,
            ))
    if not snapshots:
        raise RuntimeError(
            "No source-derived critical snapshots; increase source coverage or revise the scenario design"
        )
    limit = int(config.get("critical_pool_limit", 0))
    if limit > 0:
        snapshots = sorted(snapshots, key=lambda item: item.source_criticality,
                           reverse=True)[:limit]
    return source_values, snapshots, source_episodes, sum(len(items) for items in transitions)


def _asd(alpha_history: list[np.ndarray], window: int) -> float:
    if window < 1 or len(alpha_history) <= window:
        return float("nan")
    values = np.asarray(alpha_history, dtype=float)
    differences = values[-window:] - values[-window - 1:-1]
    return float(np.abs(differences.sum(axis=0)).mean())


def _proposal_action(
    behavior: str,
    learner: SparseExpectedTD,
    state: StateKey,
    mixed_q: np.ndarray,
    phi: np.ndarray,
    exploration: float,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, str]:
    if behavior == "uniform":
        proposal = np.full(phi.size, 1.0 / phi.size)
        return int(rng.choice(phi.size, p=proposal)), proposal, "uniform"
    if behavior == "naturalistic":
        return int(rng.choice(phi.size, p=phi)), phi.copy(), "phi"
    if behavior == "adaptive":
        action, scores = select_gap_action(learner.action_values(state),
                                           mixed_q,
                                           phi,
                                           learner.visits[state],
                                           exploration,
                                           rng=rng)
        proposal = np.zeros(phi.size, dtype=float)
        proposal[action] = 1.0
        return action, scores, "eta_gap_ucb"
    raise ValueError(f"unknown behavior policy: {behavior}")


def _run_variant(
    name: str,
    behavior: str,
    optimize_alpha: bool,
    target: str,
    snapshots: list[SnapshotSpec],
    source_values: list[dict],
    config: dict,
    encoder: StateEncoder,
    phi: np.ndarray,
    start_seed: int,
    behavior_seed: int,
) -> VariantResult:
    actions = phi.size
    budget = int(config["adaptation_episodes"])
    horizon = int(config["horizon"])
    learner = SparseExpectedTD(actions,
                               phi,
                               learning_rate=float(config["learning_rate"]),
                               gamma=float(config["gamma"]))
    alpha = uniform_alpha(len(source_values))
    start_rng = np.random.default_rng(start_seed)
    behavior_rng = np.random.default_rng(behavior_seed)
    rows_alpha: list[dict] = []
    rows_transition: list[dict] = []
    rows_q: list[dict] = []
    rows_propensity: list[dict] = []
    visited: dict[tuple[StateKey, int], np.ndarray] = {}
    alpha_history: list[np.ndarray] = []
    restore_steps = 0
    stopped = False
    window = int(config.get("asd_window", 10))
    threshold = float(config.get("asd_threshold", 0.02))
    min_episodes = int(config.get("min_adaptation_episodes", window + 1))
    update_every = int(config.get("qp_update_every", 1))
    for episode in range(1, budget + 1):
        spec = snapshots[int(start_rng.integers(len(snapshots)))]
        env = DenseCutInEnv(get_profile(target), spec.scenario)
        diagnostic: QPDiagnostics | None = None
        try:
            env.reset(seed=spec.seed)
            env.restore(DenseSnapshot(spec.scenario, spec.seed, spec.action_history))
            restore_steps += len(spec.action_history)
            for time_step in range(horizon):
                if env._is_terminated() or env._is_truncated():
                    break
                state = encoder.encode(env.state_features())
                matrix = _source_matrix(source_values, state, actions)
                known = matrix is not None
                mixed_q = (matrix @ alpha) if matrix is not None else np.zeros(actions,
                                                                               dtype=float)
                critical = bool(
                    known
                    and float(phi @ mixed_q) > float(config.get("criticality_threshold", 1e-8)))
                action, policy_values, policy_name = _proposal_action(
                    behavior if critical else "naturalistic",
                    learner,
                    state,
                    mixed_q,
                    phi,
                    float(config["exploration"]),
                    behavior_rng,
                )
                _, _, terminal, truncated, _ = env.step(action)
                next_state = None if terminal or truncated else encoder.encode(
                    env.state_features())
                td_error = learner.update(state, action, float(terminal), next_state,
                                          bool(terminal), critical)
                if critical and matrix is not None:
                    visited[(state, action)] = matrix[action].copy()
                rows_transition.append({
                    "method_variant":
                    name,
                    "episode":
                    episode,
                    "time_step":
                    time_step,
                    "snapshot_id":
                    spec.snapshot_id,
                    "scenario_id":
                    _scenario_id(spec.scenario),
                    "restore_prefix_actions":
                    len(spec.action_history),
                    "state":
                    json.dumps(state),
                    "action":
                    action,
                    "reward":
                    float(terminal),
                    "terminal":
                    bool(terminal),
                    "critical":
                    critical,
                    "source_q_known":
                    known,
                    "td_error":
                    td_error,
                    "actual_npc_acceleration":
                    float(env._cutin_vehicle.action["acceleration"]),
                })
                rows_q.append({
                    "method_variant": name,
                    "episode": episode,
                    "time_step": time_step,
                    "state": json.dumps(state),
                    "q_target": json.dumps(learner.action_values(state).tolist()),
                    "q_mixed": json.dumps(mixed_q.tolist()),
                    "gap_scores": json.dumps(np.asarray(policy_values).tolist()),
                    "visits": json.dumps(learner.visits[state].tolist()),
                    "critical": critical,
                })
                probability = float(1.0 / actions) if policy_name == "uniform" else (
                    float(phi[action]) if policy_name == "phi" else 1.0)
                rows_propensity.append({
                    "method_variant": name,
                    "episode": episode,
                    "time_step": time_step,
                    "action": action,
                    "phi_probability": float(phi[action]),
                    "behavior_probability": probability,
                    "policy_definition": policy_name,
                    "critical": critical,
                })
                if terminal or truncated:
                    break
        finally:
            env.close()
        if optimize_alpha and visited and episode % update_every == 0:
            keys = list(visited)
            design = np.vstack([visited[key] for key in keys])
            labels = np.asarray([learner.action_values(state)[action] for state, action in keys])
            diagnostic = simplex_least_squares(design, labels, alpha)
            alpha = diagnostic.alpha
        alpha_history.append(alpha.copy())
        current_asd = _asd(alpha_history, window)
        converged = bool(
            np.isfinite(current_asd) and current_asd < threshold and episode >= min_episodes)
        rows_alpha.append({
            "method_variant": name,
            "episode": episode,
            **{f"alpha_{index}": float(value)
               for index, value in enumerate(alpha)},
            "qp_status": "frozen_equal" if not optimize_alpha else
            ("not_due_or_no_critical_rows" if diagnostic is None else diagnostic.status),
            "critical_rows": len(visited),
            "asd": current_asd,
            "asd_threshold": threshold,
            "converged": converged,
            "restore_steps_cumulative": restore_steps,
        })
        if bool(config.get("stop_on_asd", False)) and converged:
            stopped = True
            break
    return VariantResult(name, alpha, learner, rows_alpha, rows_transition, rows_q,
                         rows_propensity, stopped, restore_steps)


def _evaluation_policy(strategy: str, matrix: np.ndarray | None, phi: np.ndarray,
                       alpha: np.ndarray | None, epsilon: float) -> np.ndarray:
    if strategy == "NDE-phi-H" or matrix is None:
        return phi.copy()
    if strategy.startswith("Single-SM-"):
        index = int(strategy.rsplit("-", 1)[1])
        alpha = np.eye(matrix.shape[1])[index]
    elif strategy == "Equal-Mixture-H":
        alpha = uniform_alpha(matrix.shape[1])
    assert alpha is not None
    return defensive_mixture(mixture_of_policies(matrix, alpha, phi), phi, epsilon)


def _strategy_alpha(strategy: str, source_count: int,
                    learned: dict[str, VariantResult]) -> np.ndarray | None:
    if strategy in learned:
        return learned[strategy].alpha
    if strategy.startswith("Single-SM-"):
        return np.eye(source_count)[int(strategy.rsplit("-", 1)[1])]
    if strategy == "Equal-Mixture-H":
        return uniform_alpha(source_count)
    return None


def _initial_state_proposal(
    strategy: str,
    target: str,
    scenarios: list[CutInScenario],
    reference: np.ndarray,
    source_values: list[dict],
    learned: dict[str, VariantResult],
    encoder: StateEncoder,
    phi: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Bias initial scenarios by surrogate criticality and preserve p0 support."""
    alpha = _strategy_alpha(strategy, len(source_values), learned)
    if alpha is None:
        return reference.copy()
    criticality = np.zeros(len(scenarios), dtype=float)
    for index, scenario in enumerate(scenarios):
        env = DenseCutInEnv(get_profile(target), scenario)
        try:
            env.reset(seed=0)
            state = encoder.encode(env.state_features())
        finally:
            env.close()
        matrix = _source_matrix(source_values, state, phi.size)
        if matrix is not None:
            criticality[index] = float(phi @ (matrix @ alpha))
    weighted = reference * criticality
    adaptive = weighted / weighted.sum() if weighted.sum() > 0.0 else reference
    return defensive_mixture(adaptive, reference, epsilon)


def _evaluate(
    target: str,
    scenarios: list[CutInScenario],
    source_values: list[dict],
    learned: dict[str, VariantResult],
    config: dict,
    encoder: StateEncoder,
    phi: np.ndarray,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    action_epsilon = float(config.get("action_support_epsilon", config["support_epsilon"]))
    initial_epsilon = float(config.get("initial_support_epsilon", config["support_epsilon"]))
    scenario_probabilities = _scenario_probabilities(config, scenarios)
    strategies = [
        "NDE-phi-H", *[f"Single-SM-{index}" for index in range(len(source_values))],
        "Equal-Mixture-H"
    ]
    strategies.extend(result.name for result in learned.values()
                      if not np.allclose(result.alpha, uniform_alpha(len(source_values))))
    strategies = list(dict.fromkeys(strategies))
    for strategy_index, strategy in enumerate(strategies):
        alpha = learned[strategy].alpha if strategy in learned else None
        scenario_proposal = _initial_state_proposal(
            strategy,
            target,
            scenarios,
            scenario_probabilities,
            source_values,
            learned,
            encoder,
            phi,
            initial_epsilon,
        )
        for draw in range(int(config["evaluation_draws"])):
            rng = np.random.default_rng(seed + 100000 * strategy_index + draw)
            scenario_index = int(rng.choice(len(scenarios), p=scenario_proposal))
            scenario = scenarios[scenario_index]
            env = DenseCutInEnv(get_profile(target), scenario)
            log_terms = [
                log_importance_weight(
                    np.asarray([scenario_probabilities[scenario_index]]),
                    np.asarray([scenario_proposal[scenario_index]]),
                )
            ]
            path: list[list[float]] = []
            actions: list[int] = []
            known_all = True
            known_steps = 0
            try:
                env.reset(seed=seed + 100000 * strategy_index + draw)
                for _ in range(int(config["horizon"])):
                    state = encoder.encode(env.state_features())
                    matrix = _source_matrix(source_values, state, phi.size)
                    known_all &= matrix is not None
                    known_steps += int(matrix is not None)
                    proposal = _evaluation_policy(strategy, matrix, phi, alpha, action_epsilon)
                    action = int(rng.choice(phi.size, p=proposal))
                    actions.append(action)
                    path.append(proposal.tolist())
                    log_terms.append(
                        log_importance_weight(np.asarray([phi[action]]),
                                              np.asarray([proposal[action]])))
                    _, _, terminal, truncated, _ = env.step(action)
                    if terminal or truncated:
                        break
                rows.append({
                    "strategy": strategy,
                    "draw": draw,
                    "scenario_id": _scenario_id(scenario),
                    "scenario_mode": scenario.mode,
                    "scenario_probability": float(scenario_probabilities[scenario_index]),
                    "scenario_proposal_probability": float(scenario_proposal[scenario_index]),
                    "collision": bool(env.episode_result().collision),
                    "log_weight": float(sum(log_terms)),
                    "actions": json.dumps(actions),
                    "proposal_probabilities": json.dumps(path),
                    "source_q_known_all_steps": known_all,
                    "source_q_known_steps": known_steps,
                    "episode_steps": len(actions),
                    "reference_distribution": "declared_scenario_distribution_and_phi",
                })
            finally:
                env.close()
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_source_artifacts(output: Path, source: SourceStage) -> None:
    """Persist source tables and replay starts before any target is revealed."""
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "critical_snapshot_pool.csv",
                [{
                    "snapshot_id": item.snapshot_id,
                    "scenario_id": _scenario_id(item.scenario),
                    "source_profile": item.source_profile,
                    "seed": item.seed,
                    "action_history": json.dumps(item.action_history),
                    "source_state": json.dumps(item.source_state),
                    "source_criticality": item.source_criticality,
                } for item in source.snapshots])
    for index, table in enumerate(source.values):
        states = sorted({state for state, _ in table})
        actions = max((action for _, action in table), default=-1) + 1
        values = np.asarray([[table.get((state, action), np.nan) for action in range(actions)]
                             for state in states],
                            dtype=float)
        np.savez_compressed(
            output / f"source_q_surrogate_{index}.npz",
            keys=np.asarray(states, dtype=int) if states else np.empty((0, 0), dtype=int),
            values=values,
        )
    (output / "source_stage_summary.json").write_text(json.dumps(
        {
            "source_episodes": source.episodes,
            "source_td_transitions": source.transitions,
            "critical_snapshot_pool": len(source.snapshots),
            "kind": "target-hidden_frozen_source_stage",
        },
        indent=2),
                                                      encoding="utf-8")


def _save_q(path: Path, learner: SparseExpectedTD) -> None:
    keys = list(learner.values)
    key_width = len(keys[0]) if keys else 0
    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=int) if keys else np.empty((0, key_width), dtype=int),
        values=np.asarray(list(learner.values.values()), dtype=float) if keys else np.empty(
            (0, learner.actions), dtype=float))


def _save_diagnostic_replay(output: Path, target: str, encoder: StateEncoder, horizon: int,
                            seed: int) -> None:
    from PIL import Image, ImageDraw

    scenario = CutInScenario(30.0, -2.0, "fast_intrusion")
    actions = ([0, 2] * ((horizon + 1) // 2))[:horizon]
    env = DenseCutInEnv(get_profile(target), scenario, render_mode="rgb_array")
    frames = []
    try:
        env.reset(seed=seed)
        for action in actions:
            _, _, terminal, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            if terminal or truncated:
                break
        trace = env.dense_trace()
        np.savez_compressed(output / "diagnostic_replay_trace.npz", **asdict(trace))
        collision = bool(env.episode_result().collision)
    finally:
        env.close()
    if frames:
        images = []
        for frame_index, frame in enumerate(frames):
            image = Image.fromarray(frame).convert("RGB")
            draw = ImageDraw.Draw(image)
            physics_index = min((frame_index + 1) * 4 - 1, len(trace.time) - 1)
            draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
            draw.text(
                (3, 2),
                f"AdaTE DenseRL diagnostic | target={target} | t={trace.time[physics_index]:.2f}s | collision={collision}",
                fill=(255, 255, 255))
            draw.text(
                (3, 14),
                f"residual={trace.residual_action[physics_index]:+.1f} m/s2 | actual NPC acceleration={trace.npc_acceleration[physics_index]:+.2f} m/s2",
                fill=(255, 255, 255))
            images.append(image)
        images[0].save(output / "diagnostic_replay.gif",
                       save_all=True,
                       append_images=images[1:],
                       duration=200,
                       loop=0)
    (output / "diagnostic_replay_metadata.json").write_text(json.dumps(
        {
            "target": target,
            "scenario": asdict(scenario),
            "actions": actions,
            "collision": collision,
            "frame_count": len(frames),
            "kind": "actual_highway_env_diagnostic_replay"
        },
        indent=2),
                                                            encoding="utf-8")


def _run_dense_case(
    config: dict,
    output: Path,
    manifest: Callable[[Path, dict], None],
    source: SourceStage,
    source_stage: str,
    source_shared: bool,
) -> dict:
    """Adapt and evaluate one held-out target from a frozen source stage."""
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    phi = np.asarray(config["natural_policy"], dtype=float)
    if not np.isclose(phi.sum(), 1.0):
        raise ValueError("natural_policy must sum to one")
    encoder = StateEncoder(**config["state_encoder"])
    scenarios = _scenario_pool(config)
    seed = int(config["seed"])
    variants: dict[str, VariantResult] = {}
    start_seed = seed + 10000
    for index, variant in enumerate(config.get("adaptation_variants", [])):
        result = _run_variant(
            str(variant["name"]),
            str(variant["behavior_policy"]),
            bool(variant["optimize_alpha"]),
            str(config["target_profile"]),
            source.snapshots,
            source.values,
            config,
            encoder,
            phi,
            start_seed,
            seed + 20000 * (index + 1),
        )
        variants[result.name] = result
    if not variants:
        raise ValueError("adaptation_variants cannot be empty")
    alpha_rows = [row for result in variants.values() for row in result.alpha_rows]
    transition_rows = [row for result in variants.values() for row in result.transition_rows]
    q_rows = [row for result in variants.values() for row in result.q_rows]
    propensity_rows = [row for result in variants.values() for row in result.propensity_rows]
    _write_rows(output / "mixture_coefficients_trace.csv", alpha_rows)
    _write_rows(output / "adaptation_transitions.jsonl", transition_rows)
    _write_rows(output / "adaptation_propensities.jsonl", propensity_rows)
    _write_rows(output / "q_gap_visitation.csv", q_rows)
    for name, result in variants.items():
        artifact_name = name.lower().removesuffix("-h").replace("adate",
                                                                "adaptive").replace("-", "_")
        _save_q(output / f"learned_q_{artifact_name}.npz", result.learner)
    evaluation = _evaluate(str(config["target_profile"]), scenarios, source.values, variants,
                           config, encoder, phi, seed + 700000)
    _write_rows(output / "evaluation_draws.csv", evaluation)
    summary = {
        strategy: weighted_event_estimate(
            np.asarray([row["collision"] for row in evaluation if row["strategy"] == strategy],
                       dtype=float),
            np.asarray([row["log_weight"] for row in evaluation if row["strategy"] == strategy],
                       dtype=float),
        )
        for strategy in sorted({row["strategy"]
                                for row in evaluation})
    }
    (output / "importance_sampling_summary.json").write_text(json.dumps(summary, indent=2),
                                                             encoding="utf-8")
    support_rows = []
    for strategy in sorted({row["strategy"] for row in evaluation}):
        paths = [
            np.asarray(json.loads(row["proposal_probabilities"]), dtype=float)
            for row in evaluation if row["strategy"] == strategy
        ]
        average = np.concatenate(paths).mean(axis=0) if paths else phi
        for action in range(phi.size):
            support_rows.append({
                "strategy": strategy,
                "action": action,
                "p_phi": phi[action],
                "mean_q": average[action],
                "positive_support": bool(average[action] > 0)
            })
    _write_rows(output / "proposal_support.csv", support_rows)
    _save_diagnostic_replay(output, str(config["target_profile"]), encoder, int(config["horizon"]),
                            seed + 900000)
    costs = {
        "source_episodes":
        source.episodes,
        "source_td_transitions":
        source.transitions,
        "critical_snapshot_pool":
        len(source.snapshots),
        "source_stage":
        source_stage,
        "source_stage_shared":
        source_shared,
        "source_episodes_charged":
        0 if source_shared else source.episodes,
        "run_seed":
        seed,
        "target_profile":
        str(config["target_profile"]),
        "adaptation_target_episodes":
        sum(len(result.alpha_rows) for result in variants.values()),
        "adaptation_restore_prefix_steps":
        sum(result.restore_steps for result in variants.values()),
        "evaluation_draws":
        len(evaluation),
        "elapsed_seconds":
        time.perf_counter() - start,
        "status":
        "mechanism_and_comparative_study; precision claims require the recorded RHW/CI diagnostics",
    }
    (output / "cost_ledger.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
    manifest(output, costs)
    return {
        "costs": costs,
        "summary": summary,
        "alphas": {name: result.alpha.tolist()
                   for name, result in variants.items()}
    }


def run_dense_protocol(config: dict, output: Path, manifest: Callable[[Path, dict], None]) -> dict:
    """Run one auditable, paper-aligned highway-env DenseRL study."""
    phi = np.asarray(config["natural_policy"], dtype=float)
    encoder = StateEncoder(**config["state_encoder"])
    scenarios = _scenario_pool(config)
    source_values, snapshots, episodes, transitions = _collect_source_models(
        list(config["source_profiles"]),
        scenarios,
        config,
        encoder,
        phi,
        int(config["seed"]),
    )
    source = SourceStage(source_values, snapshots, episodes, transitions)
    _write_source_artifacts(output, source)
    return _run_dense_case(config, output, manifest, source, "local", False)


def run_dense_batch(config: dict, output: Path, manifest: Callable[[Path, dict], None]) -> dict:
    """Run a target-held-out, multi-seed DenseRL confirmation study.

    Every source stage is built before its target cases, shared only within the
    same seed, and saved separately.  Target results are therefore independent
    conditional on the frozen source tables and can be aggregated without
    charging source collection once per held-out target.
    """
    output.mkdir(parents=True, exist_ok=True)
    seed_values = config["batch_seeds"] if "batch_seeds" in config else [config["seed"]]
    target_values = config["target_profiles"] if "target_profiles" in config else [
        config["target_profile"]
    ]
    seeds = [int(value) for value in seed_values]
    targets = [str(value) for value in target_values]
    sources = [str(value) for value in config["source_profiles"]]
    if not seeds or not targets or len(set(seeds)) != len(seeds) or len(
            set(targets)) != len(targets):
        raise ValueError("batch_seeds and target_profiles must be non-empty and unique")
    if set(sources) & set(targets):
        raise ValueError("target_profiles must be held out from source_profiles")
    start = time.perf_counter()
    summary_rows: list[dict] = []
    alpha_rows: list[dict] = []
    total_source_episodes = 0
    total_source_transitions = 0
    total_target_episodes = 0
    total_evaluation_draws = 0
    for seed in seeds:
        seed_config = dict(config)
        seed_config["seed"] = seed
        phi = np.asarray(seed_config["natural_policy"], dtype=float)
        encoder = StateEncoder(**seed_config["state_encoder"])
        scenarios = _scenario_pool(seed_config)
        values, snapshots, episodes, transitions = _collect_source_models(
            sources, scenarios, seed_config, encoder, phi, seed)
        source = SourceStage(values, snapshots, episodes, transitions)
        stage_dir = output / f"seed_{seed}" / "source_stage"
        _write_source_artifacts(stage_dir, source)
        total_source_episodes += episodes
        total_source_transitions += transitions
        for target in targets:
            case_config = dict(seed_config)
            case_config["target_profile"] = target
            case_dir = output / f"seed_{seed}" / f"target_{target}"
            result = _run_dense_case(
                case_config,
                case_dir,
                lambda destination, costs, run_seed=seed: manifest(destination, costs),
                source,
                str(stage_dir.relative_to(output)),
                True,
            )
            costs = result["costs"]
            total_target_episodes += int(costs["adaptation_target_episodes"])
            total_evaluation_draws += int(costs["evaluation_draws"])
            for strategy, values in result["summary"].items():
                summary_rows.append({
                    "seed": seed,
                    "target_profile": target,
                    "strategy": strategy,
                    "case_dir": str(case_dir.relative_to(output)),
                    **values
                })
            for variant, alpha in result["alphas"].items():
                alpha_rows.append({
                    "seed": seed,
                    "target_profile": target,
                    "method_variant": variant,
                    **{f"alpha_{index}": value
                       for index, value in enumerate(alpha)}
                })
    _write_rows(output / "case_strategy_summary.csv", summary_rows)
    _write_rows(output / "final_mixture_coefficients.csv", alpha_rows)
    aggregate_rows: list[dict] = []
    for target in targets:
        for strategy in sorted({row["strategy"] for row in summary_rows}):
            rows = [
                row for row in summary_rows
                if row["target_profile"] == target and row["strategy"] == strategy
            ]
            estimates = np.asarray([float(row["estimate"]) for row in rows], dtype=float)
            rhws = np.asarray([float(row["relative_half_width"]) for row in rows], dtype=float)
            aggregate_rows.append({
                "target_profile":
                target,
                "strategy":
                strategy,
                "runs":
                len(rows),
                "mean_estimate":
                float(estimates.mean()),
                "seed_sd_estimate":
                float(estimates.std(ddof=1)) if len(estimates) > 1 else float("nan"),
                "mean_relative_half_width":
                float(rhws.mean()),
                "max_relative_half_width":
                float(rhws.max()),
            })
    _write_rows(output / "cross_seed_summary.csv", aggregate_rows)
    costs = {
        "batch_seeds":
        seeds,
        "target_profiles":
        targets,
        "source_profiles":
        sources,
        "unique_source_stages":
        len(seeds),
        "source_episodes":
        total_source_episodes,
        "source_td_transitions":
        total_source_transitions,
        "adaptation_target_episodes":
        total_target_episodes,
        "evaluation_draws":
        total_evaluation_draws,
        "elapsed_seconds":
        time.perf_counter() - start,
        "status":
        "multi_target_multi_seed_confirmation; inspect per-case IS CI/RHW and cross-seed spread separately",
    }
    (output / "cost_ledger.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
    manifest(output, costs)
    return {"costs": costs, "summary_rows": summary_rows, "aggregate_rows": aggregate_rows}
