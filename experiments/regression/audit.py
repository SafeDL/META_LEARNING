"""Independent verification of native executions and policy continuation."""

from dataclasses import asdict
from math import isclose

import numpy as np

from experiments.rss import measurement_contract, trajectory_margin
from experiments.regression.protocol import DQN_SEED, DQN_WARMUP, TRAINING_STEPS
from experiments.schema import QueryFeedback
from experiments.storage import read_json


def verify_execution(row, scene, identity):
    if (row["scenario"] != scene or row["checkpoint_sha256"] != identity
            or row["environment_seed"] != 81001 or row["physical_executions"] != 1):
        raise ValueError("Executed record differs from its scenario or policy identity")
    feedback = QueryFeedback.from_dict(row["feedback"])
    trace = row["trajectory"]
    if (feedback.scenario_id != scene["scenario_id"] or feedback.rss_contract != measurement_contract()
            or feedback.collision != any(frame["collision"] for frame in trace)):
        raise ValueError("Measured feedback differs from the actual paid trajectory")
    times = np.array([frame["time_s"] for frame in trace])
    if times[0] != 0 or not np.allclose(np.diff(times), 0.05, atol=1e-12):
        raise ValueError("Native physical trajectory does not follow the registered 20 Hz clock")
    error = abs(feedback.safety.rss_margin - trajectory_margin(trace)["minimum_margin"])
    if error > 1e-12:
        raise ValueError("RSS feedback cannot be independently reproduced")
    first = next(actor for actor in trace[0]["actors"] if actor["actor_id"] == "ego")
    last = next(actor for actor in trace[-1]["actors"] if actor["actor_id"] == "ego")
    if abs(row["progress_m"] - (last["x"] - first["x"])) > 1e-12:
        raise ValueError("Driving progress differs from the native executed trajectory")
    return error


def verify_repair(report, current, historical, parent_sha, *, policy="ppo"):
    spec = report["specification"]
    if policy != "ppo":
        continuation = report["continuation"]
        warmup = DQN_WARMUP if spec["preparation"] else 0
        updates = (TRAINING_STEPS - warmup) // (4 * 8)
        if (continuation["warmup"] != warmup
                or not isclose(continuation["final_epsilon"], 0.05, rel_tol=0, abs_tol=1e-12)
                or report["gradient_updates"] != updates or report["optimization_epochs"] != updates
                or continuation["target_counter_end"] - continuation["target_counter_start"] != TRAINING_STEPS // 8):
            raise ValueError("DQN repair differs from the registered warmup, exploration or update budget")
    elif report["optimization_epochs"] != 160 or report["gradient_updates"] != 5120:
        raise ValueError("PPO repair differs from the registered update budget")
    if (report["status"] != "complete" or spec["parent_sha256"] != parent_sha
            or spec["current"] != [asdict(s) for s in current]
            or spec["historical"] != [asdict(s) for s in historical]
            or report["actual_environment_steps"] != TRAINING_STEPS):
        raise ValueError("Policy repair differs from its actual parent, paid resets or fixed update budget")
    if sum(sum(s["transitions"].values()) for s in report["curriculum"]) != TRAINING_STEPS:
        raise ValueError("Recorded curriculum does not account for every training transition")
    permitted = {s.scenario_id for s in [*current, *historical]}
    if any(not set(worker["reset_points"]).issubset(permitted) for worker in report["curriculum"]):
        raise ValueError("Repair curriculum used an unregistered failure reset")
    capability = report["capability"]
    if capability["qualified"] != all(c["passed"] for c in capability["courses"]):
        raise ValueError("Capability status is inconsistent with the recorded courses")


def verify_capability_evidence(report, path):
    from types import SimpleNamespace

    from experiments.qualification import COURSES, course_contract, course_result

    evidence = read_json(path)
    accepted = evidence["course_evidence.json"]
    if accepted["contract"] != course_contract():
        raise ValueError("Repair capability evidence uses different acceptance courses")
    rows = []
    for course in COURSES:
        saved = evidence[f"{course.name}.json.gz"]
        trace = saved["trajectory"]
        times = np.array([frame["time_s"] for frame in trace])
        if times[0] != 0 or not np.allclose(np.diff(times), 0.05, atol=1e-12):
            raise ValueError("Capability evidence does not follow the native 20 Hz clock")
        states = [next(actor for actor in frame["actors"] if actor["actor_id"] == "ego")
                  for frame in trace]
        departed = False
        for state in states:
            half_width = (abs(np.sin(state["heading"])) * state["length"]
                          + abs(np.cos(state["heading"])) * state["width"]) / 2
            departed |= (state["y"] - half_width < min(course.lane_centers) - 2
                         or state["y"] + half_width > max(course.lane_centers) + 2)
        last = states[-1]
        world = SimpleNamespace(
            ego=SimpleNamespace(crashed=any(frame["collision"] for frame in trace),
                                speed=last["speed"], position=(last["x"], last["y"])),
            trace=trace, violations={"road_departure"} if departed else set(),
            tick=len(trace) - 1, decisions=(len(trace) + 2) // 4,
        )
        result = course_result(world, course)
        if result != saved["result"]:
            raise ValueError("Capability result differs from the saved driving trajectory")
        rows.append(result)
    if (report["capability"]["courses"] != rows or accepted["courses"] != rows
            or accepted["qualified"] != all(row["passed"] for row in rows)):
        raise ValueError("Reported capability differs from its actual acceptance evidence")


def verify_optimizer_continuation(parent_path, checkpoint, *, policy="ppo", state_path=None,
                                  report=None, parent_report=None):
    import torch
    from stable_baselines3.common.type_aliases import TrainFrequencyUnit
    from experiments.regression.execution import checkpoint_identity
    from experiments.regression.preflight import POLICIES

    parent = POLICIES[policy].load(parent_path, device="cpu")
    current = POLICIES[policy].load(checkpoint, device="cpu")
    updates = 5120
    if policy != "ppo":
        updates = report["gradient_updates"]
        trace = report["continuation"]
        saved_rng = torch.load(state_path, map_location="cpu", weights_only=False)
        replay_path = state_path.with_suffix(".pkl")
        if checkpoint_identity(replay_path) != trace["buffer_sha256"]:
            raise ValueError("Saved DQN replay buffer differs from its recorded identity")
        current.load_replay_buffer(replay_path)
        buffer = current.replay_buffer
        if trace["warmup"]:
            from gymnasium.spaces import Discrete

            actions = Discrete(current.action_space.n, seed=DQN_SEED)
            count = trace["warmup"]
            expected_actions = [actions.sample() for _ in range(count)]
            actual_actions = buffer.actions[:count // buffer.n_envs].reshape(-1)
            if current.seed != DQN_SEED or not np.array_equal(actual_actions, expected_actions):
                raise ValueError("DQN warmup actions differ from the registered seeded random sequence")
        previous = parent_report["continuation"]["buffer_end"] if parent_report else [0, False]
        position = previous[0] + TRAINING_STEPS // 8
        expected = [position % buffer.buffer_size, previous[1] or position >= buffer.buffer_size]
        if (trace["buffer_start"] != previous or trace["buffer_end"] != expected
                or [buffer.pos, buffer.full] != expected or buffer.n_envs != 8
                or buffer.buffer_size != 200000 // 8 or current._n_updates - parent._n_updates != updates
                or current.learning_rate != 0.00005 or current.gamma != 0.99
                or current.batch_size != 256 or current.gradient_steps != 1
                or current.train_freq != (4, TrainFrequencyUnit.STEP) or current.target_update_interval != 10000
                or current.learning_starts != parent.num_timesteps + trace["warmup"]
                or trace["initial_epsilon"] != parent.exploration_rate
                or current.exploration_rate != trace["final_epsilon"]
                or [parent._n_calls, current._n_calls] != [trace["target_counter_start"], trace["target_counter_end"]]
                or not {"python", "numpy", "torch", "environments", "actions"} <= saved_rng.keys()
                or saved_rng["environments"] != [s["rng"] for s in report["curriculum"]]):
            raise ValueError("DQN replay, exploration, synchronization or RNG continuation failed")
    previous_state = parent.policy.optimizer.state_dict()["state"]
    current_state = current.policy.optimizer.state_dict()["state"]
    if previous_state.keys() != current_state.keys() or not all(torch.isfinite(p).all() for p in current.policy.parameters()):
        raise ValueError("Policy state inheritance or numerical validity failed")
    for key, value in current_state.items():
        if float(value["step"]) - float(previous_state[key]["step"]) != updates:
            raise ValueError("Adam state does not show the registered inherited gradient updates")
    if current.num_timesteps - parent.num_timesteps != TRAINING_STEPS:
        raise ValueError("Saved timesteps do not show the registered continuation")
