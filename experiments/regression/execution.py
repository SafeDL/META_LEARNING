"""Metered native executions, with physical caches separated by policy and purpose."""

import hashlib
import time
from dataclasses import asdict, replace

import numpy as np
from controllers.highway.decision_interface import DecisionInterface
from controllers.highway.train import ModelController
from simulator.highway.collision_env import execute
from testing_methods.common.response import ResponseArchive

from experiments.domain_risk import disclose_driving_quality, disclose_impact
from experiments.regression.preflight import POLICIES
from experiments.rss import measurement_contract, trajectory_margin
from experiments.schema import QueryFeedback, Scenario
from experiments.storage import read_json, write_json

_MODELS = {}
ATTENTION = "social_attention_ddqn"


def checkpoint_identity(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_record(checkpoint, scene, destination, *, purpose, public_interface=False, policy="ppo",
                   bank_call=None):
    identity = checkpoint_identity(checkpoint)
    if destination.exists():
        saved = read_json(destination)
        if (saved["checkpoint_sha256"] != identity or saved["scenario"] != asdict(scene)
                or saved["purpose"] != purpose or saved.get("call") != bank_call):
            raise ValueError("Execution cache does not match the requested physical identity")
        return saved
    if identity not in _MODELS:
        if policy == ATTENTION:
            from experiments.public_ads.attention_repair import load_attention

            _MODELS[identity] = load_attention(checkpoint)
        else:
            _MODELS[identity] = POLICIES[policy].load(checkpoint, device="cpu")
    started = time.perf_counter()
    attempt = destination.with_suffix(".attempt.json")
    if attempt.exists():
        raise RuntimeError("An unfinished physical execution requires audit before retrying")
    write_json(attempt, {"checkpoint_sha256": identity, "scenario": asdict(scene),
                         "purpose": purpose, "status": "running"})
    try:
        if policy == ATTENTION:
            from controllers.highway.adapted_agents import AttentionPolicy

            controller = DecisionInterface(AttentionPolicy(_MODELS[identity]))
        else:
            controller = ModelController(_MODELS[identity])
        if public_interface and policy != ATTENTION:
            controller = DecisionInterface(controller)
        execution = execute(controller, scene, environment_seed=81001)
    except Exception as error:
        write_json(attempt, {"checkpoint_sha256": identity, "scenario": asdict(scene),
                             "purpose": purpose, "status": "failed", "error": repr(error),
                             "wall_s": time.perf_counter() - started})
        raise
    if bank_call is None:
        execution = disclose_driving_quality(disclose_impact(execution))
        margin = trajectory_margin(execution.trajectory)["minimum_margin"]
        feedback = replace(execution.feedback, safety=replace(execution.feedback.safety, rss_margin=margin),
                           rss_contract=measurement_contract())
    else:
        # E01 receives the original native feedback and derives RSS on each paid reveal.
        feedback = execution.feedback
    ego = [next(a for a in f["actors"] if a["actor_id"] == "ego")
           for f in execution.trajectory]
    saved = {"scenario": asdict(scene), "checkpoint_sha256": identity,
             "purpose": purpose, "environment_seed": 81001,
             "feedback": feedback.to_dict(), "trajectory": execution.trajectory,
             "progress_m": ego[-1]["x"] - ego[0]["x"],
             "wall_s": time.perf_counter() - started, "physical_executions": 1}
    if bank_call is not None:
        from experiments.execution_events import analyze_events

        saved.update(call=bank_call, events=analyze_events(scene, execution.trajectory))
    write_json(destination, saved)
    attempt.unlink()
    return saved


def response_archive(name, records, policy="ppo"):
    feedback = [QueryFeedback.from_dict(r["feedback"]) for r in records]
    identities = {r["checkpoint_sha256"] for r in records}
    if len(identities) != 1:
        raise ValueError("One response source must have exactly one executed policy identity")
    return ResponseArchive(
        name, "learning", {"ppo": "ppo", ATTENTION: ATTENTION}.get(policy, "dqn"),
        "history", [Scenario(**r["scenario"]) for r in records],
        np.array([r.collision for r in feedback], dtype=bool),
        np.array([[r.safety.min_clearance_m, r.safety.min_ttc_s] for r in feedback]),
        provenance={"checkpoint_sha256": next(iter(identities)),
                    "physical_executions": len(records)},
        rss_margin=np.array([r.safety.rss_margin for r in feedback]),
        rss_contract=measurement_contract(),
    )
