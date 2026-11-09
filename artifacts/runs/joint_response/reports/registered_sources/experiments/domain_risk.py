"""Evaluator-only scalar disclosures from the execution that a selector paid for."""

import math
from dataclasses import replace


def disclose_impact(execution):
    """Huang et al., Eq. (2): equal-mass delta-V at the first collision tick."""
    frames = [frame for frame in execution.trajectory if frame["collision"]]
    if bool(frames) != execution.feedback.collision:
        raise ValueError("Collision feedback differs from the paid trajectory")
    delta_v = None
    if frames:
        frame = frames[0]
        actors = {actor["actor_id"]: actor for actor in frame["actors"]}
        partner_id = frame["collision_partner"]
        if partner_id is None or partner_id == "ego" or partner_id not in actors:
            raise ValueError("Impact disclosure requires the recorded collision partner")
        ego, partner = actors["ego"], actors[partner_id]
        vx = ego["speed"] * math.cos(ego["heading"]) - partner["speed"] * math.cos(
            partner["heading"]
        )
        vy = ego["speed"] * math.sin(ego["heading"]) - partner["speed"] * math.sin(
            partner["heading"]
        )
        delta_v = math.hypot(vx, vy) / 2
    safety = replace(execution.feedback.safety, impact_delta_v_mps=delta_v)
    return replace(execution, feedback=replace(execution.feedback, safety=safety))


def disclose_driving_quality(execution):
    """Only an executed safe scenario needs the native fuzzing successor score."""
    from testing_methods.baselines.scenario_fuzz.driving_quality import trajectory_score

    score = None
    frames = execution.trajectory
    duration = frames[-1]["time_s"] - frames[0]["time_s"] if frames else 0
    if not execution.feedback.collision and duration > 1:
        score, _ = trajectory_score(execution.trajectory)
    safety = replace(execution.feedback.safety, driving_quality_score=score)
    return replace(execution, feedback=replace(execution.feedback, safety=safety))
