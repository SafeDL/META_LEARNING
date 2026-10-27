"""History-only verification of final construction data and the paired interaction physics."""

import gzip
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from controllers.highway.catalog import SUTS
from simulator.highway.interaction_env import GUARD_FRONT_GAP_M, GUARD_REAR_GAP_M
from simulator.highway.observation import VehicleState

from .audit_collision_data import verify_archive
from .construction import load_construction
from .discovery_config import write_json
from .domain_risk import disclose_driving_quality, disclose_impact
from .execution_events import analyze_events
from .executions import ExecutionBank
from .independent_preparation import OUTPUT
from .protocol import TEST_RECORDS_ROOT
from .scenarios import DOMAINS


def rectangles_overlap(a, b):
    delta = np.asarray([b["x"] - a["x"], b["y"] - a["y"]])
    forwards = [np.asarray([math.cos(v["heading"]), math.sin(v["heading"])]) for v in (a, b)]
    lateral = [np.asarray([-v[1], v[0]]) for v in forwards]
    for axis in (*forwards, *lateral):
        radius = sum(
            abs(axis @ forward) * vehicle["length"] / 2
            + abs(axis @ lateral_axis) * vehicle["width"] / 2
            for vehicle, forward, lateral_axis in zip((a, b), forwards, lateral)
        )
        if abs(axis @ delta) > radius + 1e-9:
            return False
    return True


def verify_interaction(scenario, trajectory):
    p, c = scenario.parameters, scenario.categories
    initial = {a["actor_id"]: a for a in trajectory[0]["actors"]}
    restricted = c["adjacent_space"] == "restricted"
    expected = {"ego", "cut_in", "guard_front", "guard_rear"} if restricted else {"ego", "cut_in"}
    if set(initial) != expected:
        raise ValueError("The interaction condition has a different initial actor set")
    speed = p["ego_speed_mps"]
    origin = 0 if c["side"] == "left" else 8
    positions = {
        "ego": (100, 4, speed),
        "cut_in": (105 + p["gap_m"], origin, speed - p["closing_speed_mps"]),
    }
    if restricted:
        opposite = 8 - origin
        positions.update(
            guard_front=(105 + GUARD_FRONT_GAP_M, opposite, speed),
            guard_rear=(95 - GUARD_REAR_GAP_M, opposite, speed),
        )
    for name, coordinates in positions.items():
        np.testing.assert_allclose(
            [initial[name][k] for k in ("x", "y", "speed")], coordinates, atol=1e-10
        )
    if any(rectangles_overlap(a, b) for a, b in itertools.combinations(initial.values(), 2)):
        raise ValueError("The interaction starts with overlapping vehicles")
    for frame in trajectory:
        if frame["collision"]:
            break
        t = frame["time_s"]
        actors = {a["actor_id"]: a for a in frame["actors"]}
        for name, (x, y, longitudinal) in positions.items():
            if name == "ego":
                continue
            a = actors[name]
            lateral = 0.0
            if name == "cut_in":
                s = np.clip((t - p["event_time_s"]) / p["lateral_duration_s"], 0, 1)
                y += (4 - origin) * (10 * s**3 - 15 * s**4 + 6 * s**5)
                lateral = (4 - origin) * 30 * s**2 * (1 - s) ** 2 / p["lateral_duration_s"]
            np.testing.assert_allclose(
                [a["x"], a["y"], a["speed"], a["heading"]],
                [
                    x + longitudinal * t,
                    y,
                    math.hypot(longitudinal, lateral),
                    math.atan2(lateral, max(longitudinal, 1e-8)),
                ],
                atol=1e-8,
                rtol=0,
            )
        backgrounds = [v for name, v in actors.items() if name != "ego"]
        if any(rectangles_overlap(a, b) for a, b in itertools.combinations(backgrounds, 2)):
            raise ValueError("Background actors collided before the ego collision")
    return True


def verify_trace(scenario, execution, saved_events=None):
    frames = execution.trajectory
    disclosed = disclose_driving_quality(disclose_impact(execution))
    if saved_events is not None and saved_events != analyze_events(scenario, frames):
        raise ValueError("Saved scene events differ from the paid trajectory")
    if scenario.kind == "cut_in_interaction":
        verify_interaction(scenario, frames)
    if execution.feedback.collision:
        frame = next(f for f in frames if f["collision"])
        states = [VehicleState(**a) for a in frame["actors"]]
        ego = states[0]
        closest = min((ego.polygon().distance(v.polygon()), v.actor_id) for v in states[1:])[1]
        if frame["collision_partner"] != closest:
            raise ValueError(
                "The recorded first collision object differs from the simulator's nearest partner"
            )
    elif frames[-1]["time_s"] > 1 and disclosed.feedback.safety.driving_quality_score is None:
        raise ValueError("An executed safe trajectory lacks its declared driving-quality score")
    return disclosed


def audit_history(source_id, role, kind):
    deep = verify_archive(TEST_RECORDS_ROOT / role / kind / f"{source_id}.npz")
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical" / role / kind / source_id)
    for index, scene in enumerate(bank.candidates):
        with gzip.open(
            bank.root / "executions" / f"{index:06d}.json.gz", "rt", encoding="utf-8"
        ) as stream:
            payload = json.load(stream)
        verify_trace(scene, bank(scene), payload["events"])
    return {**deep, "events_physics_partner_and_quality": "passed"}


def main():
    for kind in DOMAINS:
        load_construction(kind)
    jobs = [
        (s.sut_id, s.role, kind)
        for s in SUTS
        if s.role in ("history", "development")
        for kind in DOMAINS
    ]
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(audit_history, *job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            print("independent_history_audit", len(rows), "/ 60", flush=True)
    for role in ("history", "development"):
        for kind in DOMAINS:
            if (
                len(
                    {
                        r["initial_worlds_sha256"]
                        for r in rows
                        if r["role"] == role and r["scenario"] == kind
                    }
                )
                != 1
            ):
                raise ValueError(
                    "Same-coordinate construction systems saw different initial worlds"
                )
    report = {
        "status": "passed",
        "banks": rows,
        "checked_executions": sum(r["responses"] for r in rows),
        "new_interaction_physical_calls": sum(
            r["actual_physical_calls"] for r in rows if r["scenario"] == "cut_in_interaction"
        ),
        "new_physical_calls_during_audit": 0,
        "independent_targets_read": False,
        "wall_s": time.perf_counter() - started,
    }
    write_json(OUTPUT / "construction_data_audit.json", report)
    return report


if __name__ == "__main__":
    main()
