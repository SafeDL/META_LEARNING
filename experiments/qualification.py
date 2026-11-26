"""Independent capability courses, evaluated before response-bank construction."""

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from controllers.highway.catalog import CURRENT_SUTS
from controllers.highway.contracts import runtime_contract
from controllers.highway.control import Control
from controllers.highway.factory import create
from simulator.highway.collision_env import CollisionWorld
from simulator.highway.timing import PHYSICS_HZ, PHYSICS_PER_DECISION, timing_contract

from .protocol import ROOT
from .schema import Scenario

QUALIFICATION_ROOT = ROOT / "results/k_shot_adaptation/qualification"
ENVIRONMENT_SEED = 21001


@dataclass(frozen=True)
class Course:
    name: str
    scenario: Scenario
    duration_s: float
    lane_centers: tuple[float, ...]
    max_final_speed_mps: float = 40
    min_progress_m: float = 0
    lane_change: bool = False
    initial_rear_guard: bool = False


def brake_scene(name, *, deceleration=4, event_time=2, gap=65):
    return Scenario(
        f"qualification:{name}",
        "front_brake",
        {
            "ego_speed_mps": 25,
            "gap_m": gap,
            "closing_speed_mps": 0,
            "deceleration_mps2": deceleration,
            "event_time_s": event_time,
            "brake_ramp_s": 1,
        },
    )


COURSES = (
    Course("normal_following", brake_scene("normal", event_time=3), 3, (4.0,), min_progress_m=60),
    Course("avoidable_braking", brake_scene("braking", gap=50, event_time=1), 10, (4.0,)),
    Course("full_stop", brake_scene("stop"), 18, (4.0,), max_final_speed_mps=0.5),
    Course(
        "legal_lane_change",
        Scenario(
            "qualification:lane_change",
            "lane_change_rear",
            {
                "ego_speed_mps": 25,
                "lead_gap_m": 35,
                "lead_speed_mps": 12,
                "rear_gap_m": 50,
                "rear_speed_mps": 22,
                "adjacent_front_gap_m": 90,
            },
        ),
        8,
        (0.0, 4.0),
        min_progress_m=100,
        lane_change=True,
    ),
    Course(
        "rear_constraint",
        Scenario(
            "qualification:rear_guard",
            "lane_change_rear",
            {
                "ego_speed_mps": 25,
                "lead_gap_m": 35,
                "lead_speed_mps": 20,
                "rear_gap_m": 5,
                "rear_speed_mps": 38,
                "adjacent_front_gap_m": 15,
            },
        ),
        2,
        (0.0, 4.0),
        initial_rear_guard=True,
    ),
)


# Checkpoint selection uses different coordinates from the acceptance courses.
DEVELOPMENT_COURSES = tuple(
    replace(
        course,
        scenario=replace(
            course.scenario,
            scenario_id=f"policy_development:{course.name}",
            parameters={
                **course.scenario.parameters,
                **(
                    {"gap_m": 60, "deceleration_mps2": 4.5}
                    if course.scenario.kind == "front_brake"
                    else {"lead_gap_m": 40, "adjacent_front_gap_m": 80}
                ),
            },
        ),
    )
    for course in COURSES
)


def course_contract(courses=None):
    return {
        "courses": json.loads(
            json.dumps([asdict(c) for c in (COURSES if courses is None else courses)])
        ),
        "environment_seed": ENVIRONMENT_SEED,
        "timing": timing_contract(),
        "criteria_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def course_result(world, course):
    ego = world.ego
    progress = float(ego.position[0] - 100)
    passed = (
        not ego.crashed
        and not world.violations
        and world.tick == round(course.duration_s * PHYSICS_HZ)
        and world.decisions * PHYSICS_PER_DECISION == world.tick
        and ego.speed <= course.max_final_speed_mps
        and progress >= course.min_progress_m
    )
    if course.lane_change:
        passed = passed and any(
            row["actors"][0]["y"] < 1 and abs(row["actors"][0]["heading"]) < 0.1
            for row in world.trace
        )
    if course.initial_rear_guard:
        imposed_braking = []
        for frame in world.trace:
            states = {s["actor_id"]: s for s in frame["actors"]}
            e, rear = states["ego"], states["rear"]
            if abs(e["y"] - rear["y"]) < 2.2 and e["x"] > rear["x"]:
                gap = e["x"] - rear["x"] - 5
                closing = max(0, rear["speed"] - e["speed"])
                imposed_braking.append(closing**2 / (2 * max(gap, 0.1)))
        passed = passed and max(imposed_braking, default=0) <= 3
    return {
        "course": course.name,
        "passed": bool(passed),
        "collision": bool(ego.crashed),
        "violations": sorted(world.violations),
        "final_speed_mps": float(ego.speed),
        "progress_m": progress,
        "final_y_m": float(ego.position[1]),
        "physics_ticks": world.tick,
        "decisions": world.decisions,
    }


def save_course_evidence(root, rows, *, courses, mode):
    root = Path(root)
    evidence = {
        "contract": course_contract(courses),
        "qualified": all(row["passed"] for row in rows),
        "courses": rows,
        "mode": mode,
        "trajectory_sha256": {
            c.name: hashlib.sha256((root / f"{c.name}.json.gz").read_bytes()).hexdigest()
            for c in courses
        },
    }
    (root / "course_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def verify_course_evidence(root, *, courses=None):
    from .source_registration import source_contract_matches

    courses = COURSES if courses is None else courses
    root = Path(root)
    evidence = json.loads((root / "course_evidence.json").read_text(encoding="utf-8"))
    names = [c.name for c in courses]
    current = course_contract(courses)
    recorded = evidence["contract"]
    contract_ok = recorded == current or (
        {k: v for k, v in recorded.items() if k != "criteria_source_sha256"}
        == {k: v for k, v in current.items() if k != "criteria_source_sha256"}
        and source_contract_matches(
            {"experiments/qualification.py": recorded["criteria_source_sha256"]},
            {"experiments/qualification.py": current["criteria_source_sha256"]},
        )
    )
    if (
        not contract_ok
        or not evidence["qualified"]
        or [r["course"] for r in evidence["courses"]] != names
        or not all(r["passed"] for r in evidence["courses"])
        or set(evidence["trajectory_sha256"]) != set(names)
    ):
        raise ValueError("Capability courses or acceptance criteria changed; revalidation required")
    for name, digest in evidence["trajectory_sha256"].items():
        if hashlib.sha256((root / f"{name}.json.gz").read_bytes()).hexdigest() != digest:
            raise ValueError("Capability trajectory changed after acceptance")
    return evidence


def replay_courses(root, *, courses=None):
    """Certify saved commands in the current native courses, preserving original records."""
    courses = COURSES if courses is None else courses
    root = Path(root)
    rows = []
    for course in courses:
        with gzip.open(root / f"{course.name}.json.gz", "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        trace = payload["trajectory"]
        world = CollisionWorld(
            course.scenario,
            duration_s=course.duration_s,
            lane_centers=course.lane_centers,
            environment_seed=ENVIRONMENT_SEED,
        )
        try:
            if world.trace != trace[:1]:
                raise ValueError(f"Saved capability initial state differs: {course.name}")
            while not world.terminal:
                index = len(world.trace)
                if index >= len(trace):
                    raise ValueError(f"Truncated capability trajectory: {course.name}")
                world.advance(Control(**trace[index]["requested_control"]))
                if world.trace[index:] != trace[index : len(world.trace)]:
                    raise ValueError(
                        f"Native capability replay differs: {course.name}, frame {index}"
                    )
            row = course_result(world, course)
            if (
                len(world.trace) != len(trace)
                or world.event_log != payload["events"]
                or row != payload["result"]
                or not row["passed"]
            ):
                raise ValueError(f"Saved course does not satisfy current acceptance: {course.name}")
            rows.append(row)
        finally:
            world.close()
    return save_course_evidence(root, rows, courses=courses, mode="native_saved_command_replay")


def qualify(policy_factory, *, save_root=None, courses=None):
    courses = COURSES if courses is None else courses
    rows = []
    for course in courses:
        world = CollisionWorld(
            course.scenario,
            duration_s=course.duration_s,
            lane_centers=course.lane_centers,
            environment_seed=ENVIRONMENT_SEED,
        )
        policy = policy_factory()
        while not world.terminal:
            world.advance(policy.act(world.visible_world()))
        row = course_result(world, course)
        rows.append(row)
        if save_root is not None:
            root = Path(save_root)
            root.mkdir(parents=True, exist_ok=True)
            with gzip.open(root / f"{course.name}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(
                    {"trajectory": world.trace, "events": world.event_log, "result": row},
                    stream,
                    allow_nan=False,
                )
        world.close()
    if save_root is not None:
        save_course_evidence(save_root, rows, courses=courses, mode="controller_execution")
    return {"qualified": all(row["passed"] for row in rows), "courses": rows}


def main():
    QUALIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sut in CURRENT_SUTS:
        if sut.family == "learning":
            continue
        root = QUALIFICATION_ROOT / sut.sut_id
        report = {
            "sut_id": sut.sut_id,
            "lineage": sut.lineage,
            "parameters": sut.parameters,
            "runtime_contract": runtime_contract(sut),
            **qualify(lambda: create(sut.sut_id), save_root=root),
        }
        (root / "qualification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append(report)
        print(
            sut.sut_id,
            report["qualified"],
            [r["course"] for r in report["courses"] if not r["passed"]],
            flush=True,
        )
    return rows


if __name__ == "__main__":
    main()
