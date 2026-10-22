"""Independent capability courses, evaluated before response-bank construction."""

from dataclasses import dataclass
import gzip
import json
from pathlib import Path

from controllers.highway.catalog import SUTS
from controllers.highway.factory import create
from simulation.highway.collision_env import CollisionWorld

from .protocol import ROOT
from .schema import Scenario


QUALIFICATION_ROOT = ROOT / "results" / "benchmark" / "collision" / "qualification"


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
    return Scenario(f"qualification:{name}", "front_brake",
                    {"ego_speed_mps": 25, "gap_m": gap, "closing_speed_mps": 0,
                     "deceleration_mps2": deceleration, "event_time_s": event_time,
                     "brake_ramp_s": 1})


COURSES = (
    Course("normal_following", brake_scene("normal", event_time=3), 3, (4.0,),
           min_progress_m=60),
    Course("avoidable_braking", brake_scene("braking", gap=50, event_time=1), 10, (4.0,)),
    Course("full_stop", brake_scene("stop"), 18, (4.0,), max_final_speed_mps=0.5),
    Course("legal_lane_change", Scenario("qualification:lane_change", "lane_change_rear",
                                          {"ego_speed_mps": 25, "lead_gap_m": 35,
                                           "lead_speed_mps": 12, "rear_gap_m": 50,
                                           "rear_speed_mps": 22,
                                           "adjacent_front_gap_m": 90}),
           8, (0.0, 4.0), min_progress_m=100, lane_change=True),
    Course("rear_constraint", Scenario("qualification:rear_guard", "lane_change_rear",
                                       {"ego_speed_mps": 25, "lead_gap_m": 35,
                                        "lead_speed_mps": 20, "rear_gap_m": 5,
                                        "rear_speed_mps": 38,
                                        "adjacent_front_gap_m": 15}),
           2, (0.0, 4.0), initial_rear_guard=True),
)


def qualify(policy_factory, *, save_root=None):
    rows = []
    for course in COURSES:
        world = CollisionWorld(course.scenario, duration_s=course.duration_s,
                               lane_centers=course.lane_centers, environment_seed=21001)
        policy = policy_factory()
        while not world.terminal:
            world.advance(policy.act(world.visible_world()))
        ego = world.ego
        progress = float(ego.position[0] - 100)
        passed = (not ego.crashed and not world.violations
                  and ego.speed <= course.max_final_speed_mps
                  and progress >= course.min_progress_m)
        if course.lane_change:
            passed = passed and any(row["actors"][0]["y"] < 1
                                    and abs(row["actors"][0]["heading"]) < 0.1
                                    for row in world.trace)
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
        row = {"course": course.name, "passed": bool(passed),
               "collision": bool(ego.crashed), "violations": sorted(world.violations),
               "final_speed_mps": float(ego.speed), "progress_m": progress,
               "final_y_m": float(ego.position[1]), "physics_ticks": world.tick,
               "decisions": world.decisions}
        rows.append(row)
        if save_root is not None:
            root = Path(save_root)
            root.mkdir(parents=True, exist_ok=True)
            with gzip.open(root / f"{course.name}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump({"trajectory": world.trace, "events": world.event_log,
                           "result": row}, stream, allow_nan=False)
    return {"qualified": all(row["passed"] for row in rows), "courses": rows}


def main():
    QUALIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
    (QUALIFICATION_ROOT / "nonlearning.json").write_text(
        json.dumps({"status": "running", "qualified": False, "objects": []}), encoding="utf-8")
    rows = []
    for sut in SUTS:
        if sut.family == "learning":
            continue
        root = QUALIFICATION_ROOT / sut.sut_id
        report = {"sut_id": sut.sut_id, "lineage": sut.lineage,
                  "parameters": sut.parameters,
                  **qualify(lambda: create(sut.sut_id), save_root=root)}
        (root / "qualification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append(report)
        print(sut.sut_id, report["qualified"],
              [r["course"] for r in report["courses"] if not r["passed"]], flush=True)
    (QUALIFICATION_ROOT / "nonlearning.json").write_text(
        json.dumps({"status": "complete", "qualified": all(r["qualified"] for r in rows),
                    "objects": rows}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
