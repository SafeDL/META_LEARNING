"""Read the accepted braking design without collecting or changing executions."""

from dataclasses import asdict

from controllers.highway.catalog import FORMAL_SUTS
from controllers.highway.contracts import runtime_contract
from simulator.highway.timing import timing_contract

from experiments.formal_scenarios import BRAKING_KIND, DEVELOPMENT_ROOT, FREE_BOUNDS
from experiments.schema import Scenario
from experiments.storage import read_json


def verify_development(destination=DEVELOPMENT_ROOT):
    protocol = read_json(destination / "protocol.json")
    report = read_json(destination / "summary.json")
    scenes = [Scenario(**row) for row in protocol["development_scenes"]]
    audit = report.get("independent_archive_check", {})
    if (
        protocol["bounds"] != {name: list(bounds) for name, bounds in FREE_BOUNDS[BRAKING_KIND].items()}
        or protocol["timing"] != timing_contract()
        or protocol["event_time_s"] != 0
        or protocol["brake_ramp_s"] != 0
        or protocol["duration_s"] != 14
        or len(scenes) != 128
        or len({scene.coordinate() for scene in scenes}) != 128
        or any(scene.kind != BRAKING_KIND for scene in scenes)
        or report["status"] != "development_coverage_passed"
        or not report["native_records_valid"]
        or not report["development_design_passed"]
        or report["development_count_per_ads"] != 128
        or report["ads_executions"] != 2048
        or not audit.get("passed")
        or audit.get("ads_records") != 2048
    ):
        raise ValueError("Accepted braking development evidence does not match the current scene")
    rows = {row["sut_id"]: row for row in report["rows"]}
    if len(report["rows"]) != 16 or set(rows) != {sut.sut_id for sut in FORMAL_SUTS}:
        raise ValueError("Accepted scene must cover all historical and target ADS configurations")
    for sut in FORMAL_SUTS:
        saved = protocol["executions"][sut.sut_id]
        previous, current = dict(saved["object"]), asdict(sut)
        # Supported scenario names describe the study route, not ADS behaviour.
        previous.pop("supported_scenarios")
        current.pop("supported_scenarios")
        identity = runtime_contract(sut)
        original = saved["runtime_contract"]
        if (
            previous != current
            or original.get("checkpoint_sha256") != identity.get("checkpoint_sha256")
            or any(
                identity["source_files"][name] != digest
                for name, digest in original["source_files"].items()
                if name.startswith("controllers/")
                or name in ("simulator/highway/observation.py", "simulator/highway/timing.py")
            )
        ):
            raise ValueError(f"Accepted scene was developed with another ADS: {sut.sut_id}")
        row = rows[sut.sut_id]
        if (
            row["records"] != 128
            or not 0 < row["collisions"] < 128
            or row["avoidable_collisions"] < 1
            or not row["representative_collision_avoidable"]
            or [r["scenario_id"] for r in row["rows"]] != [s.scenario_id for s in scenes]
            or any(not (destination / sut.sut_id / f"{i:03d}.json.gz").is_file() for i in range(128))
        ):
            raise ValueError(f"Incomplete accepted development records: {sut.sut_id}")
    certificate = report["joint_reference_certificate"]
    payload = read_json(destination / certificate["path"])
    reference = payload["summary"]
    if (
        payload["scenario"] != asdict(scenes[40])
        or reference != certificate["summary"]
        or not reference["passed"]
        or reference["collision"]
        or reference["minimum_body_clearance_m"] < 0.1
        or reference["minimum_road_margin_m"] < 0.05 - 1e-9
        or reference["end_time_s"] != 14
        or not certificate["all_16_ads_collide_at_same_coordinate"]
    ):
        raise ValueError("The same-coordinate legal avoidance certificate is incomplete")
    return report


if __name__ == "__main__":
    result = verify_development()
    print({key: result[key] for key in ("status", "ads_executions", "development_design_passed")})
