"""Recompute approved cut-out coverage from native factual/control trajectories."""

from controllers.highway.catalog import FORMAL_HISTORY_SUTS, FORMAL_TARGET_SUTS
from simulator.highway.observation import VehicleState

from experiments.scene_design.audit_interactions import check_dynamics, check_reference, mechanism_event
from experiments.scene_design.design_interactions import auxiliary_executions, inspect
from experiments.executions import execution_contract
from experiments.protocol import ROOT
from experiments.schema import Scenario
from experiments.storage import read_json, write_json
from experiments.scene_design.verify_secondary import DIRECTORY, compare_pair, digest


def audit(directory=DIRECTORY):
    summary = read_json(directory / "secondary_summary.json")
    protocol = read_json(directory / "secondary_protocol.json")
    base = read_json(directory / "extended_protocol.json")
    assert digest(directory / "extended_protocol.json") == protocol["base_protocol_sha256"]
    assert digest(directory / "summary.json") == protocol["original_summary_sha256"]
    assert digest(directory / "audit.json") == protocol["original_audit_sha256"]
    assert digest(ROOT / "experiments/scene_design/verify_secondary.py") == protocol["implementation_sha256"]
    systems = (*FORMAL_TARGET_SUTS, *FORMAL_HISTORY_SUTS) if summary["ads_executions"] == 2048 else FORMAL_TARGET_SUTS
    assert summary["ads_executions"] == len(systems) * 128
    assert {r["sut_id"] for r in summary["rows"]} == {s.sut_id for s in systems}
    frames = 0
    rows = []
    for sut, row in zip(systems, summary["rows"]):
        assert sut.sut_id == row["sut_id"]
        assert execution_contract(sut, kind=base["kind"]) == base["executions"][sut.sut_id]
        collisions, primary, safe = 0, 0, 0
        for ordinal, scene in enumerate(base["scenes"]):
            payload = read_json(directory / sut.sut_id / f"{ordinal:03d}.json.gz")
            assert payload["scenario"] == scene
            collision = any(f["collision"] for f in payload["trajectory"])
            result = payload["summary"]
            assert collision == payload["feedback"]["collision"] == result["collision"]
            mechanism, onset, reveal = mechanism_event(payload)
            assert mechanism == result["mechanism_collision"]
            assert onset == result["event_onset_s"] and reveal == result["obstacle_first_visible_s"]
            assert set(result["violations"]) == set(payload["feedback"]["safety"]["violations"])
            collisions += collision
            primary += mechanism and result["native_valid"] and not result["violations"]
            safe += not collision and not result["violations"] and result["end_time_s"] == 10 and result["native_valid"]
            frames += check_dynamics(payload)
        confirmed = 0
        for check in row["secondary_checks"]:
            ordinal = check["ordinal"]
            factual_path = directory / sut.sut_id / f"{ordinal:03d}.json.gz"
            factual = read_json(factual_path)
            control = read_json(directory / "counterfactual" / f"{sut.sut_id}_{ordinal:03d}.json.gz")
            assert control["factual_sha256"] == digest(factual_path)
            assert control["secondary_protocol_sha256"] == digest(directory / "secondary_protocol.json")
            assert control["execution_contract"] == base["executions"][sut.sut_id]
            assert control["scenario"] == factual["scenario"] and control["environment_seed"] == 81001
            assert control["intervention"] == "remove_obstacle_at_creation"
            assert all({a["actor_id"] for a in f["actors"]} == {"ego", "lead", "guard_rear"} for f in control["trajectory"])
            assert control["feedback"]["collision"] == any(f["collision"] for f in control["trajectory"])
            assert set(control["violations"]) == set(control["feedback"]["safety"]["violations"])
            reconstructed = compare_pair(factual, control)
            assert reconstructed == {k: v for k, v in check.items() if k not in ("ordinal", "reference_mode")}
            frames += check_dynamics(control)
            if check["confirmed"]:
                fact_geometry = inspect(Scenario(**factual["scenario"]), factual["trajectory"], factual["feedback"]["safety"]["violations"])
                assert fact_geometry["native_valid"] and not fact_geometry["violations"]
                assert control["trajectory"][-1]["time_s"] == 10
                for frame in control["trajectory"]:
                    bodies = [VehicleState(**a).polygon() for a in frame["actors"]]
                    assert all(not a.intersects(b) for i, a in enumerate(bodies) for b in bodies[i+1:])
                    assert all(b.bounds[1] >= -2-1e-8 and b.bounds[3] <= 6+1e-8 for b in bodies)
                confirmed += 1
        assert (collisions, primary, safe, confirmed) == (
            row["collisions"], row["direct_obstacle_collisions"], row["safe"], row["verified_secondary_collisions"],
        )
        certificate = row["avoidable_representative"]
        if certificate:
            ordinal, mode = certificate["ordinal"], certificate["mode"]
            factual = read_json(directory / sut.sut_id / f"{ordinal:03d}.json.gz")
            reference = read_json(directory / "reference" / f"{ordinal:03d}_{mode}.json.gz")
            assert reference["scenario"] == factual["scenario"]
            assert all({a["actor_id"] for a in f["actors"]} == {"ego", "lead", "guard_rear", "obstacle"} for f in reference["trajectory"])
            frames += check_reference(reference)
            if certificate["mechanism"] == "direct_obstacle":
                assert mechanism_event(factual)[0]
            else:
                assert certificate["mechanism"] == "verified_secondary"
                assert any(c["ordinal"] == ordinal and c["confirmed"] and c["reference_mode"] == mode for c in row["secondary_checks"])
        assert row["coverage_passed"] == bool(safe and certificate)
        rows.append({k: v for k, v in row.items() if k != "secondary_checks"})
    attempts = list((directory / "counterfactual" / "attempts").glob("*.json"))
    controls = list((directory / "counterfactual").glob("*.json.gz"))
    assert len(attempts) == len(controls) == summary["counterfactual_executions"] <= 8
    assert all(read_json(p)["status"] == "complete" for p in attempts)
    assert auxiliary_executions(directory) == summary["auxiliary_executions"] <= 64
    covered = all(r["coverage_passed"] for r in rows)
    assert (summary["status"] == "development_coverage_passed") == (covered and len(systems) == 16)
    result = {
        "passed": True, "coverage_passed": covered, "ads_records": summary["ads_executions"],
        "physics_frames_checked": frames, "collisions": sum(r["collisions"] for r in rows),
        "direct_obstacle_collisions": sum(r["direct_obstacle_collisions"] for r in rows),
        "verified_secondary_collisions": sum(r["verified_secondary_collisions"] for r in rows),
        "original_protocol_and_negative_results_unchanged": True,
        "counterfactual_executions": len(controls), "auxiliary_executions": auxiliary_executions(directory),
        "new_physical_executions": 0, "rows": rows,
    }
    write_json(directory / "secondary_audit.json", result)
    return result


if __name__ == "__main__":
    print({k: v for k, v in audit().items() if k != "rows"})
