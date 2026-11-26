"""Audit every completed historical response without reading formal target labels."""

import hashlib
import math
from pathlib import Path

import numpy as np

from testing_methods.common.response import archive_fingerprints

from .executions import ExecutionBank
from .history_transfer_experiment import OUTPUT, freeze, scenario_history, scenarios_for
from .record_audit import BankTruth
from .storage import read_json, write_json


def background_profile(bank, truth):
    from simulator.highway.collision_env import default_lane_centers

    profile = {
        "background_road_departures": 0,
        "background_road_departures_without_ego_collision": 0,
        "background_crash_before_ego_collision": 0,
        "minimum_background_road_margin_m": math.inf,
        "time_limit_without_collision_or_ego_departure": 0,
    }
    for index, scene in enumerate(bank.candidates):
        centers = default_lane_centers(scene.kind)
        if scene.kind == "side_encroachment":
            centers = (0.0, 4.0) if scene.categories["side"] == "left" else (4.0, 8.0)
        lower, upper = min(centers) - 2, max(centers) + 2
        execution = bank(scene)
        trace = execution.trajectory
        departed, first_background_crash, first_ego_crash = False, math.inf, math.inf
        for frame in trace:
            if frame["collision"]:
                first_ego_crash = min(first_ego_crash, frame["time_s"])
            if any(not c["scripted"] for c in frame["background_controls"].values()):
                first_background_crash = min(first_background_crash, frame["time_s"])
            for actor in frame["actors"]:
                if actor["actor_id"] == "ego":
                    continue
                half = (
                    abs(math.sin(actor["heading"])) * actor["length"]
                    + abs(math.cos(actor["heading"])) * actor["width"]
                ) / 2
                margin = min(actor["y"] - half - lower, upper - actor["y"] - half)
                profile["minimum_background_road_margin_m"] = min(
                    profile["minimum_background_road_margin_m"], margin
                )
                departed |= margin < -1e-9
        collision = bool(truth.labels[index])
        profile["background_road_departures"] += int(departed)
        profile["background_road_departures_without_ego_collision"] += int(
            departed and not collision
        )
        profile["background_crash_before_ego_collision"] += int(
            first_background_crash < first_ego_crash - 1e-9
        )
        profile["time_limit_without_collision_or_ego_departure"] += int(
            not collision
            and "road_departure" not in execution.feedback.safety.violations
            and trace[-1]["time_s"] >= bank.manifest["contract"]["episode_duration_s"] - 1e-9
        )
    if not math.isfinite(profile["minimum_background_road_margin_m"]):
        raise ValueError("Historical traffic scene has no measured background geometry")
    return profile


def audit(root=OUTPUT):
    root = Path(root)
    directory = root / "reports/history_quality"
    history = {kind: scenario_history(root, kind) for kind in scenarios_for(root)}
    protocol = freeze(
        directory / "protocol.json",
        {
            "scope": "Every historical native response; no formal targets or model fitting",
            "archives": {
                kind: archive_fingerprints(archives) for kind, archives in history.items()
            },
            "source_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "audit_transfer_history.py",
                    "record_audit.py",
                    "rss.py",
                    "executions.py",
                )
            },
            "collision_check": "Saved native ego-collision flags and feedback, not a new geometric collision simulator",
            "termination_check": "Measured 20 Hz clock, ego road departure and registered duration",
            "background_check": "Rotated body road extent and saved native crash-control flags",
            "formal_target_labels_read": False,
            "new_physical_executions": 0,
        },
    )
    rows = []
    for kind, archives in history.items():
        for archive in archives:
            bank = ExecutionBank(root / "physical/history" / kind / archive.source_id)
            specification = {
                "protocol": protocol,
                "scenario": kind,
                "source_id": archive.source_id,
                "physical_bank_snapshot": bank.snapshot_fingerprint,
            }
            path = directory / kind / f"{archive.source_id}.json"
            if path.exists():
                saved = read_json(path)
                if saved["specification"] != specification:
                    raise ValueError(
                        "Historical quality record belongs to another archive or physical bank"
                    )
            else:
                truth = BankTruth(bank)
                if not np.array_equal(truth.labels, archive.collision):
                    raise ValueError(
                        "Historical archive collision labels disagree with native traces"
                    )
                difference = float(np.abs(truth.margins - archive.continuous_margin).max())
                if difference > 1e-12:
                    raise ValueError(
                        "Historical archive RSS disagrees with its measured trajectory"
                    )
                saved = {
                    "specification": specification,
                    "profile": {**truth.profile, **background_profile(bank, truth)},
                    "maximum_rss_error": difference,
                }
                write_json(path, saved)
            rows.append(
                {
                    "scenario": kind,
                    "source_id": archive.source_id,
                    **saved["profile"],
                    "maximum_rss_error": saved["maximum_rss_error"],
                }
            )
            print(rows[-1], flush=True)
    from controllers.highway.catalog import FORMAL_HISTORY_SUTS

    from .formal_scenarios import HISTORY_COUNTS

    if len(rows) != len(FORMAL_HISTORY_SUTS) * len(scenarios_for(root)) or sum(
        r["records"] for r in rows
    ) != len(FORMAL_HISTORY_SUTS) * sum(HISTORY_COUNTS[k] for k in scenarios_for(root)):
        raise ValueError("Historical quality audit does not cover the registered complete cohort")
    report = {
        "protocol": protocol,
        "status": "complete",
        "banks": len(rows),
        "records": sum(r["records"] for r in rows),
        "rows": rows,
        "maximum_rss_error": max(r["maximum_rss_error"] for r in rows),
        "interpretation": "Noncollision includes road-departure terminations; quality counts do not establish transfer effectiveness",
        "formal_target_labels_read": False,
        "new_physical_executions": 0,
    }
    write_json(directory / "summary.json", report)
    return report


if __name__ == "__main__":
    audit()
