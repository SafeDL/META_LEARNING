"""Replay complete pools while the sealed collector continues measuring."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.prepare import measure_scene

from .audit_confirmation import AUDIT, REPLAY_INDICES, verify_bank
from .confirmation import CONFIRMATION, profiles, verify_lock


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    completed = [(profile, replicate) for profile in profiles()
                 for replicate in range(protocol["replicates_per_profile"])
                 if (CONFIRMATION / profile.name / f"pool_{replicate}" /
                     "cost.json").exists()]
    audited, new_calls = [], 0
    with ProcessPoolExecutor(max_workers=2,
                             mp_context=mp.get_context("spawn")) as executor:
        for profile, replicate in completed:
            folder = CONFIRMATION / profile.name / f"pool_{replicate}"
            scenes = read_json(folder / "scenarios.json")
            rows = verify_bank(folder, scenes, profile)
            destination = AUDIT / profile.name / f"pool_{replicate}.json"
            if destination.exists():
                checks = read_json(destination)
            else:
                checks = [{
                    "index": index,
                    "replay": replay
                } for index, replay in zip(
                    REPLAY_INDICES,
                    executor.map(measure_scene, [(scenes[i], profile)
                                                 for i in REPLAY_INDICES]))]
                write_json(destination, checks)
                new_calls += len(checks)
            assert [row["index"] for row in checks] == list(REPLAY_INDICES)
            for check in checks:
                replay, original = check["replay"], rows[check["index"]]
                assert replay["scenario_id"] == original["scenario_id"]
                assert abs(replay["risk"] - original["risk"]) < 1e-12
                assert replay["collision"] == original["collision"]
            audited.append({
                "profile": profile.name,
                "replicate": replicate,
                "rows": len(rows),
                "exact_physical_replays": len(checks)
            })
            print("COMPLETE POOL REPLAY VERIFIED",
                  profile.name,
                  replicate,
                  flush=True)
    write_json(
        AUDIT / "measurement_progress.json", {
            "role":
            "Partial physical audit, no method tuning or statistical conclusion",
            "audited": audited,
            "new_physical_replay_calls": new_calls,
            "complete_at_audit_start": len(completed),
            "all_available_pools_audited": len(audited) == len(completed),
            "final_full_cohort_audit_required": True
        })


if __name__ == "__main__":
    main()
