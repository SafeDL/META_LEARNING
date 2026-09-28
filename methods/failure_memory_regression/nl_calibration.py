"""Independent efficiency calibration for the next NL-IDM development cycle.

Candidate selection uses only vehicle progress and a fixed collision ceiling.
It never runs the bidirectional tester or reads a confirmation bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from highway_sim_env.envs.fbrt_unified_env import run_build_episode
from methods.failure_memory_regression.bidirectional import compile_manifest
from methods.failure_memory_regression.schema import stable_hash
from sut_algorithms.highway_env.registry import build_spec_factory


BUILDS = ("nl_v0", "nl_eff_c1", "nl_eff_c2", "nl_eff_c3")
ROOT = Path("results/method_chains/failure_memory_regression/nl_release/calibration")
SEED = 4179943
MAX_EXTRA_COLLISIONS = 7  # At most 7/75 = 9.33 percentage points on this grid.


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")


def freeze(root: Path = ROOT) -> dict:
    cases = compile_manifest("calibration", resolution=5, seed=SEED)
    manifest = root / "scenario_manifest.jsonl"
    if manifest.exists() and _jsonl(manifest) != cases:
        raise ValueError("calibration manifest changed")
    if not manifest.exists():
        _write_jsonl(manifest, cases)
    protocol = {
        "purpose": "development-only normal efficiency calibration",
        "builds": BUILDS, "simulator_seed": SEED,
        "scenario_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "selection_rule": "highest mean ego progress per scheduled second among candidates",
        "collision_constraint": f"candidate collisions <= V0 collisions + {MAX_EXTRA_COLLISIONS}",
        "inconclusive_rule": "no candidate selection if any physical episode is inconclusive",
        "tie_break": "lexicographically smaller candidate ID",
        "forbidden_selection_inputs": ["regression count", "improvement count",
                                       "tester discovery", "confirmation results"],
        "build_fingerprints": {build: build_spec_factory(build).fingerprint
                               for build in BUILDS},
    }
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(protocol)):
        raise ValueError("calibration protocol changed")
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return {"scenarios": len(cases), "required_episodes": len(cases) * len(BUILDS),
            "manifest_sha256": protocol["scenario_manifest_sha256"]}


def measure(root: Path = ROOT, limit: int | None = None) -> dict:
    cases = _jsonl(root / "scenario_manifest.jsonl")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if hashlib.sha256((root / "scenario_manifest.jsonl").read_bytes()).hexdigest() != protocol[
            "scenario_manifest_sha256"]:
        raise ValueError("calibration manifest fingerprint mismatch")
    path = root / "physical_episodes.jsonl"
    previous = _jsonl(path)
    bank = {(row["build_id"], row["scenario_id"]): row for row in previous}
    if len(bank) != len(previous):
        raise ValueError("duplicate calibration result")
    identities = {(build, case["scenario_id"]): case for build in BUILDS for case in cases}
    if set(bank) - set(identities):
        raise ValueError("calibration result outside frozen manifest")
    for (build, sid), row in bank.items():
        case = identities[build, sid]
        if (row["scenario"] != case or row["build_fingerprint"] !=
                protocol["build_fingerprints"][build] or row["simulator_seed"] != SEED):
            raise ValueError("stale calibration cache")
    new_count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for (build, sid), case in identities.items():
            if (build, sid) in bank:
                continue
            if limit is not None and new_count >= limit:
                break
            outcome, trace = run_build_episode(build, case, SEED, with_trace=True)
            duration = float(case["fixed_context"]["duration_s"])
            outcome["ego_progress_m"] = (float(trace[-1]["ego"]["x_m"])
                                          - float(trace[0]["ego"]["x_m"]))
            outcome["scheduled_duration_s"] = duration
            outcome["progress_rate_mps"] = outcome["ego_progress_m"] / duration
            outcome["calibration_execution_id"] = stable_hash({
                "physical_execution_id": outcome["execution_id"],
                "progress_definition": "last logged ego x minus initial ego x, divided by scheduled duration",
            })
            stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
            stream.flush()
            new_count += 1
    return {"new_physical_episodes": new_count, "cached_episodes": len(bank),
            "required_episodes": len(identities)}


def choose(root: Path = ROOT) -> dict:
    cases = _jsonl(root / "scenario_manifest.jsonl")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = _jsonl(root / "physical_episodes.jsonl")
    expected = {(build, case["scenario_id"]) for build in BUILDS for case in cases}
    if len(rows) != len(expected) or {(row["build_id"], row["scenario_id"])
                                      for row in rows} != expected:
        raise ValueError("calibration physical bank incomplete")
    groups = defaultdict(list)
    for row in rows:
        if row["inconclusive"] or (not row["completed"] and not row["ego_collision"]):
            raise ValueError("inconclusive calibration episode; no selection")
        if row["build_fingerprint"] != protocol["build_fingerprints"][row["build_id"]]:
            raise ValueError("stale calibration build")
        groups[row["build_id"]].append(row)
    baseline_collisions = sum(row["ego_collision"] for row in groups["nl_v0"])
    summary = []
    for build in BUILDS:
        local = groups[build]
        collisions = sum(row["ego_collision"] for row in local)
        summary.append({"build_id": build, "episodes": len(local),
                        "collisions": collisions,
                        "mean_progress_rate_mps": sum(row["progress_rate_mps"] for row in local)
                        / len(local),
                        "feasible": build != "nl_v0" and
                        collisions <= baseline_collisions + MAX_EXTRA_COLLISIONS})
    eligible = [row for row in summary if row["feasible"]]
    selected = min(eligible, key=lambda row: (-row["mean_progress_rate_mps"],
                                               row["build_id"]))["build_id"] if eligible else None
    result = {"selected": selected, "baseline_collisions": baseline_collisions,
              "max_extra_collisions": MAX_EXTRA_COLLISIONS, "candidate_summary": summary,
              "physical_episodes": len(rows), "selection_used_tester_results": False,
              "input_bank_sha256": hashlib.sha256(
                  (root / "physical_episodes.jsonl").read_bytes()).hexdigest()}
    (root / "selection.json").write_text(json.dumps(result, ensure_ascii=False,
                                                    sort_keys=True, indent=2) + "\n",
                                         encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "measure", "choose"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.root)
    elif args.action == "measure":
        result = measure(args.root, args.limit)
    else:
        result = choose(args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
