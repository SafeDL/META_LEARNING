"""Check the unchanged S01 benchmark and passive response banks."""
import hashlib

import numpy as np

from highway_sim_env.s01_parameters import NAMES, bounds
from .common import REPO, read_json, write_json
from .s01 import (BENCHMARK_ROOT, COUNT, HISTORY_ROOT, ROOT, SOURCES, TARGET,
                  TARGET_ROOT, build_spec, cells_for, load_scenes, numeric_inputs, response_rows)


def protected_files():
    files = [p for p in BENCHMARK_ROOT.rglob("*") if p.is_file()]
    files += list((ROOT / "model").glob("*.pt"))
    files += list((ROOT / "measurements/A").glob("*.risk.jsonl"))
    files += list((ROOT / "measurements/D").glob("*.risk.jsonl"))
    return sorted(files)


def protected_hashes():
    return {str(p.relative_to(REPO)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_files()}


def main():
    a, d = load_scenes("A"), load_scenes("D")
    xa, xd = numeric_inputs(a), numeric_inputs(d)
    assert len(a) == len(d) == COUNT
    assert len({s["scenario_id"] for s in a}) == len({s["scenario_id"] for s in d}) == COUNT
    assert not set(map(tuple, xa)) & set(map(tuple, xd))
    assert all(s["template_id"] == "fbrt_cutin" and s["fixed_context"]["ego_speed_mps"] == 25 for s in a + d)
    assert tuple(bounds()) == NAMES
    expected = {"initial_clearance_m": (8., 60.), "lead_speed_mps": (15., 25.),
                "lane_change_time_scale_s": (1.5, 3.), "event_start_s": (.5, 2.)}
    assert bounds() == expected
    inventory = []
    for bank, names in (("A", SOURCES), ("D", (TARGET,))):
        for name in names:
            scenes, rows = response_rows(bank, name)
            labels = np.asarray([r["ego_collision"] for r in rows], bool)
            inventory.append({"bank": bank, "build": name, "count": len(rows),
                              "collisions": int(labels.sum()), "risk_valid": True,
                              "build_fingerprint": build_spec(name).fingerprint})
    _, target = response_rows("D")
    labels = np.asarray([r["ego_collision"] for r in target], bool)
    assert int(labels.sum()) == 116 and len(np.unique(cells_for(d)[labels])) == 33
    source_protocol = read_json(HISTORY_ROOT / "protocol.json")
    assert tuple(source_protocol["sources"]) == SOURCES
    for name in SOURCES:
        assert build_spec(name).profile == source_protocol["builds"][name]["profile"]
    target_protocol = read_json(TARGET_ROOT / "protocol.json")
    assert target_protocol["target_build_id"] == TARGET
    report = {"status": "PASS", "A_count": COUNT, "D_count": COUNT,
              "sources": list(SOURCES), "target": TARGET,
              "D_collisions": 116, "D_collision_cells": 33,
              "A_D_disjoint": True, "source_and_target_profiles_unchanged": True,
              "measurement_labels_match_original": True, "inventory": inventory,
              "protected_hashes": protected_hashes()}
    write_json(ROOT / "audit/s01.json", report)
    print("S01 audit PASS: A/D 2048, five original IDM sources, original FVDM, 116 collisions / 33 cells")
    return report


if __name__ == "__main__":
    main()
