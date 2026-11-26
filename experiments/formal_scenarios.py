"""Integer physical parameters in rectangular domains, without outcome filtering.

The accepted braking domain excludes all previously observed development
coordinates. Distance is sampled directly, without braking-based filtering.
"""

import hashlib
from dataclasses import asdict
from itertools import islice
from math import prod
from pathlib import Path

import numpy as np

from simulator.highway.timing import timing_contract

from .protocol import ROOT
from .scenarios import DEVELOPMENT_BOUNDS, DOMAINS, FORMAL_SCENARIOS, validate
from .schema import Scenario
from .storage import read_json, write_json

SEEDS = {"history": 31013, "target": 61001}
HISTORY_COUNT = 500
TARGET_COUNT = 2000
BRAKING_KIND = "car_following_braking_onset"
HISTORY_COUNTS = dict.fromkeys(FORMAL_SCENARIOS, HISTORY_COUNT)
DEVELOPMENT_ROOT = ROOT / "results/scene_design/braking/critical_following"
INTERACTION_ROOT = ROOT / "results/scene_design/interaction"
FREE_BOUNDS = {
    BRAKING_KIND: {
        name: tuple(map(int, bounds)) for name, bounds in DOMAINS[BRAKING_KIND].bounds.items()
    },
    **{kind: DEVELOPMENT_BOUNDS[kind] for kind in FORMAL_SCENARIOS if kind != BRAKING_KIND},
}


def development_coordinates(kind):
    if kind not in FREE_BOUNDS:
        raise ValueError("Scenario has no current formal sampling registration")
    if kind == BRAKING_KIND:
        protocol = read_json(DEVELOPMENT_ROOT / "protocol.json")
        return {
            tuple(point)
            for name in ("excluded_formal_coordinates", "previous_development_coordinates_excluded")
            for point in protocol[name]
        }
    points = set()

    def visit(value):
        if isinstance(value, dict):
            if value.get("kind") == kind and "parameters" in value:
                points.add(tuple(value["parameters"][name] for name in FREE_BOUNDS[kind]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in sorted((INTERACTION_ROOT / kind).rglob("*protocol.json")):
        visit(read_json(path))
    if not points:
        raise ValueError("Missing registered interaction development coordinates")
    return points


def map_scene(kind, point, scenario_id):
    bounds = FREE_BOUNDS[kind]
    point = np.asarray(point, dtype=float)
    if (
        point.shape != (len(bounds),)
        or not np.isfinite(point).all()
        or ((point < 0) | (point > 1)).any()
    ):
        raise ValueError("Expected one finite coordinate in [0, 1] per independent parameter")
    free = {
        name: low + min(int(value * (high - low + 1)), high - low)
        for (name, (low, high)), value in zip(bounds.items(), point)
    }
    categories = {name: values[0] for name, values in DOMAINS[kind].categories.items()}
    scene = Scenario(scenario_id, kind, dict(free), categories)
    # A failing assertion is a domain implementation error, never a skipped point.
    validate(scene)
    return scene, free


def domain_contract():
    return {
        "name": "integer-physical-domain",
        "free_bounds": {k: [[n, *v] for n, v in b.items()] for k, b in FREE_BOUNDS.items()},
        "feature_envelopes": {k: DOMAINS[k].to_dict() for k in FREE_BOUNDS},
        "distribution": "uniform without replacement on the full Cartesian integer grid",
        "integer_step": 1,
        "units": "m, m/s, s; no encoded fractional units",
        "coordinate_seeds": SEEDS,
        "split_capacity": {
            "history": HISTORY_COUNTS,
            "target": TARGET_COUNT,
        },
        "split_allocation": "exclude observed development coordinates, then disjoint history and target",
        "development_exclusions": {
            kind: [list(point) for point in sorted(development_coordinates(kind))]
            for kind in FREE_BOUNDS
        },
        "timing": timing_contract(),
        "episode_duration_s": {kind: 14.0 if kind == BRAKING_KIND else 10.0 for kind in FREE_BOUNDS},
        "category_allocation": "balanced fixed random permutation; numeric coordinates remain unique",
        "braking_event_time_s": 0.0,
        "braking_ramp_s": 0.0,
        "environment_seed": 81001,
        "implementation": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "experiments/formal_scenarios.py",
                "experiments/scenarios.py",
                "simulator/highway/collision_env.py",
                "simulator/highway/observation.py",
                "simulator/highway/timing.py",
                "controllers/highway/control.py",
            )
        },
    }


def coordinate_stream(kind, split):
    if split not in SEEDS:
        raise ValueError("Unknown coordinate split")
    if kind not in FREE_BOUNDS:
        raise ValueError("Scenario has no current formal sampling registration")
    sizes = [hi - lo + 1 for lo, hi in FREE_BOUNDS[kind].values()]
    grid_size = prod(sizes)
    counts = {
        "history": HISTORY_COUNTS[kind],
        "target": TARGET_COUNT,
    }
    excluded = set()
    for coordinate in development_coordinates(kind):
        index, stride = 0, 1
        for value, (lo, hi), size in zip(coordinate, FREE_BOUNDS[kind].values(), sizes):
            if not lo <= value <= hi:
                break
            index += (value - lo) * stride
            stride *= size
        else:
            excluded.add(index)
    if sum(counts.values()) > grid_size - len(excluded):
        raise ValueError("Integer grid cannot hold the disjoint registered splits")
    used = set(excluded)
    for partition, count in counts.items():
        rng = np.random.default_rng(SEEDS[partition])
        indices = []
        while len(indices) < count:
            index = int(rng.integers(grid_size))
            if index not in used:
                used.add(index)
                indices.append(index)
        if partition == split:
            break
    categories = DOMAINS[kind].categories
    assignments = {
        name: np.random.default_rng(SEEDS[split]).permutation(
            np.tile(values, (len(indices) + len(values) - 1) // len(values))[:len(indices)]
        )
        for name, values in categories.items()
    }
    for ordinal, index in enumerate(indices):
        digits = []
        for size in sizes:
            index, digit = divmod(index, size)
            digits.append(digit)
        point = [(digit + 0.5) / size for digit, size in zip(digits, sizes)]
        scene, free = map_scene(kind, point, f"{kind}:integer:{split}:{SEEDS[split]}:{ordinal:06d}")
        scene = Scenario(
            scene.scenario_id, kind, scene.parameters,
            {name: str(values[ordinal]) for name, values in assignments.items()},
        )
        yield {"scenario": asdict(scene), "unit_coordinate": point, "independent": free}


def candidate_payload(kind, count, split):
    if count <= 0:
        raise ValueError("Candidate count must be positive")
    records = list(islice(coordinate_stream(kind, split), count))
    if len(records) != count:
        raise ValueError("Candidate count exceeds the registered integer split capacity")
    return {
        "status": "complete",
        "specification": {
            "kind": kind,
            "count": count,
            "split": split,
            "domain": domain_contract(),
        },
        "records": records,
        "generated_count": count,
        "rejected_count": 0,
        "physical_executions": 0,
    }


def _payload_matches(saved, expected):
    if saved == expected:
        return True
    from .source_registration import source_contract_matches

    if (
        saved.keys() != expected.keys()
        or any(saved[k] != expected[k] for k in saved if k != "specification")
        or saved["specification"].keys() != expected["specification"].keys()
        or any(saved["specification"][k] != expected["specification"][k] for k in saved["specification"] if k != "domain")
    ):
        return False
    sd, ed = saved["specification"]["domain"], expected["specification"]["domain"]
    return (
        sd.keys() == ed.keys()
        and all(sd[k] == ed[k] for k in sd if k != "implementation")
        and source_contract_matches(sd["implementation"], ed["implementation"])
    )


def prepare_candidates(kind, count, split, destination):
    path = Path(destination) / "manifest.json"
    expected = candidate_payload(kind, count, split)
    if path.exists() and not _payload_matches(read_json(path), expected):
        raise ValueError("Direct candidates differ from their registered domain or coordinates")
    write_json(path, expected)
    return [Scenario(**r["scenario"]) for r in expected["records"]]


def verify_candidates(destination, kind, count, split):
    saved = read_json(Path(destination) / "manifest.json")
    if not _payload_matches(saved, candidate_payload(kind, count, split)):
        raise ValueError("Direct candidates differ from their registered domain or coordinates")
    return [Scenario(**r["scenario"]) for r in saved["records"]]
