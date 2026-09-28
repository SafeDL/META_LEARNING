"""Frozen NL-IDM full-bank experiment and leak-resistant bidirectional replay.

The selector receives only complete parent outcomes and an oracle that reveals
one target outcome per charged query.  Transition truth is evaluator-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.special import expit

from highway_sim_env.envs.fbrt_unified_env import EXECUTION_CONTRACT, run_build_episode
from methods.failure_memory_regression.replay_utils import is_usable_outcome
from methods.failure_memory_regression.schema import stable_hash
from sut_algorithms.highway_env.nl_release import (
    NL2_PURPOSES, NL3_PURPOSES, NL_PURPOSES, NL_RELEASES,
)
from sut_algorithms.highway_env.registry import build_spec_factory


ROOT = Path("results/method_chains/failure_memory_regression/nl_release")
CATALOGUE = Path(__file__).resolve().parent / "configs" / "scenario_catalogue.yaml"
FAMILIES = ("S01", "S02", "S08")
BUILDS = tuple(NL_RELEASES)
METHODS = ("random", "static_risk", "static_boundary", "center_residual",
           "coordinate_residual", "target_only", "directed_residual")
EXPLORATORY_METHODS = ("directed_context_laplace", "directed_context_ucb",
                       "static_margin_coverage", "coordinate_margin_frontier",
                       "directed_margin_frontier", "static_dual_margin_coverage",
                       "coordinate_dual_margin_frontier",
                       "directed_dual_margin_frontier",
                       "static_dual_coverage2", "coordinate_dual_local",
                       "directed_dual_local", "static_majority_coverage2",
                       "coordinate_majority_local", "directed_majority_local",
                       "directed_gated_majority", "static_role_coverage2",
                       "coordinate_role_gated", "directed_role_gated")
SEED = 4179931
PROTOCOL_VERSION = "nl-bidirectional-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compile_manifest(split: str = "confirmation", resolution: int | None = None,
                     seed: int = SEED) -> list[dict]:
    """Predefined physical contexts and complete two-dimensional rule grids."""
    if split not in {"development", "confirmation", "calibration",
                     "development2", "confirmation2", "development3",
                     "confirmation3", "confirmation4", "ppo_confirmation",
                     "cross_sut_confirmation"}:
        raise ValueError("unsupported frozen scenario split")
    n = resolution or (5 if split in {"development", "calibration", "development2",
                                        "development3"} else 11)
    if n < 3:
        raise ValueError("grid resolution must be at least 3")
    catalogue = yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))
    cards = {card["id"]: card for card in catalogue["scenarios"]}
    contexts_by_split = {
        "development": ((-2.0, -1.0, 0.15),),
        "calibration": ((1.0, -2.0, 0.05),),
        "confirmation": ((0.0, 0.0, 0.0), (2.0, -1.0, -0.15), (-2.0, 1.0, 0.15)),
        "development2": ((-0.5, -2.0, 0.10),),
        "confirmation2": ((1.0, 0.0, 0.10), (-1.0, -1.0, -0.10),
                          (2.0, 1.0, 0.05)),
        "development3": ((0.5, 0.5, 0.0),),
        "confirmation3": ((-1.5, 0.0, 0.10), (1.5, -1.5, -0.05),
                          (0.0, 1.5, 0.15)),
        "confirmation4": ((-1.5, -1.25, -0.10), (0.5, 0.0, 0.10),
                          (1.5, 1.25, 0.0)),
        "ppo_confirmation": ((-2.5, -0.5, -0.15), (0.0, 1.0, 0.0),
                             (2.5, -1.0, 0.15)),
        "cross_sut_confirmation": ((-2.0, 1.0, 0.05),
                                    (0.5, -1.0, -0.10),
                                    (2.0, 0.5, 0.10)),
    }
    contexts = contexts_by_split[split]
    rows = []
    for family in FAMILIES:
        card = cards[family]
        names = tuple(card["research_bounds"])
        if len(names) != 2:
            raise ValueError(f"{family} is not a 2D card")
        for context_index, (ego_delta, lead_delta, event_delta) in enumerate(contexts):
            fixed = dict(card["fixed_context"])
            fixed["ego_speed_mps"] = float(fixed["ego_speed_mps"] + ego_delta)
            fixed["lead_speed_mps"] = float(fixed["lead_speed_mps"] + lead_delta)
            fixed["event_start_s"] = float(fixed["event_start_s"] + event_delta)
            context_id = f"{family}:{split}:c{context_index}"
            for i in range(n):
                for j in range(n):
                    values = {name: float(bounds[0] + (bounds[1] - bounds[0]) * index / (n - 1))
                              for name, bounds, index in zip(
                                  names, (card["research_bounds"][name] for name in names), (i, j))}
                    rows.append({
                        "scenario_id": f"nl:{split}:{family}:c{context_index}:{i:02d}:{j:02d}",
                        "template_id": card["template_id"], "catalogue_id": family,
                        "context_id": context_id, "grid_index": [i, j],
                        "grid_resolution": n, "parameterization_version": "nl_grid_v1",
                        "active_parameters": values, "fixed_context": fixed,
                        "research_bounds": card["research_bounds"],
                        "required_capability": card.get("required_capability", []),
                        "simulator_seed": seed,
                    })
    return rows


def freeze(root: Path, split: str = "confirmation", resolution: int | None = None,
           seed: int = SEED, builds: tuple[str, ...] = BUILDS,
           methods: tuple[str, ...] = METHODS) -> list[dict]:
    if len(builds) != 3 or len(set(builds)) != 3:
        raise ValueError("a release chain needs three distinct builds")
    if not methods or len(set(methods)) != len(methods) or any(
            method not in METHODS + EXPLORATORY_METHODS for method in methods):
        raise ValueError("unknown or duplicate method in frozen comparison")
    specs = [build_spec_factory(build) for build in builds]
    if specs[0].parent_build_id is not None or any(
            child.parent_build_id != parent.build_id for parent, child in zip(specs, specs[1:])):
        raise ValueError("builds are not a parent-child release chain")
    if len({spec.adapter_kind for spec in specs}) != 1 or len(
            {spec.policy_name for spec in specs}) != 1:
        raise ValueError("release chain changes the controller contract")
    rows = compile_manifest(split, resolution, seed)
    manifest_path = root / "scenario_manifest.jsonl"
    protocol_path = root / "protocol.json"
    if manifest_path.exists():
        if _read_jsonl(manifest_path) != rows:
            raise ValueError("frozen scenario manifest differs; use a new output root")
    else:
        _write_jsonl(manifest_path, rows)
    protocol = {"version": PROTOCOL_VERSION, "split": split, "resolution": rows[0]["grid_resolution"],
                "simulator_seed": seed, "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()).hexdigest(), "families": FAMILIES,
                "builds": builds, "direction_budget": 20, "total_budget": 40,
                "schedule": "alternate regression then improvement; transfer unused pool quota",
                "endpoint": "ego collision; valid noncollision requires completed episode",
                "methods": methods}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != json.loads(
                json.dumps(protocol)):
            raise ValueError("frozen protocol differs; use a new output root")
    else:
        _write_json(protocol_path, protocol)
    lineage = []
    for spec in specs:
        build = spec.build_id
        lineage.append({"build_id": build, "parent_build_id": spec.parent_build_id,
                        "purpose": (NL_PURPOSES | NL2_PURPOSES | NL3_PURPOSES).get(build, build),
                        "profile": spec.profile,
                        "build_fingerprint": spec.fingerprint})
    lineage_path = root / "version_lineage.jsonl"
    if lineage_path.exists() and _read_jsonl(lineage_path) != lineage:
        raise ValueError("release profile changed after freeze")
    _write_jsonl(lineage_path, lineage)
    return rows


def measure(root: Path, *, limit: int | None = None) -> dict:
    """Resume only when every cached row matches the frozen build and scene."""
    cases = _read_jsonl(root / "scenario_manifest.jsonl")
    if not cases:
        raise ValueError("freeze a nonempty manifest first")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    seed = int(protocol["simulator_seed"])
    builds = tuple(protocol["builds"])
    path = root / "full_response_bank.jsonl"
    previous = _read_jsonl(path)
    bank = {(row["build_id"], row["scenario_id"]): row for row in previous}
    if len(bank) != len(previous):
        raise ValueError("duplicate full-bank row")
    identities = {(build, case["scenario_id"]): (build_spec_factory(build), case)
                  for build in builds for case in cases}
    if set(bank) - set(identities):
        raise ValueError("bank contains rows outside frozen manifest")
    for key, row in bank.items():
        spec, case = identities[key]
        if (row["build_fingerprint"] != spec.fingerprint or row["scenario"] != case
                or row["simulator_seed"] != seed):
            raise ValueError(f"stale cached physical result: {key}")
    new_count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for key, (spec, case) in identities.items():
            if key in bank:
                continue
            if limit is not None and new_count >= limit:
                break
            result, _ = run_build_episode(spec.build_id, case, seed)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
            stream.flush()
            new_count += 1
    return {"new_physical_episodes": new_count, "cached_episodes": len(bank),
            "required_episodes": len(identities)}


def _label(row: dict) -> int | None:
    return int(row["ego_collision"]) if is_usable_outcome(row) else None


def transition_truth(cases: list[dict], bank: dict[tuple[str, str], dict],
                     parent: str, target: str) -> list[dict]:
    rows = []
    for case in cases:
        sid = case["scenario_id"]
        old, new = _label(bank[parent, sid]), _label(bank[target, sid])
        state = ("UNKNOWN" if old is None or new is None else
                 "REGRESSION" if old == 0 and new == 1 else
                 "IMPROVEMENT" if old == 1 and new == 0 else
                 "PERSISTENT_FAILURE" if old == 1 else "STABLE_PASS")
        rows.append({"parent_build_id": parent, "target_build_id": target,
                     "scenario_id": sid, "context_id": case["context_id"],
                     "parent_label": old, "target_label": new, "transition": state})
    return rows


class TargetOracle:
    """The only selector-facing path to a charged target observation."""

    def __init__(self, target_rows: dict[str, dict]):
        self.__rows = target_rows
        self.queried: set[str] = set()

    def query(self, scenario_id: str) -> dict:
        if scenario_id in self.queried:
            raise ValueError("repeat target query")
        self.queried.add(scenario_id)
        return dict(self.__rows[scenario_id])


def _xy(case: dict) -> np.ndarray:
    return np.asarray([(case["active_parameters"][name] - bounds[0]) / (bounds[1] - bounds[0])
                       for name, bounds in case["research_bounds"].items()], dtype=float)


def _base(xy: np.ndarray) -> np.ndarray:
    a, b = 2 * xy - 1
    return np.asarray([1.0, a, b, a * b, a * a - 1 / 3, b * b - 1 / 3])


def _fit_offset(x: np.ndarray, y: np.ndarray, offset: np.ndarray,
                variance: np.ndarray) -> np.ndarray:
    """Strongly regularized MAP logistic residual; no target data means zero."""
    theta = np.zeros(x.shape[1])
    if not len(y):
        return theta
    precision = 1.0 / variance
    for _ in range(25):
        p = expit(np.clip(offset + x @ theta, -35, 35))
        weights = np.maximum(p * (1 - p), 1e-8)
        gradient = x.T @ (p - y) + precision * theta
        hessian = x.T @ (weights[:, None] * x) + np.diag(precision)
        delta = np.linalg.solve(hessian, gradient)
        theta -= delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return theta


def _laplace_covariance(x: np.ndarray, y: np.ndarray, offset: np.ndarray,
                        theta: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Deterministic Gaussian posterior approximation at the MAP estimate."""
    precision = np.diag(1.0 / variance)
    if len(y):
        p = expit(np.clip(offset + x @ theta, -35, 35))
        precision += x.T @ ((p * (1.0 - p))[:, None] * x)
    return np.linalg.inv(precision)


class ParentModel:
    def __init__(self, cases: list[dict], parent_rows: dict[str, dict]):
        self.cases = {case["scenario_id"]: case for case in cases}
        self.parents = {sid: _label(row) for sid, row in parent_rows.items()}
        self.coefficients: dict[str, np.ndarray] = {}
        self.edges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.failure_fraction: dict[str, float] = {}
        self.lead_failure_fraction: dict[str, float] = {}
        groups: dict[str, list[dict]] = {}
        for case in cases:
            groups.setdefault(case["context_id"], []).append(case)
        self.context_ids = tuple(sorted(groups))
        self.context_index = {context: index for index, context in enumerate(self.context_ids)}
        for context, local in groups.items():
            valid = [case for case in local if self.parents[case["scenario_id"]] is not None]
            self.failure_fraction[context] = (
                sum(self.parents[case["scenario_id"]] == 1 for case in valid) /
                len(valid) if valid else 0.0)
            failed = [case for case in valid if self.parents[case["scenario_id"]] == 1]
            self.lead_failure_fraction[context] = (
                sum(parent_rows[case["scenario_id"]].get("collision_partner_role") == "lead"
                    for case in failed) / len(failed) if failed else 0.0)
            x = np.vstack([_base(_xy(case)) for case in valid]) if valid else np.empty((0, 6))
            y = np.asarray([self.parents[case["scenario_id"]] for case in valid], dtype=float)
            self.coefficients[context] = _fit_offset(x, y, np.zeros(len(y)),
                                                      np.asarray([2, 1, 1, .8, .8, .8]))
            grid = {tuple(case["grid_index"]): case for case in local}
            edges = []
            for case in local:
                label = self.parents[case["scenario_id"]]
                if label is None:
                    continue
                i, j = case["grid_index"]
                for neighbor in ((i + 1, j), (i, j + 1)):
                    other = grid.get(neighbor)
                    if other is None:
                        continue
                    other_label = self.parents[other["scenario_id"]]
                    if other_label is None or other_label == label:
                        continue
                    failure, passed = (case, other) if label == 1 else (other, case)
                    edges.append((_xy(failure), _xy(passed)))
            # Four separated, deterministic parent-only fragments per context.
            selected = []
            for failure, passed in sorted(edges, key=lambda pair: tuple((pair[0] + pair[1]) / 2)):
                midpoint = (failure + passed) / 2
                if not selected or all(np.linalg.norm(midpoint - (a + b) / 2) >= .25
                                       for a, b in selected):
                    selected.append((failure, passed))
                if len(selected) == 4:
                    break
            self.edges[context] = selected

    def risk_logit(self, case: dict) -> float:
        return float(_base(_xy(case)) @ self.coefficients[case["context_id"]])

    def boundary_distance(self, case: dict) -> float:
        edges = self.edges[case["context_id"]]
        return min((float(np.linalg.norm(_xy(case) - (a + b) / 2)) for a, b in edges),
                   default=2.0)

    def features(self, case: dict, method: str) -> np.ndarray:
        xy = _xy(case)
        common = np.asarray([1.0, *(2 * xy - 1)])
        if method in {"coordinate_dual_local", "directed_dual_local",
                      "coordinate_majority_local", "directed_majority_local",
                      "directed_gated_majority", "coordinate_role_gated",
                      "directed_role_gated"}:
            local_coordinates = np.asarray([1.0, common[1], common[2],
                                            common[1] * common[2]])
            role_context = (self.failure_fraction[case["context_id"]] >= .5 and
                            self.lead_failure_fraction[case["context_id"]] >= .5)
            if method in {"coordinate_dual_local", "coordinate_majority_local",
                          "coordinate_role_gated"}:
                blocks = np.zeros((len(self.context_ids), 4))
                if method != "coordinate_role_gated" or role_context:
                    blocks[self.context_index[case["context_id"]]] = local_coordinates
            else:
                edge_features = []
                for failure, passed in self.edges[case["context_id"]]:
                    midpoint = (failure + passed) / 2
                    normal = (passed - failure) / np.linalg.norm(passed - failure)
                    kernel = np.exp(-sum((xy - midpoint) ** 2) / (.22 ** 2))
                    edge_features.extend((kernel, kernel * float(normal @ (xy - midpoint))))
                majority = (role_context if method == "directed_role_gated" else
                            self.failure_fraction[case["context_id"]] >= .5)
                coordinates = (np.zeros(4) if method in {
                    "directed_gated_majority", "directed_role_gated"} and not majority
                               else local_coordinates)
                directed = np.concatenate((edge_features,
                                           np.zeros(8 - len(edge_features))))
                if method in {"directed_gated_majority", "directed_role_gated"} and majority:
                    directed = np.zeros(8)
                blocks = np.zeros((len(self.context_ids), 12))
                blocks[self.context_index[case["context_id"]]] = np.concatenate((
                    coordinates, directed))
            return np.concatenate((common, blocks.ravel()))
        if method in {"coordinate_residual", "coordinate_margin_frontier",
                      "coordinate_dual_margin_frontier",
                      "target_only"}:
            return np.concatenate((common, [common[1] * common[2]]))
        if method == "center_residual":
            # Ordinary failure-centered features have no pass-to-fail direction.
            centers = [a for a, _ in self.edges[case["context_id"]]]
            local = [np.exp(-sum((xy - center) ** 2) / (.22 ** 2))
                     for center in centers]
            return np.concatenate((common, local, np.zeros(4 - len(local))))
        if method in {"directed_residual", "directed_context_laplace",
                      "directed_context_ucb", "directed_margin_frontier",
                      "directed_dual_margin_frontier"}:
            local = []
            for failure, passed in self.edges[case["context_id"]]:
                midpoint = (failure + passed) / 2
                normal = (passed - failure) / np.linalg.norm(passed - failure)
                kernel = np.exp(-sum((xy - midpoint) ** 2) / (.22 ** 2))
                local.extend((kernel, kernel * float(normal @ (xy - midpoint))))
            padded = np.concatenate((local, np.zeros(8 - len(local))))
            if method == "directed_residual":
                return np.concatenate((common, padded))
            blocks = np.zeros((len(self.context_ids), 8))
            blocks[self.context_index[case["context_id"]]] = padded
            return np.concatenate((common, blocks.ravel()))
        raise ValueError(method)


def _schedule(parent_labels: dict[str, int | None], budget: int) -> list[str]:
    sizes = {"R": sum(label == 0 for label in parent_labels.values()),
             "I": sum(label == 1 for label in parent_labels.values())}
    remaining = {direction: min(budget // 2, size) for direction, size in sizes.items()}
    spare = budget - sum(remaining.values())
    for direction in ("R", "I"):
        extra = min(spare, sizes[direction] - remaining[direction])
        remaining[direction] += extra
        spare -= extra
    order = []
    while any(remaining.values()):
        for direction in ("R", "I"):
            if remaining[direction]:
                order.append(direction)
                remaining[direction] -= 1
    return order


def _parent_ttc(parent_row: dict) -> float:
    """Continuous parent-only safety margin; unknown margins sort last."""
    value = parent_row.get("min_ttc")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def _parent_failure_time(parent_row: dict) -> float:
    """Parent-only collision timing; early failures probe an unsafe context."""
    value = parent_row.get("collision_time_s")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def select(cases: list[dict], parent_rows: dict[str, dict], oracle: TargetOracle,
           method: str, *, budget: int = 40, seed: int = SEED) -> list[dict]:
    """Select without seeing unqueried target labels, including target anomalies."""
    if method not in METHODS + EXPLORATORY_METHODS:
        raise ValueError(method)
    model = ParentModel(cases, parent_rows)
    eligible = {sid: case for sid, case in model.cases.items()
                if model.parents[sid] is not None}
    order = _schedule({sid: model.parents[sid] for sid in eligible}, budget)
    rng = np.random.default_rng(seed)
    observations: list[tuple[dict, int]] = []
    queries = []
    for rank, direction in enumerate(order, 1):
        pool = [case for sid, case in eligible.items()
                if model.parents[sid] == (0 if direction == "R" else 1)]
        if not pool:
            raise AssertionError("schedule overran parent pool")
        if method in {"center_residual", "coordinate_residual",
                      "coordinate_margin_frontier",
                      "coordinate_dual_margin_frontier", "coordinate_dual_local",
                      "coordinate_majority_local", "directed_majority_local",
                      "directed_gated_majority", "coordinate_role_gated",
                      "directed_role_gated",
                      "target_only",
                      "directed_residual", "directed_context_laplace",
                      "directed_context_ucb", "directed_margin_frontier",
                      "directed_dual_margin_frontier", "directed_dual_local"}:
            # A single target response model is refitted from *all* observed
            # target labels, with direction-specific scores derived afterwards.
            width = len(model.features(pool[0], method))
            if observations:
                x = np.vstack([model.features(case, method) for case, _ in observations])
                y = np.asarray([label for _, label in observations], dtype=float)
                offset = np.asarray([0.0 if method == "target_only" else
                                     model.risk_logit(case) for case, _ in observations])
            else:
                x, y, offset = np.empty((0, width)), np.empty(0), np.empty(0)
            if method in {"coordinate_dual_local", "coordinate_majority_local",
                          "coordinate_role_gated"}:
                variance = np.asarray([.25] * 3 + [1.0] * (width - 3))
            elif method in {"directed_dual_local", "directed_majority_local",
                                  "directed_gated_majority", "directed_role_gated"}:
                variance = np.asarray(
                    [.25] * 3 + ([1.0, 1.0, 1.0, .8] + [1.5] * 8) *
                    len(model.context_ids))
            else:
                variance = (np.asarray([.25] * 3 + [1.5] * (width - 3))
                            if method in {"directed_context_laplace", "directed_context_ucb",
                                          "directed_margin_frontier",
                                          "directed_dual_margin_frontier"} else
                            np.asarray([.6] * 3 + [1.0] * (width - 3)))
            theta = _fit_offset(x, y, offset, variance)
            if method in {"directed_context_laplace", "directed_context_ucb",
                          "directed_margin_frontier", "directed_dual_margin_frontier",
                          "coordinate_dual_local", "directed_dual_local",
                          "coordinate_majority_local", "directed_majority_local",
                          "directed_gated_majority", "coordinate_role_gated",
                          "directed_role_gated"}:
                covariance = _laplace_covariance(x, y, offset, theta, variance)
                probs = []
                for case in pool:
                    features = model.features(case, method)
                    logit = model.risk_logit(case) + float(features @ theta)
                    logit_variance = float(features @ covariance @ features)
                    if method in {"directed_context_ucb", "directed_margin_frontier",
                                  "directed_dual_margin_frontier",
                                  "coordinate_dual_local", "directed_dual_local",
                                  "coordinate_majority_local", "directed_majority_local",
                                  "directed_gated_majority", "coordinate_role_gated",
                                  "directed_role_gated"} and direction == "R":
                        # One posterior standard deviation on the logit is a
                        # deterministic optimistic regression score. It can
                        # search poorly supported parent-pass contexts without
                        # a fixed coverage slot or target-truth access.
                        probs.append(float(expit(logit + np.sqrt(logit_variance))))
                    else:
                        probs.append(float(expit(logit / np.sqrt(
                            1.0 + np.pi * logit_variance / 8.0))))
            else:
                probs = [float(expit((0.0 if method == "target_only" else model.risk_logit(case)) +
                                     model.features(case, method) @ theta)) for case in pool]
            scores = probs if direction == "R" else [1 - p for p in probs]
        elif method == "static_risk":
            probs = [float(expit(model.risk_logit(case))) for case in pool]
            scores = probs if direction == "R" else [1 - p for p in probs]
        elif method in {"static_margin_coverage", "static_dual_margin_coverage",
                             "static_dual_coverage2", "static_majority_coverage2",
                             "static_role_coverage2"}:
            scores = ([-_parent_ttc(parent_rows[case["scenario_id"]]) for case in pool]
                      if direction == "R" else
                      [-_parent_failure_time(parent_rows[case["scenario_id"]])
                       for case in pool] if method in {"static_dual_margin_coverage",
                                                       "static_dual_coverage2",
                                                       "static_majority_coverage2",
                                                       "static_role_coverage2"} else
                      [1 - float(expit(model.risk_logit(case))) for case in pool])
        elif method == "static_boundary":
            scores = [-model.boundary_distance(case) for case in pool]
        else:
            scores = rng.random(len(pool)).tolist()
        # Parent-only margins probe each eligible context once before target
        # feedback. The dual revision also probes parent-fail contexts by
        # collision time. A discovered change opens a four-neighbor frontier.
        margin_methods = {"static_margin_coverage", "coordinate_margin_frontier",
                          "directed_margin_frontier", "static_dual_margin_coverage",
                          "coordinate_dual_margin_frontier",
                          "directed_dual_margin_frontier", "static_dual_coverage2",
                          "coordinate_dual_local", "directed_dual_local",
                          "static_majority_coverage2", "coordinate_majority_local",
                          "directed_majority_local", "directed_gated_majority",
                          "static_role_coverage2", "coordinate_role_gated",
                          "directed_role_gated"}
        dual_methods = {"static_dual_margin_coverage", "coordinate_dual_margin_frontier",
                        "directed_dual_margin_frontier", "static_dual_coverage2",
                        "coordinate_dual_local", "directed_dual_local",
                        "static_majority_coverage2", "coordinate_majority_local",
                        "directed_majority_local", "directed_gated_majority",
                        "static_role_coverage2", "coordinate_role_gated",
                        "directed_role_gated"}
        two_probe_methods = {"static_dual_coverage2", "coordinate_dual_local",
                             "directed_dual_local", "static_majority_coverage2",
                             "coordinate_majority_local", "directed_majority_local",
                             "directed_gated_majority", "static_role_coverage2",
                             "coordinate_role_gated", "directed_role_gated"}
        majority_methods = {"static_majority_coverage2", "coordinate_majority_local",
                            "directed_majority_local", "directed_gated_majority"}
        role_methods = {"static_role_coverage2", "coordinate_role_gated",
                        "directed_role_gated"}
        index = None
        phase = "score_rank"
        if method in margin_methods and (direction == "R" or method in dual_methods):
            margin = _parent_ttc if direction == "R" else _parent_failure_time
            counts = Counter(row["context_id"] for row in queries
                             if row["direction"] == direction)
            contexts = sorted({case["context_id"] for case in pool})
            if direction == "I" and method in majority_methods:
                contexts = [context for context in contexts
                            if model.failure_fraction[context] >= .5]
            if direction == "I" and method in role_methods:
                contexts = [context for context in contexts
                            if model.failure_fraction[context] >= .5 and
                            model.lead_failure_fraction[context] >= .5]
            target_probes = 2 if direction == "I" and method in two_probe_methods else 1
            minimum = min((counts[context] for context in contexts), default=0)
            uncovered = [context for context in contexts
                         if counts[context] == minimum and counts[context] < target_probes]
            if uncovered:
                context = uncovered[0]
                choices = [k for k, case in enumerate(pool) if case["context_id"] == context]
                if direction == "I" and counts[context] == 1 and method in two_probe_methods:
                    first = next(row for row in queries if row["direction"] == direction
                                 and row["context_id"] == context)
                    first_row = model.cases[first["scenario_id"]]["grid_index"][0]
                    separate_row = [k for k in choices
                                    if pool[k]["grid_index"][0] != first_row]
                    if separate_row:
                        choices = separate_row
                index = min(choices, key=lambda k: (
                    margin(parent_rows[pool[k]["scenario_id"]]),
                    pool[k]["scenario_id"]))
                phase = ("parent_margin_context_probe" if direction == "R" else
                         "parent_failure_context_probe")
            elif method not in {"static_margin_coverage", "static_dual_margin_coverage",
                                     "static_dual_coverage2", "static_majority_coverage2",
                                     "static_role_coverage2"}:
                discoveries = [row for row in queries if row["direction"] == direction
                               and row["discovery"]]
                if discoveries:
                    context = discoveries[-1]["context_id"]
                    found = [model.cases[row["scenario_id"]]["grid_index"]
                             for row in discoveries if row["context_id"] == context]
                    frontier = [k for k, case in enumerate(pool)
                                if case["context_id"] == context and any(
                                    abs(case["grid_index"][0] - cell[0]) +
                                    abs(case["grid_index"][1] - cell[1]) == 1
                                    for cell in found)]
                    if frontier:
                        index = min(frontier, key=lambda k: (-scores[k],
                                    margin(parent_rows[pool[k]["scenario_id"]]),
                                    pool[k]["scenario_id"]))
                        phase = ("observed_regression_frontier" if direction == "R" else
                                 "observed_improvement_frontier")
        # Frozen physical identity resolves ordinary model ties.
        if index is None:
            index = min(range(len(pool)), key=lambda k: (-scores[k], pool[k]["scenario_id"]))
        case = pool[index]
        outcome = oracle.query(case["scenario_id"])
        target_label = _label(outcome)
        if target_label is not None:
            observations.append((case, target_label))
        queries.append({"rank": rank, "direction": direction, "method": method,
                        "selection_phase": phase,
                        "scenario_id": case["scenario_id"], "context_id": case["context_id"],
                        "parent_label": model.parents[case["scenario_id"]],
                        "target_label": target_label, "target_valid": target_label is not None,
                        "score_before_query": float(scores[index]),
                        "discovery": int(target_label == (1 if direction == "R" else 0))
                        if target_label is not None else 0,
                        "episode_cost": 1})
        del eligible[case["scenario_id"]]
    return queries


def _direction_summary(queries: list[dict], truth: list[dict], method: str,
                       parent: str, target: str) -> list[dict]:
    rows = []
    for direction, transition in (("R", "REGRESSION"), ("I", "IMPROVEMENT")):
        local = [row for row in queries if row["direction"] == direction]
        successes = [row["discovery"] for row in local]
        total = sum(row["transition"] == transition for row in truth)
        upto = lambda k: sum(successes[:k])
        k = len(local)
        area = (2 * sum(upto(j) for j in range(1, k + 1)) / (k * (k + 1))) if k and total else None
        rows.append({"parent": parent, "target": target, "method": method,
                     "direction": direction, "pool_queries": k,
                     "available_changes": total, "discoveries": sum(successes),
                     "recall": sum(successes) / total if total else None,
                     "early_area": area, "D@1": upto(1), "D@5": upto(5),
                     "D@10": upto(10), "D@20": upto(20),
                     "anomaly_queries": sum(not row["target_valid"] for row in local)})
    return rows


def _regions(cases: list[dict], truth: list[dict], transition: str,
             diagonal: bool = False) -> list[set[str]]:
    """Finite-grid components, never passed to the online selector."""
    wanted = {row["scenario_id"] for row in truth if row["transition"] == transition}
    cells = {(case["context_id"], *case["grid_index"]): case["scenario_id"]
             for case in cases if case["scenario_id"] in wanted}
    pending = set(cells)
    components = []
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        steps += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    while pending:
        start = min(pending)
        pending.remove(start)
        stack = [start]
        component = {cells[start]}
        while stack:
            context, i, j = stack.pop()
            for di, dj in steps:
                neighbor = (context, i + di, j + dj)
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
                    component.add(cells[neighbor])
        components.append(component)
    return components


def evaluate(root: Path, *, budget: int = 40, random_repeats: int = 10,
             methods: tuple[str, ...] | None = None, output_root: Path | None = None,
             primary_method: str = "directed_residual") -> dict:
    cases = _read_jsonl(root / "scenario_manifest.jsonl")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    selected_methods = methods or tuple(protocol["methods"])
    if primary_method not in selected_methods:
        raise ValueError("primary method must be part of the comparison")
    output_root = output_root or root
    builds = tuple(protocol["builds"])
    if len(builds) != 3:
        raise ValueError("expected a three-release protocol")
    pairs = tuple(zip(builds, builds[1:]))
    if hashlib.sha256((root / "scenario_manifest.jsonl").read_bytes()).hexdigest() != protocol[
            "manifest_sha256"]:
        raise ValueError("manifest fingerprint mismatch")
    rows = _read_jsonl(root / "full_response_bank.jsonl")
    bank = {(row["build_id"], row["scenario_id"]): row for row in rows}
    expected = {(build, case["scenario_id"]) for build in builds for case in cases}
    if len(bank) != len(rows) or set(bank) != expected:
        raise ValueError("full response bank is incomplete or duplicated")
    for (build, sid), row in bank.items():
        case = next(case for case in cases if case["scenario_id"] == sid)
        if (row["scenario"] != case or row["build_fingerprint"] !=
                build_spec_factory(build).fingerprint or
                row["simulator_seed"] != protocol["simulator_seed"] or
                row["execution_contract_version"] != EXECUTION_CONTRACT or
                row["scenario_fingerprint"] != stable_hash({
                    "scenario": case, "execution_contract": EXECUTION_CONTRACT})):
            raise ValueError("stale full-bank response")
    all_truth, all_queries, all_summary = [], [], []
    for parent, target in pairs:
        truth = transition_truth(cases, bank, parent, target)
        all_truth.extend(truth)
        parent_rows = {case["scenario_id"]: bank[parent, case["scenario_id"]] for case in cases}
        target_rows = {case["scenario_id"]: bank[target, case["scenario_id"]] for case in cases}
        for method in selected_methods:
            for repeat in range(random_repeats if method == "random" else 1):
                oracle = TargetOracle(target_rows)
                queries = select(cases, parent_rows, oracle, method, budget=budget,
                                 seed=SEED + repeat)
                all_queries.extend({"parent": parent, "target": target, "repeat": repeat,
                                    **row} for row in queries)
                all_summary.extend({"repeat": repeat, **row} for row in
                                   _direction_summary(queries, truth, method, parent, target))
    _write_csv(output_root / "transition_truth.csv", all_truth)
    _write_jsonl(output_root / "queries.jsonl", all_queries)
    _write_csv(output_root / "summary_by_direction.csv", all_summary)
    context_rows = []
    for summary in all_summary:
        parent, target = summary["parent"], summary["target"]
        method, repeat, direction = summary["method"], summary["repeat"], summary["direction"]
        transition = "REGRESSION" if direction == "R" else "IMPROVEMENT"
        for context in sorted({case["context_id"] for case in cases}):
            local_truth = [row for row in all_truth if row["parent_build_id"] == parent
                           and row["context_id"] == context]
            local_queries = [row for row in all_queries if row["parent"] == parent
                             and row["target"] == target and row["method"] == method
                             and row["repeat"] == repeat and row["direction"] == direction
                             and row["context_id"] == context]
            changes = sum(row["transition"] == transition for row in local_truth)
            found = sum(row["discovery"] for row in local_queries)
            context_rows.append({"parent": parent, "target": target, "method": method,
                                 "repeat": repeat, "context_id": context,
                                 "direction": direction, "available_changes": changes,
                                 "queries": len(local_queries), "discoveries": found,
                                 "recall": found / changes if changes else None})
    _write_csv(output_root / "summary_by_context.csv", context_rows)
    ablations = []
    for direction in ("R", "I"):
        for parent, target in pairs:
            local = [row for row in all_summary if row["parent"] == parent and
                     row["target"] == target and row["direction"] == direction and
                     row["repeat"] == 0]
            reference = next(row for row in local if row["method"] == primary_method)
            for row in local:
                ablations.append({"parent": parent, "target": target,
                                  "direction": direction, "method": row["method"],
                                  "primary_method": primary_method,
                                  "available_changes": row["available_changes"],
                                  "discoveries": row["discoveries"],
                                  "delta_discoveries_vs_primary": (
                                      row["discoveries"] - reference["discoveries"]
                                      if row["available_changes"] else None),
                                  "early_area": row["early_area"],
                                  "delta_area_vs_primary": (
                                      row["early_area"] - reference["early_area"]
                                      if row["early_area"] is not None else None)})
    _write_csv(output_root / "ablations.csv", ablations)
    regions = []
    for parent, target in pairs:
        pair_truth = [row for row in all_truth if row["parent_build_id"] == parent]
        for direction, transition in (("R", "REGRESSION"), ("I", "IMPROVEMENT")):
            for adjacency, diagonal in ((4, False), (8, True)):
                components = _regions(cases, pair_truth, transition, diagonal)
                for method in selected_methods:
                    repeats = range(random_repeats if method == "random" else 1)
                    for repeat in repeats:
                        found = {row["scenario_id"] for row in all_queries
                                 if row["parent"] == parent and row["target"] == target
                                 and row["direction"] == direction and row["method"] == method
                                 and row["repeat"] == repeat and row["discovery"]}
                        covered = sum(bool(component & found) for component in components)
                        regions.append({"parent": parent, "target": target,
                                        "direction": direction, "adjacency": adjacency,
                                        "method": method, "repeat": repeat,
                                        "regions": len(components), "covered_regions": covered,
                                        "coverage": covered / len(components) if components else None})
    _write_csv(output_root / "region_coverage.csv", regions)
    _write_json(output_root / "cost_ledger.json", {
        "physical_full_bank_episodes": len(rows), "physical_episodes_new_in_replay": 0,
        "physical_episodes_by_build": dict(Counter(row["build_id"] for row in rows)),
        "logical_queries": len(all_queries), "logical_repeats_reuse_bank": True,
        "logical_queries_by_method": dict(Counter(row["method"] for row in all_queries)),
        "parent_full_bank_available_to_selector": True,
        "middle_release_full_bank_becomes_next_parent": True,
        "selector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_response_bank_sha256": hashlib.sha256(
            (root / "full_response_bank.jsonl").read_bytes()).hexdigest(),
    })
    report = ["# NL-IDM 双向版本变化测试", "",
              f"协议：`{protocol['version']}`；数据身份：`{protocol['split']}`；"
              f"{len(cases)} 个冻结场景，{len(rows)} 次配对物理执行。",
              f"主比较方法：`{primary_method}`；所有方法共用固定方向日程与查询预算。",
              "终点仅为 ego 碰撞；无碰撞须完成执行。以下重复查询只重放冻结银行，"
              "不增加独立物理样本。", "",
              "| 父→子 | 方向 | 真值数 | 方法 | 发现@20 | 早期面积 |", "|---|---|---:|---|---:|---:|"]
    for parent, target in pairs:
        for direction in ("R", "I"):
            for method in selected_methods:
                local = [row for row in all_summary if row["parent"] == parent and
                         row["target"] == target and row["direction"] == direction and
                         row["method"] == method]
                changes = local[0]["available_changes"]
                discoveries = sum(row["discoveries"] for row in local) / len(local)
                area = (sum(row["early_area"] for row in local) / len(local)
                        if changes else None)
                report.append(f"| {parent}→{target} | {direction} | {changes} | {method} | "
                              f"{discoveries:.2f} | {area:.3f} |" if area is not None else
                              f"| {parent}→{target} | {direction} | {changes} | {method} | "
                              f"{discoveries:.2f} | NA |")
    report.extend(["", "R=旧通过→新失败；I=旧失败→新通过。真值数为零时早期面积与"
                   "相对收益记作 NA。`transition_truth.csv` 仅供评估，不供选择器使用。",
                   "相邻网格点、同一 release chain 和选择器随机重复不构成独立样本；"
                   "本表不提供显著性结论。", ""])
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return {"truth": {state: sum(row["transition"] == state for row in all_truth)
                       for state in ("REGRESSION", "IMPROVEMENT", "UNKNOWN")},
            "summary_rows": len(all_summary), "logical_queries": len(all_queries)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "measure", "evaluate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("development", "confirmation", "calibration",
                                             "development2", "confirmation2",
                                             "development3", "confirmation3",
                                             "confirmation4", "ppo_confirmation",
                                             "cross_sut_confirmation"),
                        default="development")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--builds", nargs=3, default=list(BUILDS))
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget", type=int, default=40)
    args = parser.parse_args()
    methods = METHODS + EXPLORATORY_METHODS if args.include_exploratory else METHODS
    if args.action == "freeze":
        result = {"scenarios": len(freeze(args.root, args.split, args.resolution,
                                          builds=tuple(args.builds), methods=methods))}
    elif args.action == "measure":
        result = measure(args.root, limit=args.limit)
    else:
        result = evaluate(args.root, budget=args.budget,
                          methods=methods if args.include_exploratory else None,
                          output_root=(args.root / "exploratory") if args.include_exploratory
                          else None,
                          primary_method="directed_context_laplace" if args.include_exploratory
                          else "directed_residual")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
