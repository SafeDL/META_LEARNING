"""Failure regions, same-system pass contrasts, and compact RBF dictionaries."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from highway_env_benchmark.fbrt_parameters import ACTIVE_PARAMETERS, BOUNDS
from method_chains.failure_memory_regression.schema_v2 import PatternCard, stable_hash
from method_chains.failure_memory_regression.replay_utils import is_parent_pass, is_usable_outcome


NEW_BOUNDS = {
    "fbrt_cutin": ((8.0, 60.0), (1.5, 3.0)),
    "fbrt_cutout_static": ((8.0, 40.0), (2.0, 4.5)),
    "fbrt_lane_change_rear": ((8.0, 60.0), (0.0, 10.0)),
    "fbrt_moving_lead": ((15.0, 90.0), (20.0, 27.0)),
    "fbrt_cutin_then_brake": ((10.0, 60.0), (1.0, 6.0)),
}
PARAMS = {
    **{key: value for key, value in ACTIVE_PARAMETERS.items()},
    "fbrt_cutin": ("initial_clearance_m", "lane_change_time_scale_s"),
    "fbrt_cutout_static": ("initial_clearance_m", "lead_to_static_ttc_start_s"),
    "fbrt_lane_change_rear": ("rear_clearance_m", "rear_closing_speed_mps"),
    "fbrt_moving_lead": ("initial_clearance_m", "lead_speed_mps"),
    "fbrt_cutin_then_brake": ("initial_clearance_m", "lead_deceleration_mps2"),
}
LEGACY_ALIASES = {
    "lane_change_time_scale_s": "lane_change_duration_s",
    "lead_to_static_ttc_start_s": "static_target_ttc_s",
}


def _read_scenario(item: dict) -> dict:
    value = item.get("scenario", item)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("active_parameters"), dict):
        return {**value, **value["fixed_context"], **value["active_parameters"]}
    return value


def active_values(scenario: dict) -> tuple[float, ...]:
    scenario = _read_scenario(scenario)
    template = scenario.get("template_id", "unknown")
    names = PARAMS.get(template)
    if not names:
        return (0.5, 0.5)
    values = []
    for name, (low, high) in zip(names, bounds_for(template)):
        value = scenario.get(name)
        if value is None:
            value = scenario.get(LEGACY_ALIASES.get(name, ""))
        if value is None:
            return (0.5, 0.5)
        values.append((float(value) - low) / (high - low))
    if scenario.get("parameterization_version") == "research_v3_ego_initial":
        declared = scenario.get("research_bounds", {})
        for name in ("ego_initial_speed_mps", "ego_initial_lane_id",
                     "ego_initial_lateral_offset_m", "ego_initial_heading_offset_rad",
                     "ego_initial_x_m"):
            if name not in scenario or name not in declared:
                raise ValueError(f"v3 ego initial state is incomplete: {name}")
            low, high = declared[name]
            values.append((float(scenario[name]) - float(low)) /
                          (float(high) - float(low)))
    # Values outside a newly declared research interval remain outside [0,1];
    # clipping would collapse distinct legacy measurements onto one artificial edge.
    return tuple(float(value) for value in values)


def bounds_for(template: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if template in NEW_BOUNDS:
        # Keep a single coordinate system for historical and new episodes. New
        # `research_v2` rows use the catalogue bounds; legacy rows keep legacy bounds.
        return NEW_BOUNDS[template]
    return BOUNDS[template]


def semantic_key(record: dict) -> tuple[str, ...]:
    scenario = _read_scenario(record)
    template = record.get("template_id") or scenario.get("template_id", "unknown")
    # Coarse interaction semantics only; fault/build names and outcomes are excluded.
    interaction = {
        "fbrt_cutin": "entering_lead",
        "fbrt_cutout_static": "target_switch_static_lead",
        "fbrt_lead_emergency_brake": "lead_braking_to_stop",
        "fbrt_stop_hold_go": "lead_stop_hold_restart",
        "fbrt_lane_change_rear": "ego_lane_change_rear_traffic",
        "fbrt_moving_lead": "same_lane_cruise",
        "fbrt_cutin_then_brake": "entering_lead_then_brake",
    }.get(template, "unknown_interaction")
    context_id = record.get("context_id") or scenario.get("context_id", "legacy_unspecified")
    return (str(template), interaction, str(context_id))


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    norm2 = float(direction @ direction)
    if norm2 <= 1e-15:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(((point - start) @ direction) / norm2, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * direction)))


def build_pattern_cards(records: list[dict], session_id: str = "archive_import") -> list[PatternCard]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("visibility") == "evaluator_only" or not is_usable_outcome(record):
            continue
        key = (record.get("build_id", "unknown"),) + semantic_key(record)
        groups[key].append(record)

    cards: list[PatternCard] = []
    for (build_id, template, _interaction, context_id), group in sorted(groups.items()):
        failures = [item for item in group if item.get("ego_collision") is True]
        passes = [item for item in group if is_parent_pass(item)]
        if not failures:
            continue
        points = np.asarray([active_values(_read_scenario(item)) for item in failures])
        pass_points = np.asarray([active_values(_read_scenario(item)) for item in passes])
        n = len(points)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        neighbors = []
        for i in range(n):
            neighbors.append([j for j in np.argsort(distances[i]) if j != i][:min(3, n - 1)])
        adjacency = [set() for _ in range(n)]
        for i in range(n):
            for j in neighbors[i]:
                if i not in neighbors[j] or distances[i, j] > 0.25:
                    continue
                blocked = any(_segment_distance(p, points[i], points[j]) < 0.05
                              for p in pass_points)
                if not blocked:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        components = []
        unseen = set(range(n))
        while unseen:
            root = min(unseen)
            stack, component = [root], []
            unseen.remove(root)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current] & unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
            components.append(sorted(component))

        for component in components:
            local = points[component]
            center = np.mean(local, axis=0)
            radius = max((float(np.linalg.norm(point - center)) for point in local), default=0.0)
            # Pick up to four farthest-spread failure witnesses before adding
            # same-build nearest passing contrasts.
            reps = [component[0]]
            while len(reps) < min(4, len(component)):
                candidates = [idx for idx in component if idx not in reps]
                reps.append(max(candidates, key=lambda idx: min(
                    float(np.linalg.norm(points[idx] - points[other])) for other in reps)))
            edges: list[tuple[str, str]] = []
            pass_ids: list[str] = []
            if len(pass_points):
                for index in reps:
                    nearest = int(np.argmin(np.linalg.norm(pass_points - points[index], axis=1)))
                    passed = passes[nearest]
                    failed_id = failures[index]["execution_id"]
                    passed_id = passed["execution_id"]
                    edges.append((failed_id, passed_id))
                    pass_ids.append(passed_id)
            failure_ids = [failures[index]["execution_id"] for index in component]
            pattern_id = "pat-" + stable_hash({"build": build_id, "template": template,
                                                "failures": sorted(failure_ids)})[:16]
            statuses = {}
            for other_build in sorted({item.get("build_id", "unknown") for item in records}):
                matched = [item for item in records if item.get("build_id") == other_build
                           and item.get("template_id") == template
                           and (item.get("context_id") or _read_scenario(item).get("context_id", "legacy_unspecified")) == context_id
                           and np.linalg.norm(np.asarray(active_values(_read_scenario(item))) - center)
                           <= max(radius, 0.08)]
                statuses[other_build] = {
                    "observed": len(matched),
                    "failures": sum(item.get("ego_collision") is True for item in matched),
                    "passes": sum(item.get("completed") is True and
                                  item.get("ego_collision") is False for item in matched),
                    "status": ("confirmed_present" if any(item.get("ego_collision") is True
                                                          for item in matched)
                               else "local_pass_evidence" if matched else "unknown"),
                }
            cards.append(PatternCard(
                pattern_id=pattern_id, template_id=template,
                semantic_key=(template, _interaction, context_id), context_id=context_id,
                center=center.tolist(), radius=radius,
                failure_record_ids=failure_ids, pass_contrast_record_ids=sorted(set(pass_ids)),
                boundary_edges=edges, occurrence_by_build=statuses,
                created_in_session=session_id,
                evidence_status="observed_region" if len(component) > 1 else
                "contrast_available" if pass_ids else "singleton",
                observed_partner_roles=sorted({
                    str(failures[index].get("collision_partner_role") or
                        failures[index].get("collision_partner"))
                    for index in component
                    if failures[index].get("collision_partner_role") or
                    failures[index].get("collision_partner")
                }),
            ))
    # Retain cross-source spatial associations without merging any system labels.
    for card in cards:
        related = [other.pattern_id for other in cards
                   if other is not card and other.template_id == card.template_id
                   and other.semantic_key == card.semantic_key
                   and np.linalg.norm(np.asarray(other.center) - np.asarray(card.center)) <= 0.15]
        card.parent_pattern_ids = sorted(related)
    return cards


@dataclass
class RBFDictionary:
    template_id: str
    centers: list[dict]
    coverage_centers: list[dict]
    feature_dim: int = 2
    context_id: str = "legacy_unspecified"
    parameterization_version: str = "legacy_unspecified"
    coordinate_names: tuple[str, ...] = ()
    coordinate_bounds: tuple[tuple[float, float], ...] = ()

    @property
    def feature_ids(self) -> list[str]:
        """Legacy RBF-only IDs; use ordered_feature_ids() for the complete schema."""
        return [item["center_id"] for item in self.centers + self.coverage_centers]

    def ordered_feature_ids(self) -> list[str]:
        coordinates = list(self.coordinate_names)
        if len(coordinates) != self.feature_dim:
            coordinates = [f"coordinate_{index}" for index in range(self.feature_dim)]
        return ["bias", *(f"coord:{name}" for name in coordinates),
                *(f"failure:{item['center_id']}" for item in self.centers),
                *(f"coverage:{item['center_id']}" for item in self.coverage_centers)]

    @property
    def schema_identity(self) -> str:
        return stable_hash({"template_id": self.template_id,
                            "context_id": self.context_id,
                            "parameterization_version": self.parameterization_version,
                            "coordinate_names": list(self.coordinate_names),
                            "coordinate_bounds": [list(bounds) for bounds in self.coordinate_bounds],
                            "feature_dim": self.feature_dim})

    def ordered_feature_specs(self) -> dict[str, dict]:
        specs = {"bias": {"kind": "bias"}}
        names = list(self.coordinate_names)
        if len(names) != self.feature_dim:
            names = [f"coordinate_{index}" for index in range(self.feature_dim)]
        for index, name in enumerate(names):
            feature_id = f"coord:{name}"
            specs[feature_id] = {"kind": "coordinate", "name": name,
                                 "bounds": (list(self.coordinate_bounds[index])
                                            if index < len(self.coordinate_bounds) else None),
                                 "schema_identity": self.schema_identity}
        for prefix, items in (("failure", self.centers), ("coverage", self.coverage_centers)):
            for item in items:
                specs[f"{prefix}:{item['center_id']}"] = {
                    "kind": prefix, "center": [float(value) for value in item["center"]],
                    "bandwidth": float(item["bandwidth"]),
                    "schema_identity": self.schema_identity,
                }
        return specs

    def features(self, scenario: dict) -> np.ndarray:
        z = np.asarray(active_values(_read_scenario(scenario)), dtype=float)
        if len(z) != self.feature_dim:
            raise ValueError(f"mixed parameterization for {self.template_id}: "
                             f"expected {self.feature_dim}, got {len(z)}")
        rbf = []
        for item in self.centers + self.coverage_centers:
            center = np.asarray(item["center"], dtype=float)
            bandwidth = float(item["bandwidth"])
            rbf.append(np.exp(-float(np.sum((z - center) ** 2)) / (2 * bandwidth**2)))
        return np.asarray([1.0, *z.tolist(), *rbf], dtype=float)


def build_dictionaries(records: list[dict], cards: list[PatternCard],
                       candidate_records: list[dict], seed: int = 99017) -> dict[str, RBFDictionary]:
    dictionaries = {}
    templates = sorted({record.get("template_id", "unknown") for record in records + candidate_records})
    for template in templates:
        candidate_contexts = {(item.get("context_id") or
                               _read_scenario(item).get("context_id", "legacy_unspecified"))
                              for item in candidate_records if item.get("template_id") == template}
        source_contexts = {(item.get("context_id") or
                            _read_scenario(item).get("context_id", "legacy_unspecified"))
                           for item in records if item.get("template_id") == template}
        active_contexts = candidate_contexts or source_contexts
        if len(active_contexts) > 1:
            raise ValueError(f"multiple physical contexts need independent models: {template}: {active_contexts}")
        context_id = next(iter(active_contexts), "legacy_unspecified")
        local_cards = [card for card in cards if card.template_id == template and
                       card.context_id == context_id]
        selected: list[PatternCard] = []
        # Keep distinct event semantics first, then cover separated regions.
        for card in local_cards:
            if card.semantic_key not in {item.semantic_key for item in selected}:
                selected.append(card)
        while len(selected) < min(6, len(local_cards)):
            remaining = [card for card in local_cards if card not in selected]
            selected.append(max(remaining, key=lambda card: min(
                np.linalg.norm(np.asarray(card.center) - np.asarray(other.center))
                for other in selected)))
        selected = selected[:6]
        centers = [{"center_id": card.pattern_id, "pattern_id": card.pattern_id,
                    "center": card.center,
                    "bandwidth": float(np.clip(card.radius, 0.10, 0.35)),
                    "evidence_status": card.evidence_status}
                   for card in selected]
        pool = [active_values(_read_scenario(item)) for item in candidate_records
                if item.get("template_id") == template]
        coverage = []
        if pool:
            points = np.unique(np.asarray(pool, dtype=float), axis=0)
            rng = np.random.default_rng(seed + sum(template.encode("utf-8")))
            chosen = [int(rng.integers(len(points)))]
            while len(chosen) < min(4, len(points)):
                distance = np.min(np.linalg.norm(points[:, None, :] - points[chosen][None, :, :],
                                                  axis=2), axis=1)
                distance[chosen] = -1
                chosen.append(int(np.argmax(distance)))
            coverage = [{"center_id": f"coverage-{template}-{index}",
                         "center": points[point_index].tolist(), "bandwidth": 0.30,
                         "evidence_status": "unlabelled_coverage"}
                        for index, point_index in enumerate(chosen)]
        feature_dim = len(pool[0]) if pool else (
            len(active_values(_read_scenario(next(item for item in records
                                                  if item.get("template_id") == template))))
            if any(item.get("template_id") == template for item in records) else 2)
        if any(len(point) != feature_dim for point in pool):
            raise ValueError(f"mixed ego initial parameterization in {template}")
        exemplar = next((item for item in candidate_records + records
                         if item.get("template_id") == template), {})
        exemplar_scenario = _read_scenario(exemplar)
        coordinate_names = tuple(PARAMS.get(template, ()))
        if exemplar_scenario.get("parameterization_version") == "research_v3_ego_initial":
            coordinate_names += tuple(
                name for name in ("ego_initial_speed_mps", "ego_initial_lane_id",
                                  "ego_initial_lateral_offset_m", "ego_initial_heading_offset_rad",
                                  "ego_initial_x_m")
                if name in exemplar_scenario and name in exemplar_scenario.get("research_bounds", {}))
        coordinate_names = coordinate_names[:feature_dim]
        if len(coordinate_names) < feature_dim:
            coordinate_names += tuple(f"coordinate_{index}" for index in
                                      range(len(coordinate_names), feature_dim))
        coordinate_bounds = tuple(tuple(float(value) for value in bound)
                                  for bound in bounds_for(template)[:min(feature_dim, 2)])
        if len(coordinate_bounds) < feature_dim:
            declared = exemplar_scenario.get("research_bounds", {})
            extra_bounds = tuple(tuple(float(value) for value in declared[name])
                                 for name in coordinate_names[len(coordinate_bounds):]
                                 if name in declared)
            coordinate_bounds += extra_bounds
        if len(coordinate_bounds) < feature_dim:
            coordinate_bounds += tuple((0.0, 1.0) for _ in
                                       range(feature_dim - len(coordinate_bounds)))
        dictionaries[template] = RBFDictionary(
            template, centers, coverage, feature_dim,
            context_id=context_id,
            parameterization_version=str(exemplar_scenario.get(
                "parameterization_version", "legacy_unspecified")),
            coordinate_names=coordinate_names, coordinate_bounds=coordinate_bounds)
    return dictionaries


def new_failure_card(record: dict, dictionaries: dict[str, RBFDictionary],
                     session_id: str, max_new: int = 4) -> tuple[PatternCard, bool]:
    template = record["template_id"]
    scenario = _read_scenario(record)
    z = np.asarray(active_values(scenario), dtype=float)
    dictionary = dictionaries[template]
    existing = [np.asarray(item["center"], dtype=float) for item in dictionary.centers]
    is_new_region = not existing or min(np.linalg.norm(z - center) for center in existing) > 0.20
    pattern_id = "pat-" + stable_hash({"session": session_id,
                                       "execution_id": record["execution_id"]})[:16]
    card = PatternCard(
        pattern_id=pattern_id, template_id=template,
        semantic_key=semantic_key(record), context_id=record.get("context_id") or scenario.get(
            "context_id", "legacy_unspecified"), center=z.tolist(), radius=0.0,
        failure_record_ids=[record["execution_id"]], pass_contrast_record_ids=[],
        boundary_edges=[], occurrence_by_build={record["build_id"]: {
            "observed": 1, "failures": 1, "passes": 0, "status": "new_region_observed"}},
        created_in_session=session_id, evidence_status="singleton",
        observed_partner_roles=([str(record.get("collision_partner_role") or
                                     record.get("collision_partner"))]
                               if record.get("collision_partner_role") or
                               record.get("collision_partner") else []),
    )
    if is_new_region and sum(item.get("session_added", False)
                             for item in dictionary.centers) < max_new:
        dictionary.centers.append({"center_id": pattern_id, "pattern_id": pattern_id,
                                   "center": z.tolist(), "bandwidth": 0.15,
                                   "evidence_status": "new_region_observed",
                                   "session_added": True})
        return card, True
    return card, False


def cards_jsonl(cards: list[PatternCard], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for card in cards:
            handle.write(json.dumps(card.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
