"""Auditable S01 historical failure primitives on the four active axes."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from methods.failure_memory_regression.fm2_schema import SOURCES, TARGET, align_scenario, valid_label
from methods.failure_memory_regression.schema import PatternCard, stable_hash


def _spread_representatives(points: np.ndarray, indices: list[int], limit: int = 4) -> list[int]:
    if not indices:
        return []
    chosen = [min(indices)]
    while len(chosen) < min(limit, len(indices)):
        remaining = [i for i in indices if i not in chosen]
        chosen.append(max(remaining, key=lambda i: min(float(np.linalg.norm(points[i] - points[j]))
                                                    for j in chosen)))
    return chosen


def build_memory(rows: list[dict], candidate_coords: np.ndarray, *, exclude: str,
                 limit: int = 64) -> list[PatternCard]:
    if exclude not in (*SOURCES, TARGET):
        raise ValueError("unknown excluded build")
    allowed = set(SOURCES) - {exclude}
    clean = [r for r in rows if r.get("build_id") in allowed
             and r.get("template_id") == "fbrt_cutin" and valid_label(r) is not None
             and r.get("visibility") != "evaluator_only"]
    if any(r.get("build_id") == TARGET for r in clean):
        raise AssertionError("target leaked into historical memory")
    # Historical geometry uses the spacing of the measured source cases. A
    # denser unmeasured target candidate pool must not split source evidence.
    unique_scenes = {r["scenario_id"]: r["scenario"] for r in clean}
    source_coords = np.stack([align_scenario(scene, historical=True).values
                              for scene in unique_scenes.values()]) if unique_scenes else candidate_coords
    distance = np.linalg.norm(source_coords[:, None] - source_coords[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    h = float(np.median(np.min(distance, axis=1))) if len(distance) > 1 else 1.0
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in clean:
        key = (row["build_id"], row.get("collision_partner_role") or "UNKNOWN",
               row.get("observed_maneuver_phase") or "UNKNOWN") if valid_label(row) == 1 else (
                   row["build_id"], "PASS", "PASS")
        groups[key].append(row)
    cards = []
    for source in sorted(allowed):
        passes = [r for r in clean if r["build_id"] == source and valid_label(r) == 0]
        pass_points = np.stack([align_scenario(r["scenario"], historical=True).values
                                for r in passes]) if passes else np.empty((0, 4))
        source_failures = sum(valid_label(r) == 1 for r in clean if r["build_id"] == source)
        if source_failures == 0 and passes:
            center = pass_points.mean(axis=0)
            card = PatternCard(
                pattern_id="pat-" + stable_hash({"source": source, "pass_only": True})[:16],
                template_id="fbrt_cutin", context_id="S01:research_v3:base",
                semantic_key=("fbrt_cutin", "pass_only", source), center=center.tolist(),
                radius=float(np.max(np.linalg.norm(pass_points - center, axis=1))),
                failure_record_ids=[], pass_contrast_record_ids=[r["execution_id"] for r in passes],
                boundary_edges=[], occurrence_by_build={source: {"failures": 0, "passes": len(passes)}},
                created_in_session="fm2_s01_history", evidence_status="pass_only_coverage",
                source_build_id=source, scenario_parameters={})
            cards.append(card)
        for (build, partner, phase), failures in groups.items():
            if build != source or partner == "PASS":
                continue
            points = np.stack([align_scenario(r["scenario"], historical=True).values
                               for r in failures])
            pairwise = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            np.fill_diagonal(pairwise, np.inf)
            neighbors = [set(np.argsort(pairwise[i])[:min(4, len(points)-1)].tolist())
                         for i in range(len(points))]
            unseen = set(range(len(points)))
            while unseen:
                root = min(unseen)
                unseen.remove(root)
                stack, component = [root], []
                while stack:
                    i = stack.pop()
                    component.append(i)
                    for j in sorted(neighbors[i] & unseen):
                        if i in neighbors[j] and pairwise[i, j] <= 1.5 * h:
                            unseen.remove(j)
                            stack.append(j)
                component.sort()
                selected = _spread_representatives(points, component)
                contrasts, edges = [], []
                for i in selected:
                    if len(pass_points):
                        j = int(np.argmin(np.linalg.norm(pass_points - points[i], axis=1)))
                        if np.linalg.norm(pass_points[j] - points[i]) <= 2 * h:
                            contrasts.append(passes[j]["execution_id"])
                            edges.append((failures[i]["execution_id"], passes[j]["execution_id"]))
                local = points[component]
                center = local.mean(axis=0)
                ids = [failures[i]["execution_id"] for i in component]
                first = failures[component[0]]
                cards.append(PatternCard(
                    pattern_id="pat-" + stable_hash({"source": source, "failures": ids})[:16],
                    template_id="fbrt_cutin", context_id="S01:research_v3:base",
                    semantic_key=("fbrt_cutin", partner, phase), center=center.tolist(),
                    radius=float(max(np.linalg.norm(local - center, axis=1), default=0)),
                    failure_record_ids=ids, pass_contrast_record_ids=sorted(set(contrasts)),
                    boundary_edges=edges,
                    occurrence_by_build={source: {"failures": len(ids), "passes": len(set(contrasts))}},
                    created_in_session="fm2_s01_history",
                    evidence_status="contrast_available" if contrasts else "observed_region",
                    observed_partner_roles=[] if partner == "UNKNOWN" else [partner],
                    observed_maneuver_phases=[] if phase == "UNKNOWN" else [phase],
                    measured_margin_fields={"min_ttc": [failures[i].get("min_ttc") for i in component],
                                            "min_clearance": [failures[i].get("min_clearance") for i in component]},
                    source_build_id=source,
                    scenario_parameters=dict(first["scenario"].get("active_parameters", {}))))
    # A prolific source cannot occupy the whole memory set.
    by_source: dict[str, list[PatternCard]] = defaultdict(list)
    for card in cards:
        by_source[card.source_build_id or "UNKNOWN"].append(card)
    ordered = []
    for source in sorted(by_source):
        by_source[source].sort(key=lambda c: (-len(c.failure_record_ids), c.pattern_id))
    while len(ordered) < limit and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(ordered) < limit:
                ordered.append(by_source[source].pop(0))
    return ordered


def memory_arrays(cards: list[PatternCard], rows: list[dict]) -> dict[str, np.ndarray]:
    index = {r["execution_id"]: r for r in rows}
    values, present, spread, contrast, contrast_present, metadata, source = [], [], [], [], [], [], []
    failure_witnesses, failure_mask, pass_witnesses, pass_mask = [], [], [], []
    role_codes, phase_codes = [], []
    for card in cards:
        failure = [index[i] for i in card.failure_record_ids if i in index]
        passed = [index[i] for i in card.pass_contrast_record_ids if i in index]
        fail_points = np.stack([align_scenario(r["scenario"], historical=True).values
                                for r in failure]) if failure else np.empty((0, 4))
        pass_points = np.stack([align_scenario(r["scenario"], historical=True).values
                                for r in passed]) if passed else np.empty((0, 4))
        all_points = fail_points if len(fail_points) else pass_points
        center = all_points.mean(axis=0) if len(all_points) else np.zeros(4)
        valid = np.logical_or.reduce([align_scenario(r["scenario"], historical=True).present
                                      for r in failure + passed]) if failure or passed else np.zeros(4, bool)
        values.append(center)
        present.append(valid)
        spread.append(all_points.std(axis=0) if len(all_points) else np.zeros(4))
        contrast.append(fail_points.mean(axis=0) - pass_points.mean(axis=0)
                        if len(fail_points) and len(pass_points) else np.zeros(4))
        contrast_present.append(np.full(4, bool(len(fail_points) and len(pass_points))))
        ttc = [float(x) for x in card.measured_margin_fields.get("min_ttc", []) if x is not None]
        clearance = [float(x) for x in card.measured_margin_fields.get("min_clearance", []) if x is not None]
        metadata.append([np.log1p(len(failure)), np.log1p(len(passed)),
                         np.log1p(np.median(ttc)) if ttc else 0.0,
                         np.sign(np.median(clearance)) * np.log1p(abs(np.median(clearance)))
                         if clearance else 0.0,
                         float(bool(ttc)), float(bool(clearance)),
                         float(bool(card.observed_partner_roles)),
                         float(bool(card.observed_maneuver_phases))])
        source.append(SOURCES.index(card.source_build_id) if card.source_build_id in SOURCES else 0)
        def witnesses(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            chosen = points[:4]
            padded = np.zeros((4, 4), dtype=np.float32)
            padded[:len(chosen)] = chosen
            return padded, np.asarray([True] * len(chosen) + [False] * (4 - len(chosen)))
        f_points, f_mask = witnesses(fail_points)
        p_points, p_mask = witnesses(pass_points)
        failure_witnesses.append(f_points)
        failure_mask.append(f_mask)
        pass_witnesses.append(p_points)
        pass_mask.append(p_mask)
        role = card.observed_partner_roles[0] if card.observed_partner_roles else "UNKNOWN"
        phase = card.observed_maneuver_phases[0] if card.observed_maneuver_phases else "UNKNOWN"
        role_codes.append({"UNKNOWN": 0, "lead": 1, "rear": 2, "adjacent_front": 3}.get(role, 4))
        phase_codes.append({"UNKNOWN": 0, "before_ego_lane_change": 1,
                            "during_ego_lane_change": 2, "after_ego_lane_change": 3}.get(phase, 4))
    return {"values": np.asarray(values, np.float32).reshape(-1, 4),
            "present": np.asarray(present, bool).reshape(-1, 4),
            "spread": np.asarray(spread, np.float32).reshape(-1, 4),
            "contrast": np.asarray(contrast, np.float32).reshape(-1, 4),
            "contrast_present": np.asarray(contrast_present, np.float32).reshape(-1, 4),
            "metadata": np.asarray(metadata, np.float32).reshape(-1, 8),
            "source": np.asarray(source, np.int64),
            "failure_witnesses": np.asarray(failure_witnesses, np.float32).reshape(-1, 4, 4),
            "failure_mask": np.asarray(failure_mask, bool).reshape(-1, 4),
            "pass_witnesses": np.asarray(pass_witnesses, np.float32).reshape(-1, 4, 4),
            "pass_mask": np.asarray(pass_mask, bool).reshape(-1, 4),
            "role_code": np.asarray(role_codes, np.int64),
            "phase_code": np.asarray(phase_codes, np.int64)}
