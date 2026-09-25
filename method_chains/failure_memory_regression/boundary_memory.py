"""Local pairs of actual reference failures and nearby passes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from highway_env_benchmark.envs.fbrt_scenarios import BOUNDS, FBRTScenario


def coordinates(scenario: FBRTScenario) -> np.ndarray:
    bounds = BOUNDS[scenario.template_id]
    values = scenario.active_values()
    return np.asarray([(value - low) / (high - low)
                       for value, (low, high) in zip(values, bounds)], dtype=float)


@dataclass(frozen=True)
class Patch:
    patch_id: str
    template_id: str
    failed_id: str
    passed_id: str
    failed: np.ndarray
    passed: np.ndarray

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.passed - self.failed))

    @property
    def midpoint(self) -> np.ndarray:
        return (self.failed + self.passed) / 2

    def geometry(self, point: np.ndarray) -> tuple[float, float]:
        normal = (self.passed - self.failed) / self.width
        difference = point - self.midpoint
        u = float(normal @ difference)
        v = float(np.linalg.norm(difference - u * normal))
        return u, v

    def as_record(self) -> dict:
        return {"patch_id": self.patch_id, "template_id": self.template_id,
                "failed_id": self.failed_id, "passed_id": self.passed_id,
                "failed": self.failed.tolist(), "passed": self.passed.tolist(),
                "width": self.width, "boundary_estimate": self.midpoint.tolist()}


def build_patches(scenarios: list[FBRTScenario], source: dict[str, dict]) -> list[Patch]:
    patches = []
    for template in BOUNDS:
        failures = [scene for scene in scenarios if scene.template_id == template
                    and source[scene.scenario_id]["ego_collision"]]
        passes = [scene for scene in scenarios if scene.template_id == template
                  and source[scene.scenario_id]["completed"]
                  and not source[scene.scenario_id]["ego_collision"]]
        pairs = []
        for failed in failures:
            if passes:
                passed = min(passes, key=lambda scene: np.linalg.norm(
                    coordinates(scene) - coordinates(failed)))
                distance = float(np.linalg.norm(coordinates(passed) - coordinates(failed)))
                pairs.append((distance, failed, passed))
        kept: list[tuple[np.ndarray, np.ndarray]] = []
        for _, failed, passed in sorted(pairs, key=lambda item: (item[0], item[1].scenario_id)):
            failed_point = coordinates(failed)
            passed_point = coordinates(passed)
            if any(np.allclose(failed_point, old_failed) and
                   np.allclose(passed_point, old_passed)
                   for old_failed, old_passed in kept):
                continue
            kept.append((failed_point, passed_point))
            patches.append(Patch(f"{template}:patch:{len(kept) - 1}", template,
                                 failed.scenario_id, passed.scenario_id,
                                 failed_point, passed_point))
            if len(kept) == 8:
                break
    return patches


def nearest_patch(scenario: FBRTScenario, patches: list[Patch]) -> tuple[Patch | None, float, float, float]:
    point = coordinates(scenario)
    candidates = []
    for patch in patches:
        if patch.template_id != scenario.template_id:
            continue
        if min(np.linalg.norm(point - patch.failed),
               np.linalg.norm(point - patch.passed)) > max(3 * patch.width, 0.25):
            continue
        u, v = patch.geometry(point)
        h_n = max(patch.width, 0.03)
        h_t = max(2 * patch.width, 0.10)
        score = float(np.exp(-abs(u) / h_n - v * v / (2 * h_t * h_t)))
        candidates.append((score, patch, u, v))
    if not candidates:
        return None, 0.0, 0.0, 0.0
    score, patch, u, v = max(candidates, key=lambda item: (item[0], item[1].patch_id))
    return patch, score, u, v
