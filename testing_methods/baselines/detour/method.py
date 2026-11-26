"""DETOUR's official prioritizer with physical scenario features instead of road geometry."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist

from experiments.formal_scenarios import FREE_BOUNDS

from .core import DETOUR


@dataclass
class Test:
    index: int
    point: np.ndarray
    is_failing: bool
    is_selectable: bool


class ParameterClusterer:
    def cluster(self, tests):
        distances = pdist(np.stack([test.point for test in tests]), metric="euclidean")
        return to_tree(linkage(distances, method="ward")), distances


def parameter_features(scene):
    values = [(scene.parameters[name] - low) / (high - low)
              for name, (low, high) in FREE_BOUNDS[scene.kind].items()]
    if scene.categories:
        values.append(float(scene.categories["side"] == "right"))
    return np.array(values)


def prioritize(history, candidates, initial_feedback, count):
    # Chronological insertion retains the most recent actually executed response.
    known = {}
    for scene, collision in history:
        known[scene.coordinate()] = Test(-1, parameter_features(scene), bool(collision), False)
    for index, feedback in initial_feedback:
        scene = candidates[index]
        known[scene.coordinate()] = Test(index, parameter_features(scene), feedback.collision, False)
    if not any(test.is_failing for test in known.values()):
        raise ValueError("Official DETOUR retrieval requires at least one known failing test")
    initial = {index for index, _ in initial_feedback}
    remaining = [Test(i, parameter_features(scene), False, True)
                 for i, scene in enumerate(candidates) if i not in initial]
    selector = DETOUR(list(known.values()), remaining, ParameterClusterer(), random_seed=0)
    # Official prioritize uses integer truncation; use the next representable ratio
    # to obtain exactly the registered count after floating-point multiplication.
    ratio = np.nextafter(count / len(remaining), np.inf)
    selected = selector.prioritize(ratio)
    if len(selected) != count or len({t.index for t in selected}) != count:
        raise ValueError("DETOUR did not produce the requested unique static prefix")
    return [test.index for test in selected]
