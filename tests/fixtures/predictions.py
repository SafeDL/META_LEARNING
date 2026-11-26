"""Two-ADS prediction grid for paired reporting tests."""

from itertools import product

import numpy as np

from experiments.k_shot import prediction_metrics

DESIGN = {
    "ads": {"ads-a": "rule", "ads-b": "learning"},
    "scenes": ("zero", "positive"),
    "seeds": (11, 23),
    "methods": ("target_only", "full"),
    "candidate_count": 10,
    "replicates": 100,
}


def prediction_rows():
    rows = []
    for sut, scene, seed, method, count in product(
        DESIGN["ads"], DESIGN["scenes"], DESIGN["seeds"], DESIGN["methods"], (0, 1)
    ):
        labels = np.zeros(10, dtype=bool)
        if scene == "positive":
            labels[:2] = True
        probability = np.full(10, 0.2 if method == "target_only" else 0.1)
        metrics, _ = prediction_metrics(
            probability, labels, [str(i) for i in range(10)], list(range(1, 10))
        )
        rows.append(
            {
                "sut_id": sut,
                "scenario": scene,
                "seed": seed,
                "method": method,
                "K": count,
                "status": "complete",
                "candidate_count": 10,
                "pool_collisions": int(labels.sum()),
                **metrics,
            }
        )
    return rows
