"""Read the fixed FVDM candidate bank used by the A-to-D comparison."""

from __future__ import annotations

import numpy as np

from highway_sim_env.s01_parameters import valid_label
from methods.ras_frt_uq.protocol import RESULTS_ROOT, read_jsonl


ROOT = RESULTS_ROOT / "s01_uniform_fvdm_speed_23_mps"
TARGET = "fvdm_safety_speed_23_mps"


def target_labels(scenarios: list[dict]) -> np.ndarray:
    responses = {
        row["scenario_id"]: row
        for row in read_jsonl(ROOT / f"{TARGET}.jsonl")
    }
    labels = [valid_label(responses[scene["scenario_id"]]) for scene in scenarios]
    if any(label is None for label in labels):
        raise ValueError("D contains an inconclusive FVDM response")
    return np.asarray(labels, dtype=np.int8)
