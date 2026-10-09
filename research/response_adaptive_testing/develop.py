"""Logged development on the already measured original target pool."""
import itertools
import time

import numpy as np
import torch

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import predict

from .config import OUTPUT
from .session import AdaptiveTestingSession


def metrics(collision, selected):
    curve = np.cumsum(collision[selected])
    total = int(np.sum(collision))
    return {
        "mean_cumulative_collisions": float(curve.mean()),
        "F50": int(curve[49]),
        "F100": int(curve[99]),
        "F200": int(curve[-1]),
        "pool_collisions": total,
        "recall": float(curve[-1] / total) if total else 1.0,
        "all_failures_found": bool(curve[-1] == total),
        "curve": curve.tolist()
    }


def settings():
    for transform, strength, noise, correlation in itertools.product(
        ("raw", "logit"), (0.1, 1.0), (0.0025, 0.05), (0, 0.8)):
        yield {
            "transform": transform,
            "source_scale": strength,
            "noise": noise,
            "risk_scale": 0.25 if transform == "raw" else 1.0,
            "collision_scale": 0.25,
            "length": 0.05 if transform == "raw" else 0.5,
            "correlation": correlation,
            "lookahead": False
        }


def run_one(x, risk, collision, predictions, options):
    session = AdaptiveTestingSession(x, *predictions, options)
    selected = []
    started = time.perf_counter()
    while (index := session.next_index()) is not None:
        # Only this queried continuous risk is disclosed to the selector.
        session.observe(float(risk[index]))
        selected.append(index)
    return {
        **metrics(collision, selected), "selected_indices": selected,
        "queries": session.records,
        "options": options,
        "elapsed_s": time.perf_counter() - started,
        "stage": "development; previously disclosed full target pool"
    }


def main():
    torch.set_num_threads(1)
    bank = np.load(BASELINE / "target/responses.npz")
    predictions = predict(bank["x"], 11)
    rows = []
    for index, options in enumerate(settings()):
        path = OUTPUT / "development" / f"joint_{index:03d}.json"
        if path.exists():
            result = read_json(path)
        else:
            result = run_one(bank["x"], bank["risk"], bank["collision"],
                             predictions, options)
            write_json(path, result)
        rows.append({
            "path": str(path.relative_to(OUTPUT)),
            "options": options,
            **{
                name: result[name]
                for name in ("mean_cumulative_collisions", "F50", "F100", "F200", "recall")
            }
        })
        print(index,
              round(result["mean_cumulative_collisions"], 3),
              result["F100"],
              result["F200"],
              flush=True)
    write_json(OUTPUT / "development" / "summary.json", rows)


if __name__ == "__main__":
    main()
