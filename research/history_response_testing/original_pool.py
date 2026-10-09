"""Retained method on the fully measured original pool for baseline comparison."""
import numpy as np
import torch

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.evaluate import discovery_metrics
from methods.history_guided_testing.io import write_json

from .calibration import calibrators
from .config import OUTPUT, SEEDS, SOURCE_WEIGHTS
from .history_model import predict
from .kernel import covariance
from .session import RiskTestingSession


def main():
    torch.set_num_threads(1)
    bank = np.load(BASELINE / "target/responses.npz")
    calibration = calibrators()
    weights = np.asarray(SOURCE_WEIGHTS)
    for seed in SEEDS:
        path = OUTPUT / "original_pool" / f"candidate_{seed}.json"
        if path.exists():
            continue
        risks, collisions = predict(bank["x"], seed)
        kernel = covariance(bank["x"], risks)
        session = RiskTestingSession(bank["x"], kernel, risks, collisions,
                                     weights, calibration)
        selected = []
        while (index := session.next_index()) is not None:
            session.observe(float(bank["risk"][index]))
            selected.append(index)
        result = discovery_metrics(bank, selected)
        result["mean_cumulative_collisions"] = result.pop("discovery_auc")
        write_json(
            path, {
                **result, "selected_indices": selected,
                "queries": session.records,
                "seed": seed,
                "stage": "original-target development only"
            })
        print("Original pool",
              seed, [
                  result["checkpoints"][str(k)]["collisions"]
                  for k in (50, 100, 200)
              ],
              round(result["mean_cumulative_collisions"], 3),
              flush=True)


if __name__ == "__main__":
    main()
