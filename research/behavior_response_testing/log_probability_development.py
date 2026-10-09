"""Screen stable probability arithmetic with identical response predictors."""
import time

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.response_adaptive_testing.confirmation import CONFIRMATION
from research.response_adaptive_testing.develop import metrics

from .prediction import behavior_grid, response_tables
from .log_probability import LogProbabilityTestingSession
from .model import BehaviorResponseModel
from .session import rmse_discrepancy
from .train import OUTPUT

SEEDS = (11, 23, 37, 53, 71)
METHODS = ("log_behavior", "log_calibrated_behavior", "log_discrete")


def main():
    torch.set_num_threads(1)
    training = read_json(OUTPUT / "protocol.json")
    grid = behavior_grid()
    historical = np.array(list(
        training["historical_behavior_descriptors"].values()),
                          dtype=np.float32)
    records = []
    profiles = [
        p["name"]
        for p in read_json(CONFIRMATION / "protocol.json")["profiles"]
    ]
    write_json(
        OUTPUT / "log_probability" / "protocol.json", {
            "role": "Disclosed development numerical probability check",
            "profiles": profiles,
            "seeds": SEEDS,
            "methods": METHODS,
            "budget": 200,
            "online_feedback": "Queried continuous risk only",
            "ranking":
            "Minimize posterior log noncollision probability, equivalent in exact arithmetic to maximizing collision probability",
            "continuous_particle_count": len(grid)
        })
    for seed in SEEDS:
        states = torch.load(OUTPUT / "models" / f"predictor_{seed}.pt",
                            map_location="cuda",
                            weights_only=True)
        models = []
        for family in (0, 1):
            model = BehaviorResponseModel().cuda().eval()
            model.load_state_dict(states[str(family)])
            models.append(model)
        validation = read_json(
            OUTPUT / "models" /
            f"predictor_{seed}.json")["known_profile_validation"]
        unfitted = rmse_discrepancy(
            [validation[str(f)]["risk_rmse"]**2 for f in (0, 1)])
        fitted = read_json(OUTPUT / "calibration" /
                           f"discrepancy_{seed}.json")["parameters"]
        for name in profiles:
            for replicate in range(2):
                pool = CONFIRMATION / name / f"pool_{replicate}"
                with np.load(pool / "responses.npz") as bank:
                    risks, logits = response_tables(bank["x"],
                                                    grid,
                                                    models,
                                                    return_logits=True)
                    historical_risks, historical_logits = response_tables(
                        bank["x"], historical, models, return_logits=True)
                    for method in METHODS:
                        path = OUTPUT / "log_probability" / name / f"pool_{replicate}" / f"{method}_{seed}.json"
                        if path.exists():
                            result = read_json(path)
                        else:
                            started = time.perf_counter()
                            discrete = method == "log_discrete"
                            session = LogProbabilityTestingSession(
                                bank["x"],
                                historical_risks if discrete else risks,
                                historical_logits if discrete else logits,
                                fitted if method == "log_calibrated_behavior"
                                else unfitted)
                            selected = []
                            while (index := session.next_index()) is not None:
                                session.observe(float(bank["risk"][index]))
                                selected.append(index)
                            result = {
                                **metrics(bank["collision"], selected), "selected_indices":
                                selected,
                                "queries":
                                session.records,
                                "elapsed_s":
                                time.perf_counter() - started
                            }
                            write_json(path, result)
                        baseline = read_json(pool / "selection" /
                                             f"previous_best_{seed}.json")
                        count = result["pool_collisions"]
                        records.append({
                            "profile":
                            name,
                            "replicate":
                            replicate,
                            "seed":
                            seed,
                            "method":
                            method,
                            "difference_normalized_area":
                            (result["mean_cumulative_collisions"] -
                             baseline["mean_cumulative_collisions"]) /
                            count if count else 0,
                            "difference_recall":
                            result["recall"] - baseline["recall"],
                            "difference_F200":
                            result["F200"] - baseline["F200"]
                        })
                        print("LOG PROBABILITY",
                              name,
                              replicate,
                              seed,
                              method,
                              result["F200"],
                              flush=True)
                        write_json(
                            OUTPUT / "log_probability" / "summary.json", {
                                "records":
                                records,
                                "complete":
                                len(records) == len(profiles) * 2 *
                                len(SEEDS) * len(METHODS)
                            })
    comparison = {
        method: {
            key:
            float(
                np.mean([
                    row[f"difference_{key}"] for row in records
                    if row["method"] == method
                ]))
            for key in ("normalized_area", "recall", "F200")
        }
        for method in METHODS
    }
    write_json(OUTPUT / "log_probability" / "comparison.json", comparison)
    print("LOG PROBABILITY DIFFERENCES", comparison, flush=True)


if __name__ == "__main__":
    main()
