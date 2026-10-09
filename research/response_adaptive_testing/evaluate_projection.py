"""Validate response mappings on held-out development SUT profiles."""
import time

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import predict

from .config import BUDGET, SEEDS
from .confirmation import CONFIRMATION, verify_lock
from .develop import metrics
from .response_projection import apply_projection
from .session import AdaptiveTestingSession
from .train_projection import CHECKPOINTS, RESULTS


def main():
    torch.set_num_threads(1)
    first = read_json(CONFIRMATION / "summary.json")
    assert first["success"] is False
    verify_lock(first["protocol"])
    protocol = read_json(RESULTS / "protocol.json")
    configurations = [(control, steps) for control in protocol["controls"]
                      for steps in CHECKPOINTS]
    models = {
        (control, steps):
        read_json(RESULTS / "models" / f"{control}_{steps}.json")
        for control, steps in configurations
    }
    models[("projection_with_control_mean", 400)] = {
        **models[("projection_and_mean", 400)],
        "mean_logits": models[("mean_only", 400)]["mean_logits"]
    }
    models[("mean_with_projection_removed", 400)] = {
        **models[("projection_and_mean", 400)],
        "projection": models[("mean_only", 400)]["projection"]
    }
    records = []
    for name in protocol["validation_profiles"]:
        for replicate in range(first["protocol"]["replicates_per_profile"]):
            path = CONFIRMATION / name / f"pool_{replicate}" / "responses.npz"
            with np.load(path) as bank:
                for seed in SEEDS:
                    prediction = predict(bank["x"], seed)
                    for (control, steps), state in models.items():
                        output = RESULTS / "validation" / name / f"pool_{replicate}" / f"{control}_{steps}_{seed}.json"
                        if output.exists():
                            result = read_json(output)
                        else:
                            started = time.perf_counter()
                            session = AdaptiveTestingSession(
                                bank["x"], *prediction, state["options"])
                            apply_projection(session, bank["x"], prediction,
                                             state)
                            selected, observations = [], []
                            while (index := session.next_index()) is not None:
                                value = float(bank["risk"][index])
                                session.observe(value)
                                selected.append(index)
                                observations.append({
                                    "index": index,
                                    "continuous_risk": value
                                })
                            result = {
                                **metrics(bank["collision"], selected), "selected_indices":
                                selected,
                                "observations":
                                observations,
                                "control":
                                control,
                                "steps":
                                steps,
                                "elapsed_s":
                                time.perf_counter() - started,
                                "role":
                                "Held-out development profiles; not prospective confirmation"
                            }
                            write_json(output, result)
                        selected = result["selected_indices"]
                        assert len(selected) == len(set(selected)) == BUDGET
                        assert result["observations"] == [{
                            "index":
                            index,
                            "continuous_risk":
                            float(bank["risk"][index])
                        } for index in selected]
                        measured = metrics(bank["collision"], selected)
                        for key, value in measured.items():
                            assert result[key] == value
                        total = measured["pool_collisions"]
                        records.append({
                            "profile":
                            name,
                            "replicate":
                            replicate,
                            "seed":
                            seed,
                            "method":
                            f"{control}_{steps}",
                            **measured, "normalized_area":
                            measured["mean_cumulative_collisions"] /
                            total if total else 0
                        })
                        print("MAPPING VALIDATION",
                              name,
                              replicate,
                              seed,
                              control,
                              steps,
                              round(measured["mean_cumulative_collisions"], 3),
                              measured["F200"],
                              flush=True)
                    write_json(
                        RESULTS / "validation_summary.json", {
                            "role":
                            "Validation in failed first cohort; not new confirmation",
                            "records":
                            records,
                            "complete":
                            len(records)
                            == len(protocol["validation_profiles"]) *
                            first["protocol"]["replicates_per_profile"] *
                            len(SEEDS) * len(models)
                        })


if __name__ == "__main__":
    main()
