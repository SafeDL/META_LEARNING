"""Held-out development evaluation of the three budget-training controls."""
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
from .train_budget import CHECKPOINTS, CONTROLS, RESULTS


def summarize(records, names):
    models = {}
    for control in CONTROLS:
        for steps in CHECKPOINTS:
            method = f"{control}_{steps}"
            profiles, differences = [], []
            for name in names:
                rows = [
                    row for row in records
                    if row["profile"] == name and row["method"] == method
                ]
                assert len(rows) == 2 * len(SEEDS)
                profiles.append({
                    "profile": name,
                    **{
                        key: float(np.mean([row[key] for row in rows]))
                        for key in ("normalized_area", "recall", "F200")
                    }
                })
                differences.append(
                    {
                        "profile": name,
                        **{
                            key:
                            float(
                                np.mean([
                                    row[f"difference_{key}"] for row in rows
                                ]))
                            for key in ("normalized_area", "recall", "F200")
                        }
                    })
            models[method] = {
                "profiles": profiles,
                "differences": differences,
                "mean_difference": {
                    key: float(np.mean([row[key] for row in differences]))
                    for key in ("normalized_area", "recall", "F200")
                }
            }
    return {
        "role": "Held-out development, not independent confirmation",
        "models": models
    }


def main():
    torch.set_num_threads(1)
    first = read_json(CONFIRMATION / "summary.json")
    assert first["success"] is False
    verify_lock(first["protocol"])
    protocol = read_json(RESULTS / "protocol.json")
    models = {
        (control, steps):
        read_json(RESULTS / "models" / f"{control}_{steps}.json")
        for control in CONTROLS
        for steps in CHECKPOINTS
    }
    records = []
    for name in protocol["validation_profiles"]:
        for replicate in range(protocol["replicates_per_profile"]):
            pool = CONFIRMATION / name / f"pool_{replicate}"
            with np.load(pool / "responses.npz") as bank:
                for seed in SEEDS:
                    prediction = predict(bank["x"], seed)
                    baseline = read_json(pool / "selection" /
                                         f"previous_best_{seed}.json")
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
                                "Held-out development; not prospective confirmation"
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
                        area = measured[
                            "mean_cumulative_collisions"] / total if total else 0
                        baseline_area = baseline[
                            "mean_cumulative_collisions"] / total if total else 0
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
                            area,
                            "difference_normalized_area":
                            area - baseline_area,
                            "difference_recall":
                            measured["recall"] - baseline["recall"],
                            "difference_F200":
                            measured["F200"] - baseline["F200"]
                        })
                        print("BUDGET VALIDATION",
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
                            "Held-out development in failed first cohort",
                            "records":
                            records,
                            "complete":
                            len(records)
                            == len(protocol["validation_profiles"]) *
                            protocol["replicates_per_profile"] * len(SEEDS) *
                            len(models)
                        })
    comparison = summarize(records, protocol["validation_profiles"])
    write_json(RESULTS / "validation_comparison.json", comparison)
    for method, result in comparison["models"].items():
        print("DEVELOPMENT DIFFERENCES",
              method,
              result["mean_difference"],
              flush=True)


if __name__ == "__main__":
    main()
