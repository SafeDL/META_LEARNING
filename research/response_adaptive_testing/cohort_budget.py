"""Check the selected budget-trained model on the full development cohort."""
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
from .train_budget import RESULTS

CONTROLS = ("budget_mean", "uniform_mean", "budget_static", "budget_local")


def main():
    torch.set_num_threads(1)
    first = read_json(CONFIRMATION / "summary.json")
    assert first["success"] is False
    verify_lock(first["protocol"])
    training = read_json(RESULTS / "protocol.json")
    budget = read_json(RESULTS / "models" / "replay_budget_mean_200.json")
    uniform = read_json(RESULTS / "models" / "replay_uniform_mean_200.json")
    choice = {
        "role":
        "Full development-cohort diagnosis after held-out checkpoint choice",
        "selected_method": "replay_budget_mean_200",
        "reason":
        "Improves both held-out development indicators against the prior best and the matched ordinary-ranking fit; learned loading reduces endpoint recall",
        "controls": CONTROLS,
        "feedback": "Queried continuous risk only",
        "training_profiles": training["training_profiles"],
        "validation_profiles": training["validation_profiles"]
    }
    write_json(RESULTS / "cohort_choice.json", choice)
    records = []
    for profile in first["protocol"]["profiles"]:
        name = profile["name"]
        for replicate in range(first["protocol"]["replicates_per_profile"]):
            pool = CONFIRMATION / name / f"pool_{replicate}"
            with np.load(pool / "responses.npz") as bank:
                for seed in SEEDS:
                    prediction = predict(bank["x"], seed)
                    baseline = read_json(pool / "selection" /
                                         f"previous_best_{seed}.json")
                    for control in CONTROLS:
                        output = RESULTS / "cohort" / name / f"pool_{replicate}" / f"{control}_{seed}.json"
                        fit = "replay_uniform_mean" if control == "uniform_mean" else "replay_budget_mean"
                        validation = RESULTS / "validation" / name / f"pool_{replicate}" / f"{fit}_200_{seed}.json"
                        if output.exists():
                            result = read_json(output)
                        elif control in ("budget_mean", "uniform_mean"
                                         ) and validation.exists():
                            result = read_json(validation)
                            write_json(output, result)
                        else:
                            started = time.perf_counter()
                            state = uniform if control == "uniform_mean" else budget
                            options = {**state["options"]}
                            if control == "budget_local":
                                options["source_scale"] = 0
                            session = AdaptiveTestingSession(
                                bank["x"], *prediction, options)
                            apply_projection(session, bank["x"], prediction,
                                             state)
                            selected, observations = [], []
                            if control == "budget_static":
                                selected = torch.argsort(
                                    session.probabilities(),
                                    descending=True,
                                    stable=True)[:BUDGET].tolist()
                                observations = [{
                                    "index":
                                    index,
                                    "continuous_risk":
                                    float(bank["risk"][index])
                                } for index in selected]
                            else:
                                while (index :=
                                       session.next_index()) is not None:
                                    value = float(bank["risk"][index])
                                    session.observe(value)
                                    selected.append(index)
                                    observations.append({
                                        "index":
                                        index,
                                        "continuous_risk":
                                        value
                                    })
                            result = {
                                **metrics(bank["collision"], selected), "selected_indices":
                                selected,
                                "observations":
                                observations,
                                "control":
                                control,
                                "elapsed_s":
                                time.perf_counter() - started,
                                "role":
                                "Full previously disclosed development cohort"
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
                        base_area = baseline[
                            "mean_cumulative_collisions"] / total if total else 0
                        records.append({
                            "profile":
                            name,
                            "replicate":
                            replicate,
                            "seed":
                            seed,
                            "method":
                            control,
                            "split":
                            "training" if name in training["training_profiles"]
                            else "validation",
                            **measured, "normalized_area":
                            area,
                            "difference_normalized_area":
                            area - base_area,
                            "difference_recall":
                            measured["recall"] - baseline["recall"],
                            "difference_F200":
                            measured["F200"] - baseline["F200"]
                        })
                        print("BUDGET COHORT",
                              name,
                              replicate,
                              seed,
                              control,
                              round(measured["mean_cumulative_collisions"], 3),
                              measured["F200"],
                              flush=True)
                    write_json(
                        RESULTS / "cohort_summary.json", {
                            "role":
                            "Development only; training and held-out profiles labeled",
                            "records":
                            records,
                            "complete":
                            len(records)
                            == len(first["protocol"]["profiles"]) *
                            first["protocol"]["replicates_per_profile"] *
                            len(SEEDS) * len(CONTROLS)
                        })
    comparison = {}
    for split in ("training", "validation", "all"):
        comparison[split] = {}
        for control in CONTROLS:
            rows = [
                row for row in records if row["method"] == control and (
                    split == "all" or row["split"] == split)
            ]
            comparison[split][control] = {
                key: float(np.mean([row[f"difference_{key}"] for row in rows]))
                for key in ("normalized_area", "recall", "F200")
            }
    write_json(
        RESULTS / "cohort_comparison.json", {
            "role": "Development diagnosis, not independent significance",
            "mean_differences": comparison
        })
    print("FULL COHORT DIFFERENCES", comparison, flush=True)


if __name__ == "__main__":
    main()
