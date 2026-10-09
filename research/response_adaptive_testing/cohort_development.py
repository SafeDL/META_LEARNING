"""Evaluate a learned reference mean only after a failed cohort becomes development."""
import time

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import SOURCE_NAMES, predict

from .config import BUDGET, OUTPUT
from .confirmation import CONFIRMATION, verify_lock
from .develop import metrics
from .initial_prior import coordinates, reference_means
from .session import AdaptiveTestingSession

RESULTS = OUTPUT / "cohort_development"
CONFIGURATIONS = ("learned_mean_full", "learned_mean_local",
                  "learned_mean_source")


def select(x, prediction, options, learned, risk_feedback):
    session = AdaptiveTestingSession(x, *prediction, options)
    family, risk, collision = coordinates(x, *prediction)
    logits = risk.new_tensor(learned["mean_logits"])
    session.mean_r, session.mean_c = reference_means(
        family, risk, collision, list(range(len(SOURCE_NAMES))), logits,
        risk.new_tensor(options["collision_offset"]))
    selected, observations = [], []
    while (index := session.next_index()) is not None:
        value = float(risk_feedback(index))
        session.observe(value)
        selected.append(index)
        observations.append({"index": index, "continuous_risk": value})
    assert len(selected) == len(set(selected)) == BUDGET
    return selected, observations


def main():
    torch.set_num_threads(1)
    summary = read_json(CONFIRMATION / "summary.json")
    assert summary[
        "success"] is False, "A passed confirmation is not a failed development cohort"
    protocol = summary["protocol"]
    verify_lock(protocol)
    learned = read_json(OUTPUT /
                        "initial_prior/models/collision_mean_600.json")
    assert learned["target_data_used"] is False
    assert learned["base_options"] == protocol["candidate_options"]
    options = protocol["candidate_options"]
    configurations = {
        "learned_mean_full": options,
        "learned_mean_local": {
            **options, "source_scale": 0
        },
        "learned_mean_source": {
            **options, "positive_loading": False,
            "loading": [[0] * 6] * 2
        }
    }
    write_json(
        RESULTS / "protocol.json", {
            "role":
            "Failed prospective cohort is now development; no independent significance claim",
            "failed_round":
            protocol["round"],
            "configurations":
            CONFIGURATIONS,
            "learned_prior":
            "Historical collision mean checkpoint 600, selected on earlier development pools",
            "prediction_feedback":
            "Only selected risk; no true collision in policy inputs",
            "next_confirmation":
            "Fresh parameter profiles and coordinates; next alpha is 0.05/2**(round+1)"
        })
    records = []
    for profile in protocol["profiles"]:
        for replicate in range(protocol["replicates_per_profile"]):
            folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
            with np.load(folder / "responses.npz") as bank:
                for seed in protocol["seeds"]:
                    paths = {
                        name:
                        RESULTS / profile["name"] / f"pool_{replicate}" /
                        f"{name}_{seed}.json"
                        for name in CONFIGURATIONS
                    }
                    prediction = (predict(bank["x"], seed) if not all(
                        path.exists() for path in paths.values()) else None)
                    for name, session_options in configurations.items():
                        path = paths[name]
                        if path.exists():
                            result = read_json(path)
                        else:
                            started = time.perf_counter()
                            selected, observations = select(
                                bank["x"], prediction, session_options,
                                learned, lambda index: bank["risk"][index])
                            result = {
                                **metrics(bank["collision"], selected), "selected_indices":
                                selected,
                                "observations":
                                observations,
                                "options":
                                session_options,
                                "mean_model":
                                learned,
                                "elapsed_s":
                                time.perf_counter() - started,
                                "role":
                                "development after failed confirmation"
                            }
                            write_json(path, result)
                        selected = result["selected_indices"]
                        assert len(selected) == len(
                            set(selected)) == protocol["budget"]
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
                            profile["name"],
                            "controller":
                            profile["controller"],
                            "replicate":
                            replicate,
                            "seed":
                            seed,
                            "method":
                            name,
                            **measured, "normalized_area":
                            measured["mean_cumulative_collisions"] /
                            total if total else 0
                        })
                        print("FAILED COHORT DEVELOPMENT",
                              profile["name"],
                              replicate,
                              seed,
                              name,
                              round(result["mean_cumulative_collisions"], 3),
                              result["F200"],
                              flush=True)
                    write_json(
                        RESULTS / "summary.json", {
                            "role":
                            "development; not independent confirmation",
                            "records":
                            records,
                            "complete":
                            len(records) == len(protocol["profiles"]) *
                            protocol["replicates_per_profile"] *
                            len(protocol["seeds"]) * len(CONFIGURATIONS)
                        })


if __name__ == "__main__":
    main()
