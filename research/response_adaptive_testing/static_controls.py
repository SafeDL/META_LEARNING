"""Secondary committed rankings from fixed priors, with truth used only afterward."""
import time

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import SOURCE_NAMES, predict

from .config import BUDGET, OUTPUT, SEEDS
from .confirmation import CONFIRMATION, REPLICATES, verify_lock
from .develop import metrics
from .initial_prior import coordinates, reference_means
from .session import AdaptiveTestingSession

POLICIES = ("locked_prior_static", "learned_mean_static")
RESULTS = OUTPUT / "confirmation_static_controls"


def committed_ranking(x, prediction, options, mean_model=None):
    session = AdaptiveTestingSession(x, *prediction, options)
    if mean_model is not None:
        assert tuple(mean_model["source_names"]) == SOURCE_NAMES
        family, risk, collision = coordinates(x, *prediction)
        logits = risk.new_tensor(mean_model["mean_logits"])
        session.mean_r, session.mean_c = reference_means(
            family, risk, collision, list(range(len(SOURCE_NAMES))), logits,
            risk.new_tensor(options["collision_offset"]))
    q = session.probabilities().cpu().numpy()
    assert np.isfinite(q).all() and ((0 <= q) & (q <= 1)).all()
    selected = np.argsort(-q, kind="stable")[:BUDGET].tolist()
    assert len(selected) == len(set(selected)) == BUDGET
    return selected, q


def main():
    torch.set_num_threads(1)
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    options = protocol["candidate_options"]
    learned = read_json(OUTPUT /
                        "initial_prior/models/collision_mean_600.json")
    assert learned["target_data_used"] is False
    assert learned["base_options"] == options
    records = []
    for profile in protocol["profiles"]:
        for replicate in range(REPLICATES):
            folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
            if not (folder / "cost.json").exists():
                continue
            with np.load(folder / "responses.npz") as bank:
                for seed in SEEDS:
                    paths = {
                        policy:
                        RESULTS / profile["name"] / f"pool_{replicate}" /
                        f"{policy}_{seed}.json"
                        for policy in POLICIES
                    }
                    prediction = (predict(bank["x"], seed) if not all(
                        path.exists() for path in paths.values()) else None)
                    for policy, path in paths.items():
                        if path.exists():
                            result = read_json(path)
                        else:
                            started = time.perf_counter()
                            mean_model = learned if policy == "learned_mean_static" else None
                            # The complete sequence is fixed before any target response is accessed.
                            selected, q = committed_ranking(
                                bank["x"], prediction, options, mean_model)
                            observations = [{
                                "index":
                                index,
                                "continuous_risk":
                                float(bank["risk"][index])
                            } for index in selected]
                            result = {
                                **metrics(bank["collision"], selected), "selected_indices":
                                selected,
                                "observations":
                                observations,
                                "model_failure_probabilities":
                                q[selected].tolist(),
                                "options":
                                options,
                                "mean_model":
                                mean_model,
                                "elapsed_s":
                                time.perf_counter() - started,
                                "stage":
                                "secondary fixed-ranking control; no target outcome for ranking or training"
                            }
                            write_json(path, result)
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
                            profile["name"],
                            "controller":
                            profile["controller"],
                            "replicate":
                            replicate,
                            "seed":
                            seed,
                            "method":
                            policy,
                            **measured, "normalized_area":
                            measured["mean_cumulative_collisions"] /
                            total if total else 0,
                            "missed_failures":
                            total - measured["F200"]
                        })
                print("SECONDARY STATIC BANK",
                      profile["name"],
                      replicate,
                      flush=True)
                write_json(
                    RESULTS / "summary.json", {
                        "stage":
                        "secondary diagnostic; primary gate unchanged",
                        "records":
                        records,
                        "full_suite_complete":
                        len(records) == 48 * len(SEEDS) * len(POLICIES),
                        "selection_feedback":
                        "queried risk recorded; committed ranking does not update",
                        "unqueried_target_outcomes_used_for_ranking":
                        False
                    })


if __name__ == "__main__":
    main()
