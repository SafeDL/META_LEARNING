"""Controlled structural removals on already disclosed development pools."""
import time

import numpy as np
import torch

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.scenarios import parameter_cells
from research.history_response_testing.history_model import predict

from .config import BUDGET, OUTPUT, SEEDS
from .develop import metrics, run_one
from .session import AdaptiveTestingSession


def coverage_metrics(bank, selected):
    cells = parameter_cells(bank["x"])
    collision = bank["collision"]
    discovered = np.asarray(selected)[collision[selected]]
    elite = np.zeros(512)
    np.maximum.at(elite, cells[selected],
                  np.maximum(bank["risk"][selected] - 0.5, 0))
    return {
        "collision_cells":
        len(np.unique(cells[discovered])),
        "total_collision_cells":
        len(np.unique(cells[collision])),
        "risk_qd_score":
        float(elite.sum()),
        "collisions_per_family":
        [int((bank["x"][discovered, 4] == family).sum()) for family in (0, 1)]
    }


def main():
    torch.set_num_threads(1)
    prior = read_json(OUTPUT /
                      "confirmation/protocol.json")["candidate_options"]
    candidates = {
        "full": prior,
        "source_contrasts_only": {
            **prior, "positive_loading": False,
            "loading": [[0] * 6] * 2
        },
        "local_transfer_only": {
            **prior, "source_scale": 0
        },
        "no_risk_failure_coupling": {
            **prior, "source_scale": 0,
            "positive_loading": False,
            "loading": [[0] * 6] * 2
        },
        "static_posterior": prior,
    }
    pools = [("original", BASELINE / "target/responses.npz")]
    pools += [(path.parent.name, path)
              for path in sorted((OUTPUT /
                                  "development_pools").glob("*/responses.npz"))
              ]
    records = []
    for name, pool in pools:
        bank = np.load(pool)
        for seed in SEEDS:
            prediction = predict(bank["x"], seed)
            for method, options in candidates.items():
                path = OUTPUT / "component_controls" / f"{name}_{method}_{seed}.json"
                if path.exists():
                    result = read_json(path)
                elif method == "static_posterior":
                    started = time.perf_counter()
                    session = AdaptiveTestingSession(bank["x"], *prediction,
                                                     options)
                    q = session.probabilities().cpu().numpy()
                    selected = np.argsort(-q, kind="stable")[:BUDGET].tolist()
                    result = {
                        **metrics(bank["collision"], selected), "selected_indices":
                        selected,
                        "queries": [{
                            "index":
                            index,
                            "continuous_risk":
                            float(bank["risk"][index]),
                            "model_failure_probability":
                            float(q[index])
                        } for index in selected],
                        "options":
                        options,
                        "elapsed_s":
                        time.perf_counter() - started,
                        "stage":
                        "development; feedback measured but posterior held fixed"
                    }
                else:
                    result = run_one(bank["x"], bank["risk"],
                                     bank["collision"], prediction, options)
                result.update(
                    coverage_metrics(bank, result["selected_indices"]))
                result["component_control"] = method
                write_json(path, result)
                records.append({
                    "pool": name,
                    "method": method,
                    "seed": seed,
                    **{
                        key: result[key]
                        for key in ("mean_cumulative_collisions", "F200", "recall", "pool_collisions", "collision_cells", "risk_qd_score", "collisions_per_family")
                    }
                })
                print("COMPONENT",
                      name,
                      seed,
                      method,
                      round(result["mean_cumulative_collisions"], 3),
                      result["F200"],
                      flush=True)
            write_json(OUTPUT / "component_controls" / "summary.json", records)


if __name__ == "__main__":
    main()
