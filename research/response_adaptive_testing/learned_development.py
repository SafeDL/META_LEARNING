"""Evaluate historically learned transfer models on measured development pools."""
import numpy as np
import torch

from methods.history_guided_testing.config import ROOT as BASELINE
from methods.history_guided_testing.io import read_json, write_json
from research.history_response_testing.history_model import predict

from .config import OUTPUT, SEEDS
from .develop import run_one


def main():
    torch.set_num_threads(1)
    pools = [("original", BASELINE / "target/responses.npz")]
    pools += [(path.parent.name, path)
              for path in sorted((OUTPUT /
                                  "development_pools").glob("*/responses.npz"))
              ]
    rows = []
    for name, pool in pools:
        bank = np.load(pool)
        for seed in SEEDS:
            prediction = predict(bank["x"], seed)
            for prior, transform, step in ((prior, transform, step)
                                           for prior in ("meta_transfer",
                                                         "monotone_transfer")
                                           for transform in ("raw", "logit")
                                           for step in (100, 300, 600, 1200)):
                state = OUTPUT / "models" / prior / f"{transform}_{step}.json"
                options = read_json(state)["options"]
                prefix = "" if prior == "meta_transfer" else prior + "_"
                path = OUTPUT / "learned_development" / f"{prefix}{name}_{transform}_{step}_{seed}.json"
                if path.exists():
                    result = read_json(path)
                else:
                    result = run_one(bank["x"], bank["risk"],
                                     bank["collision"], prediction, options)
                    write_json(path, result)
                rows.append({
                    "pool": name,
                    "seed": seed,
                    "transform": transform,
                    "steps": step,
                    "prior": prior,
                    "options": options,
                    **{
                        key: result[key]
                        for key in ("mean_cumulative_collisions", "F200", "pool_collisions", "recall")
                    }
                })
                print("LEARNED",
                      name,
                      seed,
                      prior,
                      transform,
                      step,
                      round(result["mean_cumulative_collisions"], 3),
                      result["F200"],
                      flush=True)
            write_json(OUTPUT / "learned_development" / "summary.json", rows)


if __name__ == "__main__":
    main()
