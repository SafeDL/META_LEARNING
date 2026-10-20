"""Reconstruct same-history advice and verify disclosure and loss accounting."""

import numpy as np
from threadpoolctl import threadpool_limits

from benchmark.data import bank_path
from methods.baselines.cg_brt.reference import ReferencePolicy
from methods.baselines.hem.experts import HistoricalExperts
from methods.baselines.hgt.history import load_history, split_indices, subset
from methods.baselines.hgt.io import read_json, write_json
from methods.baselines.hgt.train import load_kernel

from .config import ROOT, SEEDS, SUT_IDS, pools, variants
from .evaluate import trace_path
from .freeze import verify_freeze


def main():
    verify_freeze()
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    experts = HistoricalExperts(subset(history, train))
    paid = 0
    checked = 0
    with threadpool_limits(limits=1):
        for stage in ("development", "confirmation"):
            for pool in pools(stage):
                with np.load(bank_path(pool, SUT_IDS[0])) as bank:
                    x = bank["x"].copy()
                prediction = experts.predict(x)
                for seed in SEEDS:
                    covariance = (
                        load_kernel(seed, "global_feedback")
                        .covariance(x)
                        .detach()
                        .numpy()
                    )
                    for sut in SUT_IDS:
                        with np.load(bank_path(pool, sut)) as bank:
                            risk = bank["risk"].copy()
                        baseline = read_json(
                            trace_path(stage, pool, sut, "cg_brt", seed)
                        )
                        for policy in variants(stage):
                            trace = read_json(
                                trace_path(stage, pool, sut, policy, seed)
                            )
                            selected = trace["selected_indices"]
                            if len(selected) != 200 or len(set(selected)) != 200:
                                raise ValueError("Invalid paid-query count")
                            if selected[:10] != baseline["selected_indices"][:10]:
                                raise ValueError("Shared initialization changed")
                            if policy != "cg_brt":
                                reference = ReferencePolicy(
                                    covariance,
                                    prediction.mean,
                                    prediction.group_weights,
                                    prediction.fixed_mean(),
                                    seed,
                                    200,
                                    10,
                                )
                                remaining = np.ones(len(x), dtype=bool)
                            charged = 0.0
                            for step, (index, q, d) in enumerate(
                                zip(selected, trace["queries"], trace["decisions"])
                            ):
                                if q["index"] != index or q["risk"] != float(
                                    risk[index]
                                ):
                                    raise ValueError(
                                        "Recorded risk differs from paid disclosure"
                                    )
                                if "collision" in q:
                                    raise ValueError(
                                        "Collision reached the risk-only selector"
                                    )
                                if policy != "cg_brt":
                                    advice, _ = reference.recommend(remaining, step)
                                    if advice != d["reference_index"]:
                                        raise ValueError(
                                            "Reference did not use the same paid history"
                                        )
                                    reference.observe(index, q["risk"], q["high_risk"])
                                    remaining[index] = False
                                    if policy != "unprotected" and index != advice:
                                        if (
                                            d["loss_charged"]
                                            != d["posterior_replacement_loss"]
                                        ):
                                            raise ValueError(
                                                "Replacement bypassed loss accounting"
                                            )
                                    charged += d["loss_charged"]
                            if charged > 20 + 1e-10:
                                raise ValueError(
                                    "Posterior replacement budget exceeded"
                                )
                            checked += 1
                            paid += len(selected)
    measurement = read_json(ROOT / "measurement_status.json")
    if measurement["completed_calls"] != 12288:
        raise ValueError("Independent physical bank is incomplete")
    result = {
        "status": "complete",
        "trajectories_verified": checked,
        "paid_queries_verified": paid,
        "same_history_reference_verified": True,
        "shared_initialization_verified": True,
        "replacement_ledger_verified": True,
        "collision_disclosures": 0,
        "independent_physical_calls": 12288,
        "frozen_decision_sources_models_and_inputs_unchanged": True,
    }
    write_json(ROOT / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
