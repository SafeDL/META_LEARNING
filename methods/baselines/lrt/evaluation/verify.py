"""Verify all paid records, initialization and replacement ledgers."""

import numpy as np

from methods.baselines.dart.evaluation.config import SUT_IDS
from methods.baselines.dart.evaluation.freeze import verify_freeze
from methods.baselines.hgt.io import read_json, write_json

from ..config import ROOT
from .evaluate import bank_path, cases, case_variants, trace_path
from .protocol import freeze


def main():
    verify_freeze()
    freeze()
    count = 0
    for pool, seed, condition in cases():
        for sut in SUT_IDS:
            with np.load(bank_path(pool, sut)) as bank:
                risk = bank["risk"].copy()
            baseline = read_json(trace_path(pool, sut, seed, "cg_brt", condition))
            for variant in case_variants(pool, condition):
                trace = read_json(trace_path(pool, sut, seed, variant, condition))
                selected = trace["selected_indices"]
                assert len(selected) == len(set(selected)) == 200
                assert selected[:10] == baseline["selected_indices"][:10]
                for index, query in zip(selected, trace["queries"]):
                    assert query["index"] == index and query["risk"] == float(
                        risk[index]
                    )
                    assert query["high_risk"] == bool(risk[index] > 0.5)
                assert all(
                    d.get("collision") is None for d in trace.get("disclosures", [])
                )
                if variant not in ("cg_brt", "previous", "unprotected"):
                    remaining = 20.0
                    for decision in trace["decisions"]:
                        assert (
                            abs(decision["loss_remaining_before"] - remaining) < 1e-10
                        )
                        if decision["selected_index"] != decision["reference_index"]:
                            p = decision["candidate_indices"].index(
                                decision["selected_index"]
                            )
                            losses = np.asarray(decision["candidate_losses"])
                            expected = (
                                losses[1, p]
                                if variant == "nominal_guard"
                                else losses[:, p].max()
                            )
                            assert decision["loss_charged"] == expected
                        else:
                            assert decision["loss_charged"] == 0
                        remaining -= decision["loss_charged"]
                        assert remaining >= -1e-10
                        assert abs(decision["loss_remaining_after"] - remaining) < 1e-10
                count += 1
    write_json(
        ROOT / "verification.json",
        {
            "status": "complete",
            "trajectories": count,
            "paid_queries": count * 200,
            "shared_initialization": True,
            "replacement_ledgers": True,
            "frozen_baselines_unchanged": True,
            "collision_disclosures": 0,
            "new_physical_calls": 0,
        },
    )


if __name__ == "__main__":
    main()
