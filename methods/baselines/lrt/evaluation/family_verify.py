"""Audit saved family trajectories against raw responses and frozen inputs."""

from zipfile import ZipFile

import numpy as np

from methods.baselines.hgt.io import read_json, write_json
from benchmark.data import bank_path, measured_seconds
from benchmark.initialization import initial_indices
from methods.common.snapshots import current_path, current_source, verify_source

from .by_family import (
    BUDGET,
    FAMILIES,
    INITIAL,
    METHODS,
    PROJECT,
    ROOT,
    SEED,
    SUT_IDS,
)
from .family_report import verify_trace


def verify_observations(trace, risk, scenario_ids, durations, initial):
    verify_trace(trace)
    selected = trace["selected_indices"]
    assert selected[:INITIAL] == initial
    assert len(trace["disclosures"]) == BUDGET
    assert [row["index"] for row in trace["queries"]] == selected
    np.testing.assert_array_equal(
        [row["risk"] for row in trace["queries"]], risk[selected]
    )
    np.testing.assert_array_equal(trace["scenario_ids"], scenario_ids[selected])
    np.testing.assert_array_equal(trace["execution_s"], durations[selected])
    for row, index in zip(trace["disclosures"], selected):
        assert row["index"] == index and row["risk"] == float(risk[index])
        assert row["collision"] is None
    timing = trace["timing"]
    assert timing["task_initialization_s"] >= 0
    for key in ("query_compute_s", "response_transport_s"):
        values = np.asarray(timing[key])
        assert (
            values.shape == (BUDGET,)
            and np.isfinite(values).all()
            and (values >= 0).all()
        )
    if trace["method"] in ("lrt", "dart", "no_value_learning"):
        remaining = 20.0
        for decision in trace["decisions"]:
            assert abs(decision["loss_remaining_before"] - remaining) < 1e-9
            if decision["selected_index"] == decision["reference_index"]:
                expected = 0.0
            else:
                position = decision["candidate_indices"].index(
                    decision["selected_index"]
                )
                losses = np.asarray(decision["candidate_losses"])
                expected = float(
                    losses[position]
                    if trace["method"] == "dart"
                    else losses[:, position].max()
                )
                assert remaining > 0
            assert abs(decision["loss_charged"] - expected) < 1e-10
            remaining -= decision["loss_charged"]
            assert remaining >= -1e-9
            assert abs(decision["loss_remaining_after"] - remaining) < 1e-9
        assert abs(trace["loss_budget_remaining"] - remaining) < 1e-9


def main():
    weights = []
    for family in FAMILIES:
        base = ROOT / family
        weights.extend(
            (
                base / "kernel/models/seed_11.pt",
                base / "encoder/models/decision_11.pt",
                base / "models/full_11.pt",
            )
        )
    packages = (
        "methods/baselines/lrt",
        "methods/baselines/cg_brt",
        "methods/baselines/dart",
        "methods/baselines/hgt",
        "methods/baselines/hem",
        "methods/baselines/saem",
    )
    sources = [p for folder in packages for p in (PROJECT / folder).glob("*.py")]
    sources.extend(
        PROJECT / name
        for name in (
            "benchmark/initialization.py",
            "benchmark/timing.py",
            "benchmark/risk_channel.py",
            "methods/baselines/lrt/evaluation/by_family.py",
            "methods/common/prior.py",
            "methods/common/student.py",
            "methods/common/workspace.py",
        )
    )
    with ZipFile(ROOT / "frozen_inputs.zip") as archive:
        entries = {current_path(name): name for name in archive.namelist()}
        for path in sources + weights:
            relative = path.relative_to(PROJECT).as_posix()
            entry = entries.get(relative)
            expected = current_source(archive.read(entry), entry) if entry else None
            verify_source(PROJECT, relative, expected)
    count = 0
    for family, code in FAMILIES.items():
        for pool in (0, 1, 2):
            for sut in SUT_IDS:
                with np.load(bank_path(pool, sut)) as bank:
                    keep = bank["x"][:, 4] == code
                    risk, ids = bank["risk"][keep], bank["scenario_id"][keep]
                durations = measured_seconds(pool, sut, ids)
                initial = initial_indices(len(risk), INITIAL, SEED)
                for method in METHODS:
                    path = (
                        ROOT
                        / family
                        / "trajectories"
                        / f"pool_{pool}"
                        / sut
                        / f"{method}_{SEED}.json"
                    )
                    trace = read_json(path)
                    assert (
                        trace["family"],
                        trace["pool"],
                        trace["sut"],
                        trace["method"],
                        trace["seed"],
                    ) == (family, pool, sut, method, SEED)
                    assert trace["positives"] == int((risk > 0.5).sum())
                    verify_observations(trace, risk, ids, durations, initial)
                    count += 1
    assert count == 288
    review = (
        read_json(ROOT / "timing_review.json")
        if (ROOT / "timing_review.json").exists()
        else None
    )
    write_json(
        ROOT / "verification.json",
        {
            "status": "complete",
            "formal_trajectories": count,
            "formal_paid_queries": count * BUDGET,
            "frozen_policy_source_files": len(sources),
            "frozen_weights": len(weights),
            "shared_initialization": True,
            "unique_queries": True,
            "raw_responses_match": True,
            "archived_execution_times_match": True,
            "replacement_ledgers": True,
            "collision_disclosures": 0,
            "new_physical_calls": 0,
            "timing_review": review,
        },
    )
    print(
        f"VERIFIED {count} trajectories, {count * BUDGET} paid queries, {len(weights)} weights",
        flush=True,
    )


if __name__ == "__main__":
    main()
