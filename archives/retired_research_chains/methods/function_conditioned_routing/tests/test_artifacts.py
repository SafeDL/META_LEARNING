import json
from pathlib import Path

import numpy as np

from highway_sim_env.data.response_bank import ResponseBank
from sut_algorithms.highway_env.idm_profiles import PROFILE_NAMES
from methods.function_conditioned_routing.benchmark import RELEASE_NAMES
from methods.function_conditioned_routing.experiment import (
    severity_response,
)


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
RESULTS = REPOSITORY / "results" / "method_chains"


def test_formal_artifacts_cover_every_declared_target_and_method() -> None:
    results = RESULTS / PACKAGE.name
    for directory, names in (
        ("aligned_benchmark", PROFILE_NAMES),
        ("functional_shift_benchmark", RELEASE_NAMES),
    ):
        benchmark = results / directory
        summary = json.loads((benchmark / "summary.json").read_text(encoding="utf-8"))
        assert summary["protocol"]["sut_names"] == list(names)
        assert summary["protocol"]["source_suts_per_fold"] == 5
        assert "Function-Conditioned Mining" in summary["mean_metrics"]
        for name in (
            "response_bank.npz",
            "mining_results.csv",
            "routing_diagnostics.csv",
            "oracle_headroom.json",
            "critical_recall_curve.png",
            "held_out_prediction_error.png",
            "target_recall_by_budget.png",
            "report.md",
        ):
            assert (benchmark / name).stat().st_size > 0


def test_formal_function_shift_acceptance_gates() -> None:
    results = RESULTS / PACKAGE.name / "functional_shift_benchmark"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    robustness = json.loads(
        (results / "robustness_summary.json").read_text(encoding="utf-8")
    )
    validation = summary["validation"]
    assert validation["oracle_function_gain_at_50"] > 0.15
    assert validation["proposed_gain_over_adate_at_50"] > 0.15
    assert validation["gap_to_function_oracle_at_50"] < 0.03
    assert robustness["proposed_gain_over_adate"]["wins"] == 3


def test_formal_function_shift_contains_each_dangerous_replay() -> None:
    gif_dir = RESULTS / PACKAGE.name / "functional_shift_benchmark" / "gifs"
    manifest = json.loads((gif_dir / "manifest.json").read_text(encoding="utf-8"))
    replays = manifest["replays"]
    assert {replay["mode"] for replay in replays} == {
        "fast_intrusion",
        "cutin_braking",
        "lead_braking",
        "stop_and_go",
        "slow_lead_following",
    }
    for replay in replays:
        assert replay["collision"] or replay["near_miss"]
        assert replay["physical_duration_seconds"] >= 3.0
        assert (gif_dir / replay["gif"]).stat().st_size > 0


def test_aligned_bank_is_the_original_mining_detour_bank() -> None:
    aligned = ResponseBank.load(
        RESULTS / PACKAGE.name / "aligned_benchmark" / "response_bank.npz"
    )
    original = ResponseBank.load(
        RESULTS / "detour_fusion" / "response_bank.npz"
    )
    assert aligned.sut_names == original.sut_names
    for name in (
        "anchors",
        "vulnerability",
        "collisions",
        "near_misses",
        "min_ttc",
        "min_distance",
        "completed",
        "modes",
    ):
        assert np.array_equal(getattr(aligned, name), getattr(original, name))


def test_function_shift_bank_contains_physical_scenario_controls() -> None:
    bank = ResponseBank.load(
        RESULTS / PACKAGE.name / "functional_shift_benchmark" / "response_bank.npz"
    )
    assert bank.scenario_controls is not None
    assert bank.scenario_controls.shape == (240, 2)
    assert np.all((bank.scenario_controls >= 0.0) & (bank.scenario_controls <= 1.0))
    assert np.all((bank.collisions | bank.near_misses).sum(axis=1) > 0)


def test_severity_response_breaks_crash_ties_without_reordering_events() -> None:
    bank = ResponseBank(
        anchors=np.zeros((3, 2)),
        sut_names=("source",),
        vulnerability=np.asarray([[1.0, 1.0, 0.99]]),
        collisions=np.asarray([[True, True, False]]),
        near_misses=np.asarray([[False, False, True]]),
        min_ttc=np.asarray([[0.5, 2.0, 0.2]]),
        min_distance=np.zeros((1, 3)),
        completed=np.ones((1, 3), dtype=bool),
        modes=np.asarray(["cutin", "cutin", "cutin"]),
    )
    response = severity_response(bank, [0])[0]
    assert response[0] > response[1] > response[2]
