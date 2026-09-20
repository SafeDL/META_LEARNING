from __future__ import annotations

import shutil
from pathlib import Path

from replications.adate_highway_env.adate.visualize import render

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]
RESULTS = REPOSITORY / "results" / "highway_replications" / "adate"


def test_response_mixture_visualizations_use_semantic_names(tmp_path):
    source = RESULTS / "mixture"
    for name in (
            "mixture_coefficients_trace.csv",
            "target_query_trace.csv",
            "source_response_bank.npz",
    ):
        shutil.copy2(source / name, tmp_path / name)

    created = {path.name for path in render(tmp_path)}
    assert created == {
        "mixture_coefficients_vulnerability.png",
        "mixture_coefficients_collision.png",
        "coefficient_stability.png",
        "source_response_boundary.png",
        "critical_discovery.png",
    }


def test_cross_target_visualizations_use_semantic_names(tmp_path):
    source = RESULTS / "dense"
    for name in ("case_strategy_summary.csv", "final_mixture_coefficients.csv"):
        shutil.copy2(source / name, tmp_path / name)
    for case in source.glob("seed_*/target_*"):
        destination = tmp_path / case.relative_to(source)
        destination.mkdir(parents=True)
        shutil.copy2(case / "evaluation_draws.csv", destination / "evaluation_draws.csv")

    created = {path.name for path in render(tmp_path)}
    assert created == {
        "cross_target_estimates.png",
        "cross_target_precision.png",
        "cross_target_coefficients.png",
        "mode_stratified_estimates.png",
    }
