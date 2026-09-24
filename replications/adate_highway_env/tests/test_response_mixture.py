from __future__ import annotations

import csv
import json

import numpy as np

from highway_env_benchmark.data.response_bank import ResponseBank
from replications.adate_highway_env.adate.experiment import run_mixture
from replications.adate_highway_env.adate.mixture import simplex_least_squares
from replications.adate_highway_env.adate.mixture_selector import MixtureSelector


def test_simplex_qp_recovers_known_convex_mixture_and_records_feasible_solution():
    design = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.8, 0.2, 0.0], [0.2, 0.8, 1.0]])
    truth = np.array([0.2, 0.5, 0.3])
    fitted = simplex_least_squares(design, design @ truth)
    assert not fitted.fallback
    assert fitted.simplex_violation < 1e-8
    assert np.allclose(fitted.alpha, truth, atol=1e-6)


def test_first_proposal_cannot_depend_on_hidden_target_row():
    source = np.array([[0.2, 0.9, 0.1], [0.2, 0.4, 0.7]])
    left = MixtureSelector(source, budget=2, variant="sequential")
    right = MixtureSelector(source, budget=2, variant="sequential")
    assert left.propose() == right.propose() == 1
    left.observe(1, 0.0)
    right.observe(1, 1.0)
    assert left.selected == right.selected == [1]


def test_configured_source_target_roles_recover_an_exact_matching_source(tmp_path):
    anchors = np.array([[10.0, -5.0], [20.0, -2.0], [30.0, 1.0]])
    source_a = np.array([0.9, 0.5, 0.2])
    source_b = np.array([0.1, 0.3, 0.8])
    bank = ResponseBank(
        anchors=anchors,
        sut_names=("source-a", "source-b", "target-copy"),
        vulnerability=np.vstack([source_a, source_b, source_a]),
        collisions=np.zeros((3, 3), dtype=bool),
        near_misses=np.zeros((3, 3), dtype=bool),
        min_ttc=np.ones((3, 3)),
        min_distance=np.ones((3, 3)),
        completed=np.ones((3, 3), dtype=bool),
    )
    bank_path = tmp_path / "bank.npz"
    bank.save(bank_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    run_mixture(
        {
            "seed": 1,
            "budget": 2,
            "responses": ["vulnerability"],
            "static_ks": [1],
            "source_profiles": ["source-a", "source-b"],
            "target_profiles": ["target-copy"],
            "_path": str(config_path),
        },
        tmp_path / "output",
        bank_path,
    )
    with (tmp_path / "output" / "method_summary.csv").open(encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    sequential = next(row for row in summaries if row["method_variant"].endswith("sequential"))
    assert np.allclose(json.loads(sequential["final_alpha"]), [1.0, 0.0], atol=1e-6)
    with (tmp_path / "output" / "source_target_response_agreement.csv").open(encoding="utf-8") as stream:
        agreement = list(csv.DictReader(stream))
    exact = next(row for row in agreement if row["source_profile"] == "source-a")
    assert float(exact["max_absolute_error"]) == 0.0
