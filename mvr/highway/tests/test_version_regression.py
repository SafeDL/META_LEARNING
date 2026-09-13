"""Contract tests for chronological PR-BRVT version-regression replay."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mvr.highway.diva.low_rank_prior import LowRankPrior
from mvr.highway.diva.posterior import adapt_posterior
from mvr.highway.diva.version_mining import (
    TargetReplay,
    deterministic_argmax,
    masked_probability_scores,
    sequential_indices,
    trace_counts,
)
from mvr.highway.diva.version_reference import (
    VersionReference,
    build_version_reference,
    classify_regressions,
)
from mvr.highway.data import response_bank
from mvr.highway.sut.version_lineage import load_version_lineages


def _reference() -> VersionReference:
    return VersionReference(
        previous_safe=np.array([True, True, False, True, True, True]),
        previous_vulnerability=np.array([0.1, 0.8, 0.9, 0.4, 0.2, 0.7]),
        historical_critical=np.array([False, True, True, False, True, True]),
    )


def _prior() -> LowRankPrior:
    return LowRankPrior.fit(
        np.array([[0.1, 0.2, 0.7, 0.3, 0.5, 0.4], [0.2, 0.1, 0.6, 0.4, 0.6, 0.3], [0.3, 0.2, 0.5, 0.2, 0.4, 0.5]]),
        rank=2,
    )


def test_manifest_rejects_non_direct_predecessor(tmp_path):
    manifest = {
        "versions": [
            {"lineage_id": "x", "version_id": "x1", "version_order": 1, "predecessor_id": None, "controller": "IDM", "profile": {"name": "x1", "controller": "IDM"}},
            {"lineage_id": "x", "version_id": "x2", "version_order": 2, "predecessor_id": "x1", "controller": "IDM", "profile": {"name": "x2", "controller": "IDM"}},
            {"lineage_id": "x", "version_id": "x3", "version_order": 3, "predecessor_id": "x1", "controller": "IDM", "profile": {"name": "x3", "controller": "IDM"}},
        ]
    }
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="direct prior"):
        load_version_lineages(path)


def test_labels_distinguish_new_reintroduced_and_persistent_events():
    history = np.array([[False, True, True, False, True, False], [False, False, True, False, True, True], [False, False, True, False, True, False]])
    reference = build_version_reference(
        history, np.zeros_like(history), np.zeros(6), history[-1],
        np.zeros(6, dtype=bool), np.ones(6, dtype=bool),
    )
    labels = classify_regressions(
        reference, np.array([True, True, True, False, False, True]),
        np.zeros(6, dtype=bool), np.ones(6, dtype=bool),
    )
    assert labels.regression.tolist() == [True, True, False, False, False, True]
    assert labels.new_in_archive.tolist() == [True, False, False, False, False, False]
    assert labels.reintroduced.tolist() == [False, True, False, False, False, True]
    counts = trace_counts(list(range(6)), labels)
    assert counts["regression_count"] == 3
    assert counts["new_in_archive_count"] + counts["reintroduced_count"] == 3


def test_unknown_previous_outcome_is_not_eligible():
    reference = build_version_reference(
        np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool), np.zeros(3),
        np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), np.array([True, False, True]),
    )
    assert reference.previous_safe.tolist() == [True, False, True]


def test_masked_zero_probabilities_never_select_excluded_anchor():
    assert deterministic_argmax(np.array([-np.inf, 0.0, -np.inf, 0.0])) == 1


def test_reveal_enforces_budget_and_uniqueness():
    replay = TargetReplay(np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), budget=1)
    replay.reveal(0)
    with pytest.raises(ValueError, match="budget"):
        replay.reveal(1)
    replay = TargetReplay(np.zeros(3), np.zeros(3), np.zeros(3), np.ones(3), budget=2)
    replay.reveal(0)
    with pytest.raises(ValueError, match="only once"):
        replay.reveal(0)


def test_frozen_and_sequential_have_identical_first_selection():
    prior = _prior()
    reference = _reference()
    target = np.array([0.1, 0.7, 0.5, 0.2, 0.8, 0.4])
    frozen, _, _ = sequential_indices(
        prior, reference, TargetReplay(target, np.zeros(6), np.zeros(6), np.ones(6), 2),
        2, 0.75, 0.03, False,
    )
    sequential, _, _ = sequential_indices(
        prior, reference, TargetReplay(target, np.zeros(6), np.zeros(6), np.ones(6), 2),
        2, 0.75, 0.03, True,
    )
    assert frozen[0] == sequential[0]


def test_unqueried_target_values_do_not_change_the_next_selection():
    prior = _prior()
    reference = _reference()
    target_a = np.array([0.1, 0.7, 0.5, 0.2, 0.8, 0.4])
    target_b = target_a.copy()
    target_b[4] = 99.0
    selected_a, _, _ = sequential_indices(prior, reference, TargetReplay(target_a, np.zeros(6), np.zeros(6), np.ones(6), 2), 2, 0.75, 0.03, True)
    selected_b, _, _ = sequential_indices(prior, reference, TargetReplay(target_b, np.zeros(6), np.zeros(6), np.ones(6), 2), 2, 0.75, 0.03, True)
    assert selected_a == selected_b


def test_eligible_shortfall_uses_only_available_candidates():
    prior = _prior()
    reference = VersionReference(np.array([True, False, True, False, True, False]), np.zeros(6), np.zeros(6, dtype=bool))
    selected, _, _ = sequential_indices(
        prior, reference, TargetReplay(np.zeros(6), np.zeros(6), np.zeros(6), np.ones(6), 3),
        3, 0.75, 0.03, True,
    )
    assert len(selected) == len(set(selected)) == 3


def test_version_scorer_masks_all_noneligible_positions():
    prior = _prior()
    reference = _reference()
    posterior = adapt_posterior(prior, np.array([], dtype=int), np.array([]))
    scores = masked_probability_scores(prior, posterior, reference, (), 0.75, 0.03)
    assert np.isneginf(scores[2])


def test_future_version_data_cannot_change_a_past_prior_or_reference():
    versions = np.array([
        [0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.2, 0.5],
        [0.4, 0.3, 0.6], [0.5, 0.4, 0.7], [0.6, 0.5, 0.8],
    ])
    before = LowRankPrior.fit(versions[:3], rank=2)
    versions[-1] = 99.0
    after = LowRankPrior.fit(versions[:3], rank=2)
    assert np.allclose(before.mean, after.mean)
    assert np.allclose(before.basis, after.basis)


def test_historical_reappearance_never_suppresses_an_eligible_probability():
    prior = _prior()
    posterior = adapt_posterior(prior, np.array([], dtype=int), np.array([]))
    first = _reference()
    second = VersionReference(
        first.previous_safe, first.previous_vulnerability,
        np.logical_not(first.historical_critical),
    )
    assert np.allclose(
        masked_probability_scores(prior, posterior, first, (), 0.75, 0.03),
        masked_probability_scores(prior, posterior, second, (), 0.75, 0.03),
        equal_nan=True,
    )


def test_zero_regressions_have_no_recall_claim():
    labels = classify_regressions(
        _reference(), np.zeros(6, dtype=bool), np.zeros(6, dtype=bool), np.ones(6, dtype=bool)
    )
    assert labels.regression.sum() == 0
    assert trace_counts([0, 1], labels)["regression_count"] == 0


def test_legacy_response_builder_keeps_registry_rows(monkeypatch):
    class Result:
        vulnerability = 0.0
        collision = False
        near_miss = False
        min_ttc = np.inf
        min_distance = 1.0
        completed = True

    monkeypatch.setattr(response_bank, "run_cutin_episode", lambda *args, **kwargs: Result())
    bank = response_bank.build_response_bank(np.array([[8.0, -2.0]]), seed=1)
    assert bank.sut_names == response_bank.PROFILE_NAMES
