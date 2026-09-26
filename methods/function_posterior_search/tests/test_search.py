from __future__ import annotations

import numpy as np

from methods.function_posterior_search.search import (
    PosteriorSearch,
    run_adaptive_campaign,
)


def test_duplicate_hypotheses_do_not_change_predictions_or_decisions():
    sources = np.asarray([[0.1, 0.9, 0.2, 0.7], [0.8, 0.2, 0.9, 0.1]])
    events = sources > 0.5
    modes = np.asarray(["cutin"] * 4)
    original = PosteriorSearch(sources, events, modes, 0.15)
    repeated = PosteriorSearch(sources[[0, 0, 0, 1]], events[[0, 0, 0, 1]], modes, 0.15)
    original.observe(0, 0.3)
    repeated.observe(0, 0.3)
    np.testing.assert_allclose(original.predict(), repeated.predict())
    assert original.propose() == repeated.propose()


def test_observations_update_only_their_function():
    sources = np.asarray([[0.1, 0.8, 0.2, 0.9], [0.9, 0.2, 0.8, 0.1]])
    selector = PosteriorSearch(sources, sources > 0.5, np.asarray(["a", "a", "b", "b"]), 0.1)
    before = selector.predict()[0]
    selector.observe(0, 0.1)
    after = selector.predict()[0]
    assert after[1] > before[1]
    np.testing.assert_array_equal(after[2:], before[2:])


def test_last_execution_maximizes_immediate_event_probability():
    sources = np.asarray([[0.1, 0.8, 0.2], [0.9, 0.7, 0.1]])
    selector = PosteriorSearch(sources, sources > 0.5, np.asarray(["a"] * 3), 0.1)
    chosen, _ = selector.propose()
    assert chosen == 1


def test_campaign_reads_one_response_per_execution_without_array_access():
    rng = np.random.default_rng(12)
    sources = rng.random((3, 60))
    modes = np.repeat(["a", "b", "c"], 20)

    class QueryOracle:
        def __init__(self):
            self.queried = []

        def __array__(self, *args, **kwargs):
            raise AssertionError("complete target array was requested")

        def __getitem__(self, index):
            assert isinstance(index, (int, np.integer))
            assert index not in self.queried
            self.queried.append(int(index))
            return float(sources[0, index])

    oracle = QueryOracle()
    selected, decisions = run_adaptive_campaign(
        sources, sources > 0.6, modes, oracle, 0.15,
    )
    np.testing.assert_array_equal(selected, oracle.queried)
    assert len(selected) == len(np.unique(selected)) == 50
    assert [row["step"] for row in decisions] == list(range(1, 51))
    assert set(modes[selected[:10]]) == set(modes)


def test_unqueried_target_changes_do_not_change_campaign():
    rng = np.random.default_rng(17)
    sources = rng.random((3, 60))
    target = rng.random(60)
    modes = np.repeat(["a", "b", "c"], 20)
    arguments = (sources, sources > 0.5, modes)
    first, _ = run_adaptive_campaign(*arguments, target, 0.15)
    changed = target.copy()
    changed[np.setdiff1d(np.arange(60), first)] = 1.0 - changed[
        np.setdiff1d(np.arange(60), first)
    ]
    second, _ = run_adaptive_campaign(*arguments, changed, 0.15)
    np.testing.assert_array_equal(first, second)
