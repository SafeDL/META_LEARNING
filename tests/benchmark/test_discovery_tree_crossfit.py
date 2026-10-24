"""Coordinate exclusion and degenerate-class regressions for source forests."""

import numpy as np

from benchmark.discovery_tree_crossfit import coordinate_folds, heldout_probability


PARAMETERS = {'n_estimators': 64, 'min_samples_leaf': 4, 'max_features': 1.,
              'bootstrap': False, 'n_jobs': 1, 'random_state': 1301}


def test_shared_folds_have_independent_numpy_reconstruction():
    shuffled = np.arange(512)
    np.random.RandomState(1301).shuffle(shuffled)
    cursor = 0
    for (train, test), size in zip(coordinate_folds(), (103, 103, 102, 102, 102)):
        np.testing.assert_array_equal(test, np.sort(shuffled[cursor:cursor + size]))
        assert not set(train) & set(test) and set(train) | set(test) == set(range(512))
        cursor += size
    assert cursor == 512


def test_heldout_label_poison_does_not_change_source_prediction():
    x = np.random.default_rng(17).uniform(size=(512, 4))
    y = x[:, 0] > .4
    train, test = coordinate_folds()[0]
    before = heldout_probability(x, y, train, test, PARAMETERS)
    poisoned = y.copy()
    poisoned[test] = ~poisoned[test]
    np.testing.assert_array_equal(before, heldout_probability(x, poisoned, train, test, PARAMETERS))


def test_single_class_source_probabilities_are_exact():
    x = np.random.default_rng(19).uniform(size=(512, 4))
    train, test = coordinate_folds()[0]
    for label in (False, True):
        prediction = heldout_probability(x, np.full(512, label), train, test, PARAMETERS)
        np.testing.assert_array_equal(prediction, np.full(len(test), float(label)))
