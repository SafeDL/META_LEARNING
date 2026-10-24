"""Decision-value identities independent of fitted driving models."""

import itertools

import numpy as np

from methods.gcrt.batch_reordering import reordering_values


def test_coherent_joint_distribution_recovers_original_ens():
    labels = np.asarray(list(itertools.product((0, 1), repeat=4)))
    weights = np.random.default_rng(61).uniform(size=16)
    weights /= weights.sum()
    p = weights @ labels
    conditional = np.empty((4, 2, 4))
    for candidate in range(4):
        for value in (0, 1):
            subset = labels[:, candidate] == value
            conditional[candidate, value] = weights[subset] @ labels[subset] / weights[subset].sum()
    for horizon in (1, 2, 4):
        values = reordering_values(p, conditional, np.arange(4), ['d', 'b', 'a', 'c'], horizon)
        np.testing.assert_allclose(values['adjusted'], values['original'], atol=1e-14)
        np.testing.assert_allclose(values['fixed_batch_drift'], 0, atol=1e-14)


def test_uniform_probability_shift_does_not_create_reordering_reward():
    p = np.array([.2, .4, .3, .1])
    conditional = np.array([[[.2, .3, .2, .1], [.3, .5, .6, .2]]] * 4)
    first = reordering_values(p, conditional, np.arange(4), ['a', 'b', 'c', 'd'], 3)
    shifted = reordering_values(p, conditional + .1, np.arange(4), ['a', 'b', 'c', 'd'], 3)
    np.testing.assert_allclose(first['adjusted'], shifted['adjusted'], atol=1e-14)
    np.testing.assert_allclose(shifted['original'] - first['original'], .2, atol=1e-14)


def test_greedy_boundary_and_nonnegative_reallocation_value():
    p = np.array([.2, .7, .1, .4])
    conditional = np.random.default_rng(63).uniform(size=(3, 2, 4))
    available = np.array([0, 2, 3])
    one = reordering_values(p, conditional, available, ['a', 'b', 'c', 'd'], 1)
    np.testing.assert_array_equal(one['adjusted'], p[available])
    full = reordering_values(p, conditional, available, ['a', 'b', 'c', 'd'], 3)
    np.testing.assert_allclose(full['adjusted'], full['passive_batch'], atol=1e-14)
