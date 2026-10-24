import copy

import numpy as np

from benchmark.discovery_converged_diagnosis import weights_for_prefix


def example_queries(count):
    return [{'index': i % 3, 'prequery_group_log_likelihood': [-.1 * (i + 1), -.2, -.3],
             'feedback': {'collision': bool(i % 2)}} for i in range(count)]


def test_reliability_snapshot_uses_no_current_or_future_values():
    x = np.array([[0.], [.3], [.9]])
    prior = np.array([.4, .35, .25])
    original = example_queries(45)
    changed = copy.deepcopy(original)
    for q in changed[8:]:
        q['index'] = 2
        q['prequery_group_log_likelihood'] = [-100., -20., -1.]
    left = weights_for_prefix(original, x, prior, 8)
    right = weights_for_prefix(changed, x, prior, 8)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(left[0], left[1], atol=1e-14)


def test_recent_window_drops_old_scores_without_rescoring_paid_labels():
    x = np.array([[0.], [.3], [.9]])
    prior = np.array([.4, .35, .25])
    original = example_queries(45)
    changed = copy.deepcopy(original)
    for q in changed[:8]:
        q['prequery_group_log_likelihood'] = [-20., -.01, -.01]
    a = weights_for_prefix(original, x, prior, 40)
    b = weights_for_prefix(changed, x, prior, 40)
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.allclose(a[0], b[0])
    for q in changed:
        q['feedback']['collision'] = not q['feedback']['collision']
    c = weights_for_prefix(changed, x, prior, 40)
    for left, right in zip(b, c):
        np.testing.assert_array_equal(left, right)


def test_target_floor_preserves_mass_and_never_reduces_target_weight():
    x = np.array([[0.], [.3], [.9]])
    prior = np.array([.4, .35, .25])
    queries = [{'index': 0, 'prequery_group_log_likelihood': [-.01, -.1, -10.]}] * 8
    original, recent, floor = weights_for_prefix(queries, x, prior, 8)
    assert np.all(floor[-1] >= .25)
    assert np.all(floor[-1] >= original[-1])
    for weights in (original, recent, floor):
        np.testing.assert_allclose(weights.sum(0), 1.)
        assert np.isfinite(weights).all() and np.min(weights) >= 0
    np.testing.assert_allclose(floor[0] / floor[1], original[0] / original[1])
