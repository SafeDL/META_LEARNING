import copy

import numpy as np
import pytest
import torch

from methods.common.gp_classifier import ClassificationKernel, collision_probabilities
from methods.gcrt.counterfactual_adaptation import batch_utility, virtual_update
from methods.gcrt.residual_scale import update_reliability
from methods.gcrt.residual_search import learned_probabilities


@pytest.mark.parametrize('label', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_virtual_refit_uses_only_supplied_labels_and_preserves_actual_state(monkeypatch, label, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    torch.set_num_threads(1)
    torch.manual_seed(1307)
    x = np.random.default_rng(1307).uniform(-1, 1, (8, 3))
    base, raw = ClassificationKernel(3, deep=True).to(device), ClassificationKernel(3).to(device)
    current, physical = copy.deepcopy(base), copy.deepcopy(raw)
    sources = {'means': torch.zeros(2, 8, dtype=torch.float64, device=device),
               'covariances': torch.eye(8, dtype=torch.float64, device=device)[None].repeat(2, 1, 1)*.02}
    log_weights = torch.tensor([.375, .375, .25], dtype=torch.float64, device=device).log()[:, None].expand(-1, 8).clone()
    saved_weights = log_weights.clone()
    states = [{k: v.clone() for k, v in m.state_dict().items()} for m in (base, raw, current, physical)]
    paid, labels = [1, 3], [True, False]
    with torch.no_grad():
        before = learned_probabilities(current, None, x, paid, labels, sources, float(base.mean), .3)
    fits = []

    def fit(model, fitted_x, fitted_y, **settings):
        fits.append((fitted_x.copy(), list(fitted_y), settings))
        with torch.no_grad():
            model.mean.add_(.03 if label else -.03)

    monkeypatch.setattr('methods.gcrt.counterfactual_adaptation.fit_target', fit)
    branch = virtual_update(base, raw, current, physical, x, paid, labels, sources,
                            float(base.mean), .3, log_weights, before, 6, label)
    assert len(fits) == 2
    for fitted_x, fitted_y, settings in fits:
        np.testing.assert_array_equal(fitted_x, x[[1, 3, 6]])
        assert fitted_y == [True, False, label] and settings['steps'] == 10
    assert {f[2]['learning_rate'] for f in fits} == {.003, .01}
    expected = update_reliability(saved_weights, before, 6, label, torch.tensor(x, device=device), .3)
    torch.testing.assert_close(branch['log_weights'], expected)
    torch.testing.assert_close(log_weights, saved_weights, rtol=0, atol=0)
    for model, state in zip((base, raw, current, physical), states):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert paid == [1, 3] and labels == [True, False]
    with torch.no_grad():
        conditional = learned_probabilities(current, None, x, [1, 3, 6], [True, False, label],
                                              sources, float(base.mean), .3)
        raw_probability = collision_probabilities(physical, x[[1, 3, 6]], [True, False, label], x)
    expected_probability = .9*(expected.softmax(0)*conditional).sum(0)+.1*raw_probability
    torch.testing.assert_close(branch['probabilities']['fixed'], expected_probability)


def test_one_feedback_batch_utility_counts_current_query_and_excludes_paid_points():
    p = np.array([.2, .4, .9, .5])
    branches = {False: np.array([.99, .99, .1, .3]), True: np.array([.99, .99, .8, .7])}
    assert batch_utility(p, 1, branches, [1, 2, 3], 1) == pytest.approx(.4)
    assert batch_utility(p, 1, branches, [1, 2, 3], 2) == pytest.approx(.4+.6*.3+.4*.8)
    assert batch_utility(p, 1, branches, [1, 2, 3], 3) == pytest.approx(.4+.6*.4+.4*1.5)
    with pytest.raises(ValueError, match='horizon'):
        batch_utility(p, 1, branches, [1, 2, 3], 4)
