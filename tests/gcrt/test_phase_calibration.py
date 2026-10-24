from types import SimpleNamespace

import numpy as np
import torch

from methods.common.classification import logistic_moments
from methods.common.gp_classifier import ClassificationKernel
from methods.gcrt.phase_calibration import calibrated_phase_bases


def test_initial_marginal_probability_matches_historical_risk_set_frequency():
    history = [SimpleNamespace(source_id='a', family='rule', role='history'),
               SimpleNamespace(source_id='b', family='learning', role='history')]
    observations = {'a': np.array([[1., np.nan, np.nan], [0., 1., np.nan], [0., 0., 0.]]),
                    'b': np.array([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]])}
    model = ClassificationKernel(3)
    with torch.no_grad():
        model.log_scale.fill_(np.log(np.sqrt(6.685)))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    public = np.random.default_rng(71).uniform(size=(31, 3))
    bases, checks = calibrated_phase_bases(model, history, observations, public)
    for base, expected, check in zip(bases, [.25, 2/7, 1/3], checks):
        with torch.no_grad():
            variance = base.covariance(public).diagonal().numpy()
        p = logistic_moments(np.full(31, base.mean.item()), variance)[0]
        np.testing.assert_allclose(p.mean(), expected, atol=1e-11, rtol=1e-11)
        assert check['registered_marginal_probability_mean'] > expected
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
