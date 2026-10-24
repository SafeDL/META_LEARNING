"""Kernel controls for target-adapted historical response similarity."""

import copy

import numpy as np
import pytest
import torch

from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from methods.gcrt.response_adaptation import ResponseKernel, response_coordinates
from benchmark.discovery_recovery_audit import numpy_probabilities


def fixture():
    torch.manual_seed(7)
    x = np.random.default_rng(7).random((12,6))
    target = ClassificationKernel(3,deep=True)
    return x,target,ResponseKernel(copy.deepcopy(target),3)


def test_kernel_is_symmetric_positive_semidefinite():
    x,_,model = fixture()
    covariance = model.covariance(x).detach().numpy()
    np.testing.assert_allclose(covariance,covariance.T,atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > -1e-10


def test_no_response_matches_original_fsbo_predictions():
    x,target,model = fixture()
    model.response = False
    labels = [False,True,False,True]
    actual = collision_probabilities(model,x[:4],labels,x).detach()
    expected = collision_probabilities(target,x[:4,:3],labels,x[:,:3]).detach()
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_fitted_source_parameters_have_finite_gradients_and_fixed_control_stays_fixed():
    x,_,model = fixture()
    initial = model.log_response_scale.detach().clone()
    fit_target(model,x[:6],[False,True,False,True,False,True],steps=2,learning_rate=.003)
    assert torch.isfinite(model.log_response_scale.grad).all()
    assert not torch.equal(initial,model.log_response_scale)
    _,target,_ = fixture()
    fixed = ResponseKernel(target,3,adaptive=False)
    initial = fixed.log_response_scale.detach().clone()
    fit_target(fixed,x[:6],[False,True,False,True,False,True],steps=2,learning_rate=.003)
    torch.testing.assert_close(initial,fixed.log_response_scale,atol=0,rtol=0)
    assert fixed.log_response_scale.grad is None


def test_source_permutation_preserves_probabilities():
    x,_,model = fixture()
    permutation = [2,0,1]
    with torch.no_grad():
        model.log_response_scale.add_(torch.tensor([-.3,.2,.7]))
        model.log_response_lengthscale.add_(torch.tensor([.2,-.5,.1]))
    permuted = copy.deepcopy(model)
    with torch.no_grad():
        permuted.log_response_scale.copy_(model.log_response_scale[permutation])
        permuted.log_response_lengthscale.copy_(model.log_response_lengthscale[permutation])
    other_x = np.column_stack((x[:,:3],x[:,3:][:,permutation]))
    labels = [False,True,True,False]
    a = collision_probabilities(model,x[:4],labels,x).detach()
    b = collision_probabilities(permuted,other_x[:4],labels,other_x).detach()
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_response_construction_rejects_development_roles():
    class Archive:
        role = 'development'
    with pytest.raises(ValueError,match='historical'):
        response_coordinates([Archive()],np.zeros((4,2)))


def test_independent_numpy_laplace_matches_deployed_prediction_after_adaptation():
    x,_,model = fixture()
    selected = [2,7,9,3,5]
    labels = [False,True,True,False,True]
    fit_target(model,x[selected],labels,steps=2,learning_rate=.003)
    expected = collision_probabilities(model,x[selected],labels,x).detach().numpy()
    np.testing.assert_allclose(numpy_probabilities(model,x,selected,labels),expected,
                               atol=2e-10,rtol=2e-10)
