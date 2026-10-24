import numpy as np
import pytest
import torch

from methods.common.gp_classifier import collision_probabilities, laplace
from methods.gcrt.joint_response import (MARGIN_NOISE, JointResponseKernel, fit_joint,
                                         joint_loss, margin_values, posterior,
                                         predictive_log_likelihood, probabilities)


def fixture(uncorrelated=False):
    torch.manual_seed(19)
    model = JointResponseKernel(3,uncorrelated=uncorrelated)
    x = np.random.default_rng(19).random((9,3))
    return model,x


def test_joint_kernel_and_conditioned_covariance_are_positive_semidefinite():
    model,x = fixture()
    mean,covariance = model.joint_mean(9),model.joint_covariance(x)
    before = covariance.detach().clone()
    mean_before = mean.detach().clone()
    _,mu,cov = posterior(mean,covariance,[1,3,7],[True,False,False],margin_values([.2,2,5]))
    for value in (covariance,cov):
        torch.testing.assert_close(value,value.T,atol=1e-12,rtol=1e-12)
        assert torch.linalg.eigvalsh(value).min() > -1e-9
    torch.testing.assert_close(covariance,before,atol=0,rtol=0)
    torch.testing.assert_close(mean,mean_before,atol=0,rtol=0)
    assert torch.isfinite(mu).all()


def test_zero_correlation_preserves_binary_gp_probability_and_factorizes_evidence():
    model,x = fixture(uncorrelated=True)
    selected,labels = [1,3,7],[True,False,False]
    values = margin_values([.2,2,5])
    prior_cov = model.joint_covariance(x)
    nll,mean,cov = posterior(model.joint_mean(9),prior_cov,selected,labels,values)
    p = probabilities(mean,cov)

    class CachedKernel:
        # Isolate Gaussian conditioning from float32 feature batch rounding.
        mean = model.mean

        def covariance(self,a,b=None):
            rows = [np.flatnonzero((x == row).all(1))[0] for row in a]
            columns = rows if b is None else [np.flatnonzero((x == row).all(1))[0] for row in b]
            return prior_cov[:9,:9][rows][:,columns]

    cached = CachedKernel()
    expected = collision_probabilities(cached,x[selected],labels,x)
    torch.testing.assert_close(p,expected,atol=1e-10,rtol=1e-10)
    margin_cov = prior_cov[9:,9:][selected][:,selected]+MARGIN_NOISE**2*torch.eye(3,dtype=torch.float64)
    normal = torch.distributions.MultivariateNormal(model.margin_mean.expand(3),margin_cov)
    gaussian_nll = -normal.log_prob(torch.as_tensor(values))
    torch.testing.assert_close(nll,laplace(cached,x[selected],labels)[0]+gaussian_nll,atol=1e-10,rtol=1e-10)


def test_negative_cross_response_correlation_uses_clearance_without_relabeling_collision():
    model,x = fixture()
    prior = model.joint_mean(9),model.joint_covariance(x)
    low = posterior(*prior,[2],[False],margin_values([.05]))
    high = posterior(*prior,[2],[False],margin_values([8]))
    assert probabilities(low[1],low[2])[2] > probabilities(high[1],high[2])[2]+1e-3
    # A positive geometric distance at a collision is a valid separate observation.
    assert torch.isfinite(joint_loss(model,x[:1],[True],[.7]))


def test_joint_and_binary_only_gradients_are_finite_with_fixed_uncorrelated_control():
    for uncorrelated in (False,True):
        for use_margin in (False,True):
            model,x = fixture(uncorrelated)
            fit_joint(model,x[:5],[True,False,True,False,False],[.2,2,0,8,3],
                      steps=2,use_margin=use_margin)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if uncorrelated:
                assert model.correlation == 0
                assert model.raw_correlation.grad is None


def test_batched_source_conditioning_matches_separate_components():
    model,x = fixture()
    mean = model.joint_mean(9)
    cov = model.joint_covariance(x)
    means = torch.stack((mean,mean+.1))
    covariances = torch.stack((cov,cov*1.1))
    args = ([1,4,6],[True,False,False],margin_values([.1,4,2]))
    result = posterior(means,covariances,*args)
    for i in range(2):
        expected = posterior(means[i],covariances[i],*args)
        for actual,value in zip(result,expected):
            torch.testing.assert_close(actual[i],value,atol=2e-10,rtol=2e-10)


def test_joint_predictive_score_uses_new_margin_before_the_new_collision_label_only():
    model,x = fixture()
    mean,cov = model.joint_mean(9),model.joint_covariance(x)
    yes,p_yes = predictive_log_likelihood(mean,cov,2,True,.7)
    no,p_no = predictive_log_likelihood(mean,cov,2,False,.7)
    torch.testing.assert_close(p_yes,p_no,atol=0,rtol=0)
    torch.testing.assert_close(yes-no,torch.log(p_yes/(1-p_yes)),atol=1e-12,rtol=1e-12)
    binary,p = predictive_log_likelihood(mean,cov,2,True,90,use_margin=False)
    torch.testing.assert_close(binary,probabilities(mean,cov)[2].log(),atol=1e-12,rtol=1e-12)
    assert torch.isfinite(p)


def test_margin_transform_handles_unbounded_measurement_and_rejects_invalid_values():
    assert margin_values(np.inf) == margin_values(100)
    assert margin_values(0) == 0
    with pytest.raises(ValueError,match='nonnegative'):
        margin_values(-1)


@pytest.mark.parametrize('label_mode',['none','all','mixed'])
def test_full_128_paid_support_has_finite_joint_evidence_gradients(label_mode):
    model,_ = fixture()
    with torch.no_grad():
        model.raw_correlation.fill_(np.arctanh(-.99))
    x = np.random.default_rng(53).random((128,3))
    labels = (np.zeros(128,dtype=bool) if label_mode == 'none' else
              np.ones(128,dtype=bool) if label_mode == 'all' else x[:,0] > .7)
    clearance = np.maximum(0,3*x[:,1]-labels)
    loss = joint_loss(model,x,labels,clearance)/128
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
