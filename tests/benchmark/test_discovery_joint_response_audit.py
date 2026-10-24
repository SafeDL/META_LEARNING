import numpy as np
import pytest
import torch

from benchmark.discovery_joint_response_audit import prepare_numpy, predict_numpy
from methods.gcrt.joint_response import JointResponseKernel, margin_values, posterior


@pytest.mark.parametrize('with_margin',[False,True])
def test_independent_numpy_joint_posterior_matches_batched_torch_reference(with_margin):
    torch.manual_seed(11)
    model = JointResponseKernel(3)
    x = np.random.default_rng(11).random((10,3))
    mean = model.joint_mean(10).detach()
    covariance = model.joint_covariance(x).detach()
    mean = torch.stack((mean,mean+.1))
    covariance = torch.stack((covariance,covariance*1.1))
    selected,labels = [1,4,8],[True,False,False]
    margin = margin_values([.3,2,5]) if with_margin else None
    expected = posterior(mean,covariance,selected,labels,margin)
    state = prepare_numpy(mean.numpy(),covariance.numpy(),selected,np.asarray(labels,dtype=float),margin)
    queries = [0,3,10,13]
    actual_mean,actual_cov = predict_numpy(mean.numpy(),covariance.numpy(),state,queries,full_covariance=True)
    np.testing.assert_allclose(actual_mean,expected[1][:,queries],atol=2e-10,rtol=2e-10)
    np.testing.assert_allclose(actual_cov,expected[2][:,queries][:,:,queries],atol=2e-10,rtol=2e-10)
    _,variance = predict_numpy(mean.numpy(),covariance.numpy(),state,queries)
    np.testing.assert_allclose(variance,actual_cov.diagonal(axis1=-2,axis2=-1),atol=2e-10,rtol=2e-10)
