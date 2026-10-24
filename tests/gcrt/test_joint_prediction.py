import numpy as np
import pytest
import torch

from methods.gcrt.joint_prediction import marginal_posterior
from methods.gcrt.joint_response import JointResponseKernel, margin_values, posterior


@pytest.mark.parametrize('batch',[False,True])
@pytest.mark.parametrize('with_margin',[False,True])
def test_marginal_prediction_matches_full_gaussian_posterior(batch,with_margin):
    torch.manual_seed(23)
    model = JointResponseKernel(3)
    x = np.random.default_rng(23).random((12,3))
    mean,covariance = model.joint_mean(12),model.joint_covariance(x)
    if batch:
        mean = torch.stack((mean,mean+.1))
        covariance = torch.stack((covariance,covariance*1.2))
    selected,labels = [2,7,9,1],[True,False,False,True]
    values = margin_values([.1,3,5,.7]) if with_margin else None
    full = posterior(mean,covariance,selected,labels,values)
    actual = marginal_posterior(mean,covariance,selected,labels,values)
    expected = full[1],full[2].diagonal(dim1=-2,dim2=-1),full[2][...,torch.arange(12),12+torch.arange(12)]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=2e-10,rtol=2e-10)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is required for execution parity')
def test_full_budget_marginal_prediction_matches_cpu_on_cuda():
    torch.manual_seed(71)
    model = JointResponseKernel(3)
    x = np.random.default_rng(71).random((160,3))
    mean,covariance = model.joint_mean(160).detach(),model.joint_covariance(x).detach()
    selected = list(range(128))
    labels = x[:128,0] > .7
    values = margin_values(3*x[:128,1])
    cpu = marginal_posterior(mean,covariance,selected,labels,values)
    gpu = marginal_posterior(mean.cuda(),covariance.cuda(),selected,labels,values)
    for a,b in zip(cpu,gpu):
        torch.testing.assert_close(a,b.cpu(),atol=2e-8,rtol=2e-8)
