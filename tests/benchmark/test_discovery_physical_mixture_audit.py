import numpy as np
import pytest
import torch

from benchmark.discovery_physical_mixture_audit import classifier_probabilities_numpy
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities


@pytest.mark.parametrize('deep',[False,True])
def test_numpy_classifier_reconstruction_matches_original_kernel_batches(deep):
    torch.manual_seed(31)
    model = ClassificationKernel(3,deep=deep)
    x = np.random.default_rng(31).random((12,3))
    paid,labels = [1,4,8],[True,False,True]
    with torch.no_grad():
        expected = collision_probabilities(model,x[paid],labels,x).numpy()
    actual = classifier_probabilities_numpy(model,x,paid,labels)
    np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-10)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_numpy_classifier_reconstruction_handles_cuda_deep_kernel_at_full_support():
    torch.manual_seed(31)
    model = ClassificationKernel(5,deep=True).to('cuda')
    generator = np.random.default_rng(31)
    x = generator.random((512,5))
    paid = generator.choice(512,128,replace=False).tolist()
    labels = generator.random(128) < .3
    with torch.no_grad():
        expected = collision_probabilities(model,x[paid],labels,x).cpu().numpy()
    actual = classifier_probabilities_numpy(model,x,paid,labels)
    np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=2e-8)
