import copy

import numpy as np
import pytest
import torch

from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.baselines.fsbo.search import select
from methods.baselines.response_baselines.collision import run
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target


class Oracle:
    def __init__(self,candidates):
        self.candidates = candidates

    def query(self,index,*,purpose):
        return QueryFeedback(self.candidates[index].scenario_id,index % 2 == 0,SafetyMeasurements(.7,2),.1)


def test_explicit_device_selector_preserves_original_cpu_fsbo_exactly():
    torch.manual_seed(31)
    candidates = sample('cut_in',8,'development')
    model = ClassificationKernel(features(candidates).shape[1],deep=True)
    expected = run(candidates,Oracle(candidates),method='fsbo',initial_indices=[0,1],budget=4,
                   pretrained=copy.deepcopy(model),adaptation_config={'adaptation_steps':10,'adaptation_rate':.003})
    actual = select(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),
                    initial_indices=[0,1],budget=4,device='cpu')
    assert actual['selected_indices'] == expected['selected_indices']
    np.testing.assert_array_equal([q['prequery_probability'] for q in actual['queries']],
                                  [q['prequery_probability'] for q in expected['queries']])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_fsbo_uses_the_original_fit_and_collision_probability_functions():
    torch.manual_seed(31)
    candidates = sample('cut_in',8,'development')
    x = features(candidates)
    model = ClassificationKernel(x.shape[1],deep=True).to('cuda')
    actual = select(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),
                    initial_indices=[0,1,4],budget=3,device='cuda')
    adapted = copy.deepcopy(model)
    paid,labels = [],[]
    for q in actual['queries']:
        with torch.no_grad():
            p = float(collision_probabilities(adapted,x[paid],labels,x)[q['index']])
        assert q['prequery_probability'] == pytest.approx(p,abs=2e-9)
        paid.append(q['index'])
        labels.append(q['feedback']['collision'])
        adapted = copy.deepcopy(model)
        fit_target(adapted,x[paid],labels,steps=10,learning_rate=.003)
    assert actual['device'].startswith('cuda')
