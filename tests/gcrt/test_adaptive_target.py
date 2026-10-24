import copy

import numpy as np
from scipy.special import logsumexp
import torch

from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.common.gp_classifier import ClassificationKernel
from methods.gcrt.adaptive_target import reliability_update, select
from methods.gcrt.posterior import CollisionPosterior


def test_reliability_is_prequential_normalized_nonmutating_and_global_at_infinite_bandwidth():
    x = np.arange(10).reshape(5,2)/10
    prior = np.broadcast_to(np.log([[.6],[.4]]),(2,5)).copy()
    original = prior.copy()
    p = np.array([.2,.8])
    actual,ll = reliability_update(prior,x,2,p,True,None)
    infinite,_ = reliability_update(prior,x,2,p,True,np.inf)
    expected = original + np.log(p[:,None])
    expected -= logsumexp(expected,axis=0)
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    np.testing.assert_allclose(infinite,actual,atol=1e-14)
    np.testing.assert_array_equal(prior,original)
    np.testing.assert_allclose(np.exp(actual).sum(0),1,atol=1e-14)
    np.testing.assert_allclose(ll,np.log(p),atol=1e-14)


def test_paid_only_target_fits_and_component_permutation_preserve_selection(monkeypatch):
    candidates = sample('cut_in',8,'development')
    x = features(candidates)
    means = np.stack((np.linspace(-1,1,8),np.linspace(.5,-.5,8)))
    covariance = np.broadcast_to(np.eye(8),(2,8,8)).copy()
    source = CollisionPosterior(means,covariance,[.3,.7],component_names=['a','b'])
    permuted = CollisionPosterior(means[::-1],covariance[::-1],[.7,.3],component_names=['b','a'])
    torch.manual_seed(7)
    pretrained = ClassificationKernel(x.shape[1],deep=True)
    fits = []

    def fit(model,paid_x,labels,**settings):
        fits.append((paid_x.copy(),list(labels),settings))
        with torch.no_grad():
            model.mean.add_(len(labels)*.05)

    class Oracle:
        def __init__(self):
            self.paid = []

        def query(self,index,*,purpose):
            self.paid.append((index,purpose))
            return QueryFeedback(candidates[index].scenario_id,index % 2 == 0,
                                 SafetyMeasurements(1,2),.1)

    monkeypatch.setattr('methods.gcrt.adaptive_target.fit_target',fit)
    oracle = Oracle()
    result = select(candidates,oracle,initial_indices=[3,1],pretrained=copy.deepcopy(pretrained),
                    source_model=source,bandwidth=.3,budget=5)
    assert result['actual_queries'] == 5
    assert result['selected_indices'] == [i for i,_ in oracle.paid]
    assert [purpose for _,purpose in oracle.paid] == ['initial','initial','target','target','target']
    assert len(fits) == 5
    np.testing.assert_allclose(source.weights,[.3,.7],atol=1e-14)
    np.testing.assert_allclose(result['component_prior_weights'],np.array([.3,.7,.25])/1.25)
    for count,(paid_x,labels,settings) in enumerate(fits,1):
        paid = result['selected_indices'][:count]
        np.testing.assert_array_equal(paid_x,x[paid])
        assert labels == [i % 2 == 0 for i in paid]
        assert settings == {'steps':10,'learning_rate':.003}
    state = np.broadcast_to(np.log(result['component_prior_weights'])[:,None],(3,8)).copy()
    for q in result['queries']:
        p = np.asarray(q['prequery_component_probabilities'])
        weights = np.exp(state[:,q['index']])
        np.testing.assert_allclose(q['prequery_component_weights'],weights,atol=1e-12)
        np.testing.assert_allclose(q['prequery_probability'],weights @ p,atol=1e-12)
        state,ll = reliability_update(state,x,q['index'],p,q['feedback']['collision'],.3)
        np.testing.assert_allclose(q['prequery_component_log_likelihood'],ll,atol=1e-12)
        np.testing.assert_allclose(q['component_weights_after_query'],np.exp(state[:,q['index']]),atol=1e-12)
    other = select(candidates,Oracle(),initial_indices=[3,1],pretrained=copy.deepcopy(pretrained),
                   source_model=permuted,bandwidth=.3,budget=5)
    assert other['selected_indices'] == result['selected_indices']
    np.testing.assert_allclose([q['prequery_probability'] for q in result['queries']],
                               [q['prequery_probability'] for q in other['queries']],atol=1e-12)


def test_raw_target_control_has_no_source_context_and_fits_both_paid_models(monkeypatch):
    candidates = sample('cut_in',4,'development')
    rates = []

    def fit(model,x,labels,*,steps,learning_rate):
        rates.append((len(labels),steps,learning_rate))

    class Oracle:
        def query(self,index,*,purpose):
            return QueryFeedback(candidates[index].scenario_id,False,SafetyMeasurements(1,2),.1)

    monkeypatch.setattr('methods.gcrt.adaptive_target.fit_target',fit)
    pretrained = ClassificationKernel(features(candidates).shape[1],deep=True)
    result = select(candidates,Oracle(),initial_indices=[1],pretrained=pretrained,
                    source_model=None,raw_target=True,budget=3)
    assert result['component_names'] == ['target_physical_gp','target_fsbo']
    assert result['component_prior_weights'] == [.5,.5]
    assert rates == [(n,10,rate) for n in (1,2,3) for rate in (.01,.003)]
