import copy
import json

import numpy as np
import pytest
from scipy.special import logsumexp
import torch

from benchmark.query import PaidExecution, QueryLedger, run_isolated
from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.gcrt.joint_response import JointResponseKernel
from methods.gcrt.joint_search import select as select_joint
from methods.gcrt.physical_mixture import select
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target


def fixture():
    torch.manual_seed(37)
    candidates = sample('cut_in',8,'development')
    x = features(candidates)
    model = JointResponseKernel(x.shape[1])
    mean = model.joint_mean(8).detach().numpy()
    sources = {'names':['source_a','source_b'],'weights':np.array([.6,.4]),'meta_mean':mean,
               'means':np.stack((mean+.1,mean-.1)),
               'covariances':np.stack((np.eye(16)*.03,np.eye(16)*.02))}
    return candidates,x,model,sources


class Oracle:
    def __init__(self,candidates,margin=.7):
        self.candidates,self.margin = candidates,margin
        self.paid = []

    def query(self,index,*,purpose):
        if any(i == index for i,_ in self.paid):
            raise ValueError('Repeated paid query')
        self.paid.append((index,purpose))
        return QueryFeedback(self.candidates[index].scenario_id,index % 2 == 0,
                             SafetyMeasurements(self.margin,2),.1)


def test_fixed_physical_mass_normalizes_probabilities_and_fits_only_paid_feedback(monkeypatch):
    candidates,x,model,sources = fixture()
    original = copy.deepcopy(model.state_dict())
    fits = {'joint':[],'physical':[]}

    def fit_joint(adapted,paid_x,labels,clearance,**settings):
        fits['joint'].append((paid_x.copy(),list(labels),list(clearance),settings))
        with torch.no_grad():
            adapted.mean.add_(.02*len(labels))

    def fit_physical(adapted,paid_x,labels,**settings):
        fits['physical'].append((paid_x.copy(),list(labels),settings))
        with torch.no_grad():
            adapted.mean.add_(.01*len(labels))

    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_joint',fit_joint)
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_target',fit_physical)
    oracle = Oracle(candidates)
    result = select(candidates,oracle,pretrained=model,sources=sources,
                    mode='physical_global',initial_indices=[1,3],budget=5,device='cpu')
    assert result['selected_indices'] == [i for i,_ in oracle.paid]
    assert len(set(result['selected_indices'])) == 5
    assert [p for _,p in oracle.paid] == ['initial','initial','target','target','target']
    assert result['queries'][0]['prequery_physical_probability'] == pytest.approx(.5)
    for n,q in enumerate(result['queries'],1):
        weights = np.array(q['prequery_component_weights'])
        probabilities = np.array(q['prequery_component_probabilities'])
        assert weights.sum() == pytest.approx(1)
        assert weights[-1] == pytest.approx(.1)
        assert q['prequery_probability'] == pytest.approx(weights @ probabilities,abs=2e-12)
        assert q['prequery_probability'] >= .1*q['prequery_physical_probability']-1e-12
        expected_likelihood = np.log(probabilities[:-1] if q['feedback']['collision'] else 1-probabilities[:-1])
        np.testing.assert_allclose(q['prequery_joint_log_likelihood'],expected_likelihood,atol=2e-12)
        np.testing.assert_allclose(q['prequery_outer_weights'],[.9,.1],atol=1e-14)
        np.testing.assert_allclose(q['outer_weights_after_query'],[.9,.1],atol=1e-14)
        paid = result['selected_indices'][:n]
        paid_x,labels,clearance,settings = fits['joint'][n-1]
        np.testing.assert_array_equal(paid_x,x[paid])
        assert labels == [i % 2 == 0 for i in paid]
        assert clearance == [.7]*n
        assert settings == {'steps':10,'learning_rate':.003,'use_margin':False}
        raw_x,raw_y,raw_settings = fits['physical'][n-1]
        np.testing.assert_array_equal(raw_x,x[paid])
        assert raw_y == labels
        assert raw_settings == {'steps':10,'learning_rate':.01}
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,original[name],atol=0,rtol=0)


def test_zero_physical_mass_recovers_the_existing_joint_binary_local_rule(monkeypatch):
    candidates,_,model,sources = fixture()
    monkeypatch.setattr('methods.gcrt.physical_mixture.PHYSICAL_MASS',0)
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_joint',lambda *args,**kwargs:None)
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_target',lambda *args,**kwargs:None)
    monkeypatch.setattr('methods.gcrt.joint_search.fit_joint',lambda *args,**kwargs:None)
    expected = select_joint(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),sources=sources,
                            mode='joint_binary',initial_indices=[1],budget=4,device='cpu')
    actual = select(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),sources=sources,
                    mode='physical_local',initial_indices=[1],budget=4,device='cpu')
    assert actual['selected_indices'] == expected['selected_indices']
    np.testing.assert_allclose([q['prequery_probability'] for q in actual['queries']],
                               [q['prequery_probability'] for q in expected['queries']],atol=2e-12)


@pytest.mark.parametrize('mode',['physical_global','physical_local','physical_mean','physical_adaptive'])
def test_physical_mixture_is_source_permutation_invariant(monkeypatch,mode):
    candidates,_,model,sources = fixture()
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_joint',lambda *args,**kwargs:None)
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_target',lambda *args,**kwargs:None)
    permuted = {k:(v[::-1].copy() if k in ('means','covariances','weights') else v) for k,v in sources.items()}
    permuted['names'] = sources['names'][::-1]
    results = [select(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),sources=s,
                      mode=mode,initial_indices=[1],budget=4,device='cpu') for s in (sources,permuted)]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_allclose([q['prequery_probability'] for q in results[0]['queries']],
                               [q['prequery_probability'] for q in results[1]['queries']],atol=2e-12)


def test_adaptive_outer_control_uses_only_prequery_binary_scores(monkeypatch):
    candidates,_,model,sources = fixture()
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_joint',lambda *args,**kwargs:None)
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_target',lambda *args,**kwargs:None)
    result = select(candidates,Oracle(candidates),pretrained=model,sources=sources,
                    mode='physical_adaptive',initial_indices=[1,2],budget=4,device='cpu')
    score = np.log([.9,.1])
    for q in result['queries']:
        np.testing.assert_allclose(q['prequery_outer_weights'],np.exp(score-logsumexp(score)),atol=2e-12)
        p = np.array([q['prequery_joint_group_probability'],q['prequery_physical_probability']])
        likelihood = np.log(p if q['feedback']['collision'] else 1-p)
        np.testing.assert_allclose(q['prequery_outer_binary_log_likelihood'],likelihood,atol=2e-12)
        score += likelihood
        np.testing.assert_allclose(q['outer_weights_after_query'],np.exp(score-logsumexp(score)),atol=2e-12)
    assert not np.allclose(result['queries'][0]['outer_weights_after_query'],[.9,.1])


@pytest.mark.parametrize('mode',['physical_global','physical_target','physical_fsbo'])
def test_binary_physical_mixtures_ignore_all_online_clearance(mode):
    candidates,_,model,sources = fixture()
    if mode == 'physical_fsbo':
        model = ClassificationKernel(model.dimension,deep=True)
    results = [select(candidates,Oracle(candidates,margin),pretrained=copy.deepcopy(model),
                      sources=None if mode in ('physical_target','physical_fsbo') else sources,
                      mode=mode,initial_indices=[0],budget=3,device='cpu') for margin in (.01,90)]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_array_equal([q['prequery_probability'] for q in results[0]['queries']],
                                  [q['prequery_probability'] for q in results[1]['queries']])


def test_margin_ablation_changes_only_the_joint_branch_at_the_same_prefix():
    candidates,_,model,sources = fixture()
    results = [select(candidates,Oracle(candidates,margin),pretrained=copy.deepcopy(model),sources=sources,
                      mode='physical_margin',initial_indices=[0,1,2],budget=3,device='cpu')
               for margin in (.01,90)]
    np.testing.assert_array_equal([q['prequery_physical_probability'] for q in results[0]['queries']],
                                  [q['prequery_physical_probability'] for q in results[1]['queries']])
    assert not np.allclose([q['prequery_joint_group_probability'] for q in results[0]['queries']],
                           [q['prequery_joint_group_probability'] for q in results[1]['queries']])


def test_fsbo_mixture_preserves_the_original_classifier_fit_and_prediction():
    candidates,x,_,_ = fixture()
    model = ClassificationKernel(x.shape[1],deep=True)
    result = select(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),sources=None,
                    mode='physical_fsbo',initial_indices=[0,1,4],budget=3,device='cpu')
    adapted = copy.deepcopy(model)
    paid,labels = [],[]
    for q in result['queries']:
        with torch.no_grad():
            p = float(collision_probabilities(adapted,x[paid],labels,x)[q['index']])
        assert q['prequery_joint_group_probability'] == pytest.approx(p,abs=2e-12)
        assert q['prequery_probability'] == pytest.approx(.9*p+.1*q['prequery_physical_probability'],abs=2e-12)
        paid.append(q['index'])
        labels.append(q['feedback']['collision'])
        adapted = copy.deepcopy(model)
        fit_target(adapted,x[paid],labels,steps=10,learning_rate=.003)
    assert result['learned_group_output_channels'] == 1


def test_invalid_initialization_and_target_source_context_are_rejected_before_query():
    candidates,_,model,sources = fixture()
    oracle = Oracle(candidates)
    with pytest.raises(ValueError,match='initialization'):
        select(candidates,oracle,pretrained=model,sources=sources,mode='physical_global',
               initial_indices=[1,1],budget=3,device='cpu')
    with pytest.raises(ValueError,match='matched target'):
        select(candidates,oracle,pretrained=model,sources=sources,mode='physical_target',
               initial_indices=[1],budget=3,device='cpu')
    assert oracle.paid == []


def test_public_feature_function_is_shared_by_both_kernels_and_local_distance(monkeypatch):
    candidates,x,_,sources = fixture()
    public_x = np.column_stack((x,np.arange(len(x)) % 2))
    model = JointResponseKernel(public_x.shape[1])
    fits = []
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_joint',
                        lambda model,x,*args,**kwargs:fits.append(x.copy()))
    monkeypatch.setattr('methods.gcrt.physical_mixture.fit_target',
                        lambda model,x,*args,**kwargs:fits.append(x.copy()))
    result = select(candidates,Oracle(candidates),pretrained=model,sources=sources,
                    mode='physical_local',initial_indices=[1],budget=2,device='cpu',
                    feature_fn=lambda scenes:public_x)
    for n in (1,2):
        for paid_x in fits[2*(n-1):2*n]:
            np.testing.assert_array_equal(paid_x,public_x[result['selected_indices'][:n]])
    score = np.log(result['joint_component_prior_weights'])
    first,second = result['queries']
    influence = np.exp(-np.square(public_x[second['index']]-public_x[first['index']]).mean()/(2*.3**2))
    expected = score+np.array(first['prequery_joint_log_likelihood'])*influence
    np.testing.assert_allclose(second['prequery_joint_weights'],np.exp(expected-logsumexp(expected)),atol=2e-12)


def test_process_isolated_physical_mixture_preserves_budget_and_hides_trajectories():
    candidates,_,model,sources = fixture()

    def execute(scenario):
        index = candidates.index(scenario)
        feedback = QueryFeedback(scenario.scenario_id,index % 2 == 0,SafetyMeasurements(.7,2),.1)
        return PaidExecution(feedback,[{'private_trajectory_marker':True}])

    ledger = QueryLedger(candidates,4,execute)
    result = run_isolated(select,{'candidates':candidates,'pretrained':model,'sources':sources,
                          'mode':'physical_global','initial_indices':[1,3],'budget':4,'device':'cpu'},ledger)
    assert len(ledger.records) == 4
    assert result['selected_indices'] == [r['index'] for r in ledger.records]
    assert 'private_trajectory_marker' not in json.dumps(result)
