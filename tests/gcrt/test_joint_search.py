import copy

import numpy as np
from scipy.special import logsumexp
import torch

from benchmark.scenarios import features, sample
from benchmark.schema import QueryFeedback, SafetyMeasurements
from methods.common.classification import logistic_moments
from methods.gcrt.joint_response import MARGIN_NOISE, JointResponseKernel, margin_values
from methods.gcrt.joint_search import select, transfer_prior


def fixture():
    torch.manual_seed(37)
    candidates = sample('cut_in',8,'development')
    x = features(candidates)
    model = JointResponseKernel(x.shape[1])
    mean = model.joint_mean(8).detach().numpy()
    source = {'names':['source_a','source_b'],'weights':np.array([.6,.4]),'meta_mean':mean,
              'means':np.stack((mean+.1,mean-.1)),
              'covariances':np.stack((np.eye(16)*.03,np.eye(16)*.02))}
    return candidates,x,model,source


class Oracle:
    def __init__(self,candidates,margin=.7):
        self.candidates,self.margin = candidates,margin
        self.paid = []

    def query(self,index,*,purpose):
        if any(i == index for i,_ in self.paid):
            raise ValueError('Repeated query rejected by the paid oracle')
        self.paid.append((index,purpose))
        return QueryFeedback(self.candidates[index].scenario_id,index % 2 == 0,
                             SafetyMeasurements(self.margin,2),.1)


def test_joint_search_scores_before_label_fitting_and_is_source_permutation_invariant(monkeypatch):
    candidates,x,model,source = fixture()
    fits = []

    def fit(adapted,paid_x,labels,clearance,**settings):
        fits.append((paid_x.copy(),list(labels),list(clearance),settings))
        with torch.no_grad():
            adapted.mean.add_(.02*len(labels))

    monkeypatch.setattr('methods.gcrt.joint_search.fit_joint',fit)
    oracle = Oracle(candidates)
    result = select(candidates,oracle,pretrained=copy.deepcopy(model),sources=source,
                    mode='joint_local',initial_indices=[1,3],budget=5,device='cpu')
    assert result['selected_indices'] == [i for i,_ in oracle.paid]
    assert len(set(result['selected_indices'])) == 5
    assert [p for _,p in oracle.paid] == ['initial','initial','target','target','target']
    log_weight = np.broadcast_to(np.log(result['component_prior_weights'])[:,None],(3,8)).copy()
    for n,q in enumerate(result['queries'],1):
        mu,cov = np.asarray(q['prequery_joint_means']),np.asarray(q['prequery_joint_covariances'])
        v = cov[:,1,1]+MARGIN_NOISE**2
        residual = margin_values(q['feedback']['safety']['min_clearance_m'])-mu[:,1]
        cond_mean = mu[:,0]+cov[:,0,1]*residual/v
        cond_var = cov[:,0,0]-cov[:,0,1]**2/v
        p = logistic_moments(cond_mean,cond_var)[0]
        log_density = -.5*(np.log(2*np.pi*v)+residual**2/v)
        ll = log_density+np.log(p if q['feedback']['collision'] else 1-p)
        np.testing.assert_allclose(q['prequery_component_log_likelihood'],ll,atol=2e-12)
        before = np.exp(log_weight[:,q['index']])
        np.testing.assert_allclose(q['prequery_component_weights'],before,atol=2e-12)
        np.testing.assert_allclose(q['prequery_probability'],before @ logistic_moments(mu[:,0],cov[:,0,0])[0],atol=2e-12)
        influence = np.exp(-np.square(x-x[q['index']]).mean(1)/(2*.3**2))
        log_weight += ll[:,None]*influence
        log_weight -= logsumexp(log_weight,axis=0)
        np.testing.assert_allclose(q['component_weights_after_query'],np.exp(log_weight[:,q['index']]),atol=2e-12)
        paid_x,labels,clearance,settings = fits[n-1]
        np.testing.assert_array_equal(paid_x,x[result['selected_indices'][:n]])
        assert labels == [i % 2 == 0 for i in result['selected_indices'][:n]]
        assert clearance == [.7]*n
        assert settings == {'steps':10,'learning_rate':.003,'use_margin':True}
    permuted = {k:(v[::-1].copy() if k in ('means','covariances','weights') else v) for k,v in source.items()}
    permuted['names'] = source['names'][::-1]
    other = select(candidates,Oracle(candidates),pretrained=copy.deepcopy(model),sources=permuted,
                   mode='joint_local',initial_indices=[1,3],budget=5,device='cpu')
    assert other['selected_indices'] == result['selected_indices']
    np.testing.assert_allclose([q['prequery_probability'] for q in result['queries']],
                               [q['prequery_probability'] for q in other['queries']],atol=1e-12)


def test_binary_feedback_ablation_ignores_online_clearance_values():
    candidates,_,model,source = fixture()
    results = [select(candidates,Oracle(candidates,margin),pretrained=copy.deepcopy(model),
                      sources=source,mode='joint_binary',initial_indices=[1],budget=3,device='cpu')
               for margin in (.01,90)]
    assert results[0]['selected_indices'] == results[1]['selected_indices']
    np.testing.assert_array_equal([q['prequery_probability'] for q in results[0]['queries']],
                                  [q['prequery_probability'] for q in results[1]['queries']])


def test_zero_source_information_reduces_to_target_prior():
    _,x,model,source = fixture()
    source['means'][:] = source['meta_mean']
    source['covariances'][:] = 0
    tensors = {k:torch.as_tensor(v,dtype=torch.float64) for k,v in source.items() if k != 'names'}
    mean,covariance = transfer_prior(model,x,tensors,'joint_local')
    for i in range(3):
        torch.testing.assert_close(mean[i],model.joint_mean(8),atol=0,rtol=0)
        torch.testing.assert_close(covariance[i],model.joint_covariance(x),atol=0,rtol=0)


def test_known_positive_maximum_is_removed_from_future_paid_choices(monkeypatch):
    candidates,_,model,source = fixture()
    source['means'][:,0] = 10
    source['means'][:,1:8] = -5
    monkeypatch.setattr('methods.gcrt.joint_search.fit_joint',lambda *args,**kwargs:None)
    result = select(candidates,Oracle(candidates),pretrained=model,sources=source,
                    mode='joint_local',initial_indices=[0],budget=4,device='cpu')
    assert result['selected_indices'][0] == 0
    assert len(set(result['selected_indices'])) == 4
