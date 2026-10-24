"""Joint-response search with a separate raw-feature collision GP."""

import copy
import time

import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.classification import torch_moments
from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
from .joint_prediction import marginal_posterior
from .joint_response import fit_joint, margin_values, predictive_log_likelihood
from .joint_search import transfer_prior


PHYSICAL_MASS = .1
MODES = ('physical_global','physical_local','physical_mean','physical_target','physical_adaptive',
         'physical_margin','physical_fsbo')


def select(candidates,oracle,*,pretrained,sources,mode,initial_indices,budget=128,device=None,feature_fn=features):
    torch.set_num_threads(1)
    if mode not in MODES:
        raise ValueError('Unknown physical-mixture comparison')
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('Physical-mixture initialization must fit the paid budget')
    if (sources is None) != (mode in ('physical_target','physical_fsbo')):
        raise ValueError('Only the matched target control receives no source posteriors')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    target_base = pretrained.to(device)
    model = copy.deepcopy(target_base)
    x = feature_fn(candidates)
    x_tensor = torch.as_tensor(x,dtype=torch.float64,device=device)
    physical_base = ClassificationKernel(x.shape[1]).to(device)
    physical = copy.deepcopy(physical_base)
    if sources is not None:
        names = [*sources['names'],'target_deep']
        sources = {k:torch.as_tensor(v,dtype=torch.float64,device=device)
                   for k,v in sources.items() if k != 'names'}
        prior = torch.cat((.75*sources['weights'],torch.tensor([.25],dtype=torch.float64,device=device)))
    else:
        names,prior = ['target_deep'],torch.ones(1,dtype=torch.float64,device=device)
    joint_log_weight = prior.log()[:,None].expand(-1,len(x)).clone()
    outer_prior = torch.tensor([1-PHYSICAL_MASS,PHYSICAL_MASS],dtype=torch.float64,device=device)
    outer_log_weight = outer_prior.log()[:,None].expand(-1,len(x)).clone()
    prior_mode = 'joint_mean' if mode == 'physical_mean' else 'joint_local'
    use_margin = mode == 'physical_margin'
    selected,labels,clearance,queries = [],[],[],[]
    remaining = set(range(len(candidates)))
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            if mode == 'physical_fsbo':
                joint_p = collision_probabilities(model,x[selected],labels,x)[None]
                mean,variance,cross = None,None,None
            else:
                mean,covariance = transfer_prior(model,x,sources,prior_mode)
                mean,variance,cross = marginal_posterior(mean,covariance,selected,labels,
                                                        margin_values(clearance) if use_margin else None)
                del covariance
                joint_p = torch_moments(mean[:, :len(x)],variance[:, :len(x)])[0]
            joint_weights = joint_log_weight.softmax(dim=0)
            joint_probability = (joint_weights*joint_p).sum(0)
            physical_probability = collision_probabilities(physical,x[selected],labels,x)
            group_probability = torch.stack((joint_probability,physical_probability))
            outer_weights = outer_log_weight.softmax(dim=0)
            probability = (outer_weights*group_probability).sum(0).cpu().numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum-1e-12),
                        key=lambda i:candidates[i].scenario_id)
        selection_s = time.perf_counter()-started
        feedback = oracle.query(index,purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback,QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('Physical mixture received a different paid scenario')
        with torch.no_grad():
            if mode == 'physical_fsbo':
                local_mean,local_covariance = None,None
                conditional_p = joint_p[:,index]
                likelihood = (conditional_p if feedback.collision else 1-conditional_p).clamp_min(1e-15).log()
            else:
                local_mean = torch.stack((mean[:,index],mean[:,len(x)+index]),dim=-1)
                local_covariance = torch.stack((torch.stack((variance[:,index],cross[:,index]),dim=-1),
                                                torch.stack((cross[:,index],variance[:,len(x)+index]),dim=-1)),dim=-2)
                likelihood,conditional_p = predictive_log_likelihood(local_mean,local_covariance,0,
                                            feedback.collision,feedback.safety.min_clearance_m,use_margin=use_margin)
            influence = ((-torch.square(x_tensor-x_tensor[index]).mean(1)/(2*.3**2)).exp()
                         if mode == 'physical_local' else torch.ones(len(x),dtype=torch.float64,device=device))
            joint_log_weight += likelihood[:,None]*influence
            joint_log_weight -= torch.logsumexp(joint_log_weight,dim=0)
            outer_likelihood = (group_probability[:,index] if feedback.collision else
                                1-group_probability[:,index]).clamp_min(1e-15).log()
            before_outer = outer_weights[:,index]
            effective_weights = torch.cat((before_outer[0]*joint_weights[:,index],before_outer[1:]))
            if mode == 'physical_adaptive':
                outer_log_weight += outer_likelihood[:,None]
                outer_log_weight -= torch.logsumexp(outer_log_weight,dim=0)
            row = {'query':step+1,'index':index,'feedback':feedback.to_dict(),
                   'prequery_probability':float(probability[index]),
                   'prequery_component_probabilities':torch.cat((joint_p[:,index],physical_probability[index:index+1])).cpu().tolist(),
                   'prequery_component_weights':effective_weights.cpu().tolist(),
                   'prequery_joint_group_probability':float(joint_probability[index]),
                   'prequery_physical_probability':float(physical_probability[index]),
                   'prequery_joint_weights':joint_weights[:,index].cpu().tolist(),
                   'prequery_outer_weights':before_outer.cpu().tolist(),
                   'prequery_joint_log_likelihood':likelihood.cpu().tolist(),
                   'prequery_outer_binary_log_likelihood':outer_likelihood.cpu().tolist(),
                   'conditional_collision_probabilities':conditional_p.cpu().tolist(),
                   'prequery_joint_means':local_mean.cpu().tolist() if local_mean is not None else None,
                   'prequery_joint_covariances':local_covariance.cpu().tolist() if local_covariance is not None else None,
                   'joint_weights_after_query':joint_log_weight[:,index].exp().cpu().tolist(),
                   'outer_weights_after_query':outer_log_weight[:,index].exp().cpu().tolist(),
                   'selection_s':selection_s,
                   'reason':'common_initialization' if step < len(initial_indices) else mode}
        selected.append(index)
        labels.append(feedback.collision)
        clearance.append(feedback.safety.min_clearance_m)
        remaining.remove(index)
        del mean,variance,cross,joint_p,joint_weights,outer_weights,group_probability
        started = time.perf_counter()
        model = copy.deepcopy(target_base)
        if mode == 'physical_fsbo':
            fit_target(model,x[selected],labels,steps=10,learning_rate=.003)
        else:
            fit_joint(model,x[selected],labels,clearance,steps=10,learning_rate=.003,use_margin=use_margin)
        if model.mean.is_cuda:
            torch.cuda.synchronize()
        row['joint_adaptation_s'] = time.perf_counter()-started
        started = time.perf_counter()
        physical = copy.deepcopy(physical_base)
        fit_target(physical,x[selected],labels,steps=10,learning_rate=.01)
        if physical.mean.is_cuda:
            torch.cuda.synchronize()
        row['physical_adaptation_s'] = time.perf_counter()-started
        row['adaptation_s'] = row['joint_adaptation_s']+row['physical_adaptation_s']
        queries.append(row)
    return {'method':mode,'queries':queries,'selected_indices':selected,'actual_queries':len(queries),
            'maximum_queries':budget,'termination':'budget_exhausted',
            'component_names':[*names,'target_physical'],
            'learned_group_output_channels':1 if mode == 'physical_fsbo' else 2,
            'joint_component_prior_weights':prior.cpu().tolist(),'outer_prior_weights':outer_prior.cpu().tolist(),
            'physical_mass_is_fixed':mode != 'physical_adaptive',
            'feedback_used':('paid collision and clearance; physical component uses collision only'
                             if use_margin else 'paid collision labels only')}
