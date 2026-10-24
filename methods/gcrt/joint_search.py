"""Paid collision/clearance search with source joint-posterior transfer."""

import copy
import time

import torch

from benchmark.scenarios import features
from benchmark.schema import QueryFeedback
from methods.common.classification import torch_moments
from .joint_prediction import marginal_posterior
from .joint_response import fit_joint, margin_values, predictive_log_likelihood


def transfer_prior(model,x,sources,mode):
    base_mean,base_covariance = model.joint_mean(len(x)),model.joint_covariance(x)
    if sources is None:
        return base_mean[None],base_covariance[None]
    mean = sources['means']+(base_mean-sources['meta_mean'])
    if mode == 'joint_mean':
        covariance = base_covariance[None].expand(len(mean),-1,-1)
    elif mode == 'joint_shared':
        shared = torch.einsum('s,sij->ij',sources['weights'],sources['covariances'])
        covariance = (base_covariance+shared)[None].expand(len(mean),-1,-1)
    else:
        covariance = base_covariance[None]+sources['covariances']
    return torch.cat((mean,base_mean[None])),torch.cat((covariance,base_covariance[None]))


def select(candidates,oracle,*,pretrained,sources,mode,initial_indices,budget=128,device=None):
    torch.set_num_threads(1)
    if (not 0 < len(initial_indices) <= budget <= len(candidates)
            or len(set(initial_indices)) != len(initial_indices)
            or any(type(i) is not int or not 0 <= i < len(candidates) for i in initial_indices)):
        raise ValueError('Joint response initialization must fit the common paid budget')
    only_target = mode in ('joint_target','joint_uncorrelated')
    if (sources is None) != only_target:
        raise ValueError('Only target controls must receive no historical posterior cache')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    target_base = pretrained.to(device)
    model = copy.deepcopy(target_base)
    x = features(candidates)
    x_tensor = torch.as_tensor(x,dtype=torch.float64,device=device)
    if sources is not None:
        names = [*sources['names'],'target_only']
        sources = {key:torch.as_tensor(value,dtype=torch.float64,device=device)
                   for key,value in sources.items() if key != 'names'}
        prior = torch.cat((.75*sources['weights'],torch.tensor([.25],dtype=torch.float64,device=device)))
    else:
        names,prior = ['target_only'],torch.ones(1,dtype=torch.float64,device=device)
    log_weight = prior.log()[:,None].expand(-1,len(candidates)).clone()
    selected,labels,clearance,queries = [],[],[],[]
    remaining = set(range(len(candidates)))
    use_margin = mode != 'joint_binary'
    for step in range(budget):
        started = time.perf_counter()
        with torch.no_grad():
            mean,covariance = transfer_prior(model,x,sources,mode)
            mean,variance,cross = marginal_posterior(mean,covariance,selected,labels,
                                      margin_values(clearance) if use_margin else None)
            del covariance
            component_p = torch_moments(mean[:, :len(x)],variance[:, :len(x)])[0]
            weights = log_weight.softmax(dim=0)
            probability = (weights*component_p).sum(0).cpu().numpy()
        if step < len(initial_indices):
            index = initial_indices[step]
        else:
            maximum = max(probability[i] for i in remaining)
            index = min((i for i in remaining if probability[i] >= maximum-1e-12),
                        key=lambda i:candidates[i].scenario_id)
        selection_s = time.perf_counter()-started
        feedback = oracle.query(index,purpose='initial' if step < len(initial_indices) else 'target')
        if not isinstance(feedback,QueryFeedback) or feedback.scenario_id != candidates[index].scenario_id:
            raise ValueError('Joint response search received a different paid scenario')
        with torch.no_grad():
            local_mean = torch.stack((mean[:,index],mean[:,len(x)+index]),dim=-1)
            local_covariance = torch.stack((torch.stack((variance[:,index],cross[:,index]),dim=-1),
                                            torch.stack((cross[:,index],variance[:,len(x)+index]),dim=-1)),dim=-2)
            likelihood,conditional_p = predictive_log_likelihood(local_mean,local_covariance,0,
                    feedback.collision,feedback.safety.min_clearance_m,use_margin=use_margin)
            before = weights[:,index].cpu().tolist()
            influence = (torch.ones(len(x),dtype=torch.float64,device=device) if mode == 'joint_global' else
                         (-torch.square(x_tensor-x_tensor[index]).mean(1)/(2*.3**2)).exp())
            log_weight = log_weight+likelihood[:,None]*influence
            log_weight -= torch.logsumexp(log_weight,dim=0)
            row = {'query':step+1,'index':index,'feedback':feedback.to_dict(),
                   'prequery_probability':float(probability[index]),'component_names':names,
                   'prequery_component_probabilities':component_p[:,index].cpu().tolist(),
                   'prequery_component_weights':before,'prequery_component_log_likelihood':likelihood.cpu().tolist(),
                   'conditional_collision_probabilities':conditional_p.cpu().tolist(),
                   'prequery_joint_means':local_mean.cpu().tolist(),
                   'prequery_joint_covariances':local_covariance.cpu().tolist(),
                   'component_weights_after_query':log_weight[:,index].exp().cpu().tolist(),
                   'selection_s':selection_s,
                   'reason':'common_initialization' if step < len(initial_indices) else mode}
        selected.append(index)
        labels.append(feedback.collision)
        clearance.append(feedback.safety.min_clearance_m)
        remaining.remove(index)
        del mean,variance,cross,component_p,weights
        started = time.perf_counter()
        model = copy.deepcopy(target_base)
        fit_joint(model,x[selected],labels,clearance,steps=10,learning_rate=.003,use_margin=use_margin)
        if model.mean.is_cuda:
            torch.cuda.synchronize()
        row['adaptation_s'] = time.perf_counter()-started
        queries.append(row)
    return {'method':mode,'queries':queries,'selected_indices':selected,'actual_queries':len(queries),
            'maximum_queries':budget,'termination':'budget_exhausted','component_names':names,
            'component_prior_weights':prior.cpu().tolist(),
            'feedback_used':'paid collision and clearance' if use_margin else 'paid collision labels only'}
