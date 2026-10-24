"""Raw-feedback and independent NumPy checks for joint response transfer."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.common.classification import logistic_moments
from methods.common.response import archive_fingerprints
from methods.gcrt.joint_response import MARGIN_NOISE, fit_joint, margin_values
from methods.gcrt.joint_training import load_model, training_contract

from .construction import load_construction
from .discovery_config import read_json, write_json
from .discovery_joint_response import (PRIORS, REPORT, gate, model_path, protocol, read_prior)
from .discovery_joint_response import METHODS, RUNS
from .discovery_config import development_tasks
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def solve_cholesky(factor,rhs):
    return np.linalg.solve(factor.swapaxes(-1,-2),np.linalg.solve(factor,rhs))


def prepare_numpy(mean,covariance,selected,labels,margin):
    selected = np.asarray(selected,dtype=int)
    size = mean.shape[-1]//2
    mean_support = mean[...,selected]
    gram = covariance[...,selected,:][..., :,selected].copy()
    margin_factor,margin_alpha,margin_to_class = None,None,None
    if margin is not None:
        indices = size+selected
        margin_gram = covariance[...,indices,:][..., :,indices]+MARGIN_NOISE**2*np.eye(len(selected))
        margin_factor = np.linalg.cholesky(margin_gram)
        margin_alpha = solve_cholesky(margin_factor,(margin-mean[...,indices])[...,None])[...,0]
        cross = covariance[...,selected,:][..., :,indices]
        mean_support = mean_support+(cross @ margin_alpha[...,None])[...,0]
        margin_to_class = np.linalg.solve(margin_factor,cross.swapaxes(-1,-2))
        gram -= margin_to_class.swapaxes(-1,-2) @ margin_to_class
    gram += 1e-5*np.eye(len(selected))
    f = mean_support.copy()
    for _ in range(10):
        p = expit(f)
        weight = np.maximum(p*(1-p),1e-9)
        root = np.sqrt(weight)
        factor = np.linalg.cholesky(np.eye(len(selected))+root[..., :,None]*gram*root[..., None,:])
        b = weight*(f-mean_support)+labels-p
        solved = solve_cholesky(factor,(root*(gram @ b[...,None])[...,0])[...,None])[...,0]
        f = mean_support+(gram @ (b-root*solved)[...,None])[...,0]
    p = expit(f)
    root = np.sqrt(np.maximum(p*(1-p),1e-9))
    factor = np.linalg.cholesky(np.eye(len(selected))+root[..., :,None]*gram*root[..., None,:])
    alpha = np.linalg.solve(gram,(f-mean_support)[...,None])[...,0]
    return selected,margin_factor,margin_alpha,margin_to_class,root,factor,alpha


def predict_numpy(mean,covariance,state,queries,*,full_covariance=False):
    selected,margin_factor,margin_alpha,margin_to_class,root,factor,alpha = state
    queries = np.asarray(queries,dtype=int)
    value = mean[...,queries].copy()
    cross = covariance[...,queries,:][..., :,selected].copy()
    diagonal = covariance.diagonal(axis1=-2,axis2=-1)[...,queries].copy()
    result_covariance = covariance[...,queries,:][..., :,queries].copy() if full_covariance else None
    if margin_factor is not None:
        indices = mean.shape[-1]//2+selected
        margin_cross = covariance[...,queries,:][..., :,indices]
        value += (margin_cross @ margin_alpha[...,None])[...,0]
        solved = np.linalg.solve(margin_factor,margin_cross.swapaxes(-1,-2))
        diagonal -= np.square(solved).sum(-2)
        cross -= solved.swapaxes(-1,-2) @ margin_to_class
        if full_covariance:
            result_covariance -= solved.swapaxes(-1,-2) @ solved
    value += (cross @ alpha[...,None])[...,0]
    solved = np.linalg.solve(factor,root[..., :,None]*cross.swapaxes(-1,-2))
    diagonal -= np.square(solved).sum(-2)
    if full_covariance:
        result_covariance -= solved.swapaxes(-1,-2) @ solved
    return value,result_covariance if full_covariance else diagonal


def prior_numpy(model,x,sources,mode):
    with torch.no_grad():
        mean = model.joint_mean(len(x)).cpu().numpy()
        covariance = model.joint_covariance(x).cpu().numpy()
    if sources is None:
        return mean[None],covariance[None]
    means = sources['means']+mean-sources['meta_mean']
    if mode == 'joint_mean':
        covariances = np.broadcast_to(covariance,(len(means),*covariance.shape))
    elif mode == 'joint_shared':
        shared = np.einsum('s,sij->ij',sources['weights'],sources['covariances'])
        covariances = np.broadcast_to(covariance+shared,(len(means),*covariance.shape))
    else:
        covariances = sources['covariances']+covariance
    return np.concatenate((means,mean[None])),np.concatenate((covariances,covariance[None]))


def audit_source_cache(kind):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history,_ = load_construction(kind)
        model,metadata = load_model(model_path(kind))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        with np.load(PRIORS/f'{kind}.npz',allow_pickle=False) as saved:
            specification = json.loads(str(saved['specification']))
            if (specification['history_fingerprints'] != archive_fingerprints(history)
                    or specification['model_sha256'] != hashlib.sha256(model_path(kind).read_bytes()).hexdigest()):
                raise ValueError('Source cache identity differs from role-checked historical responses')
            x = saved['x']
            sampled = np.linspace(0,len(x)-1,8,dtype=int)
            for s,archive in enumerate(history):
                count = len(archive.scenarios)
                combined = np.concatenate((archive.x,x))
                size = len(combined)
                with torch.no_grad():
                    mean = model.joint_mean(size).cpu().numpy()[None]
                    covariance = model.joint_covariance(combined).cpu().numpy()[None]
                state = prepare_numpy(mean,covariance,range(count),archive.collision.astype(float),
                                       margin_values(archive.safety[:,0]))
                query = np.r_[count+sampled,size+count+sampled]
                expected_mean,expected_cov = predict_numpy(mean,covariance,state,query,full_covariance=True)
                cached = np.r_[sampled,len(x)+sampled]
                np.testing.assert_allclose(saved['means'][s,cached],expected_mean[0],atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(saved['covariances'][s][np.ix_(cached,cached)],expected_cov[0],atol=2e-7,rtol=2e-6)
                print('joint_cache_audit',kind,archive.source_id,flush=True)
        return {'scenario':kind,'source_count':len(history),'candidate_points_per_source':8,
                'joint_covariance_entries_per_source':16*16,'status':'passed'}


def check_run(row,candidates,feedbacks):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        output = read_json(ROOT/row['path'])
        spec = output['specification']
        if (spec['protocol'] != read_json(REPORT/'protocol.json')
                or spec['task'] != {k:row[k] for k in spec['protocol']['tasks'][0]}
                or spec['candidate_ids'] != [s.scenario_id for s in candidates]):
            raise ValueError('Joint run differs from the registered public task')
        checkpoint = model_path(row['scenario'],row['method'] == 'joint_uncorrelated')
        if spec['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Joint run checkpoint differs')
        queries,ledger = output['result']['queries'],output['query_ledger']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(set(selected)) != 128 or len(ledger) != 128
                or selected[:8] != initial_indices(512,8,row['algorithm_seed'])
                or output['result']['selected_indices'] != selected):
            raise ValueError('Joint run budget, uniqueness, or common initialization differs')
        failures = {dedup_key(s) for s,f in zip(candidates,feedbacks) if f['collision']}
        found,curve = set(),[]
        for n,(q,record) in enumerate(zip(queries,ledger),1):
            if (q['query'] != n or record['call'] != n or record['status'] != 'complete'
                    or record['index'] != q['index'] or record['feedback'] != q['feedback']
                    or record['purpose'] != ('initial' if n <= 8 else 'target')
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]):
                raise ValueError('Joint feedback differs from the original paired collision/clearance execution')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[q['index']]))
            curve.append(len(found))
        recall = [v/len(failures) if failures else None for v in curve]
        metrics = output['metrics']
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recall[-1]
                or metrics['pool_unique_collisions'] != len(failures)
                or any(row[k] != value for k,value in metrics.items())):
            raise ValueError('Joint metrics differ from complete original collision labels')
        for b in (16,32,64,128):
            if metrics['checkpoints'][str(b)] != {'unique_collisions':curve[b-1],'collision_recall':recall[b-1]}:
                raise ValueError('Joint metric checkpoint differs')
        np.testing.assert_allclose(metrics['discovery_auc'],np.mean(curve),atol=1e-12)
        history,development = load_construction(row['scenario'])
        target_base,metadata = load_model(checkpoint)
        if (spec['history_fingerprints'] != archive_fingerprints(history)
                or metadata['specification']['history_fingerprints'] != archive_fingerprints(history)
                or metadata['specification']['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('Joint model or source role fingerprints differ')
        source = (None if row['method'] in ('joint_target','joint_uncorrelated') else
                  read_prior(row['scenario'],candidates,history,spec['checkpoint_sha256']))
        prior = np.ones(1) if source is None else np.r_[.75*source['weights'],.25]
        names = ['target_only'] if source is None else [*source['names'],'target_only']
        if output['result']['component_names'] != names:
            raise ValueError('Joint component identity differs')
        np.testing.assert_allclose(output['result']['component_prior_weights'],prior,atol=1e-14)
        x = features(candidates)
        evidence = np.zeros((len(prior),512))
        use_margin = row['method'] != 'joint_binary'
        device = spec['protocol']['device']
        target_base = target_base.to(device)
        numerical = []
        for n,q in enumerate(queries,1):
            i,y = q['index'],q['feedback']['collision']
            mean,cov = np.asarray(q['prequery_joint_means']),np.asarray(q['prequery_joint_covariances'])
            unconditioned = logistic_moments(mean[:,0],cov[:,0,0])[0]
            np.testing.assert_allclose(q['prequery_component_probabilities'],unconditioned,atol=2e-12)
            scores = np.log(prior)+evidence[:,i]
            weights = np.exp(scores-logsumexp(scores))
            np.testing.assert_allclose(q['prequery_component_weights'],weights,atol=2e-12)
            np.testing.assert_allclose(q['prequery_probability'],weights @ unconditioned,atol=2e-12)
            if n in (9,65,128):
                paid = selected[:n-1]
                labels = np.array([feedbacks[j]['collision'] for j in paid],dtype=bool)
                clearance = np.array([np.inf if feedbacks[j]['safety']['min_clearance_m'] is None else
                                      feedbacks[j]['safety']['min_clearance_m'] for j in paid],dtype=float)
                model = copy.deepcopy(target_base)
                fit_joint(model,x[paid],labels,clearance,use_margin=use_margin)
                prior_mean,prior_cov = prior_numpy(model,x,source,row['method'])
                state = prepare_numpy(prior_mean,prior_cov,paid,labels.astype(float),
                                       margin_values(clearance) if use_margin else None)
                class_mean,class_variance = predict_numpy(prior_mean,prior_cov,state,range(512))
                component_p = logistic_moments(class_mean,class_variance)[0]
                expected_mean,expected_cov = predict_numpy(prior_mean,prior_cov,state,[i,512+i],full_covariance=True)
                np.testing.assert_allclose(mean,expected_mean,atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(cov,expected_cov,atol=2e-7,rtol=2e-6)
                np.testing.assert_allclose(unconditioned,component_p[:,i],atol=2e-7,rtol=2e-6)
                logs = np.log(prior[:,None])+evidence
                all_weights = np.exp(logs-logsumexp(logs,axis=0))
                probability = (all_weights*component_p).sum(0)
                available = set(range(512))-set(paid)
                maximum = max(probability[j] for j in available)
                if probability[i] < maximum-2e-7:
                    raise ValueError('Joint selected query differs from independent full-pool probabilities')
                numerical.append({'query':n,'mixture_probability':float(probability[i]),
                                  'acquisition_gap':float(maximum-probability[i])})
            conditional_mean,conditional_variance = mean[:,0].copy(),cov[:,0,0].copy()
            density = np.zeros(len(prior))
            if use_margin:
                variance = cov[:,1,1]+MARGIN_NOISE**2
                observed = q['feedback']['safety']['min_clearance_m']
                residual = margin_values(np.inf if observed is None else observed)-mean[:,1]
                conditional_mean += cov[:,0,1]*residual/variance
                conditional_variance -= cov[:,0,1]**2/variance
                density = -.5*(np.log(2*np.pi*variance)+residual**2/variance)
            conditional_p = logistic_moments(conditional_mean,conditional_variance)[0]
            likelihood = density+np.log(np.maximum(conditional_p if y else 1-conditional_p,1e-15))
            np.testing.assert_allclose(q['conditional_collision_probabilities'],conditional_p,atol=2e-12)
            np.testing.assert_allclose(q['prequery_component_log_likelihood'],likelihood,atol=2e-10,rtol=2e-12)
            influence = (np.ones(512) if row['method'] == 'joint_global' else
                         np.exp(-np.square(x-x[i]).mean(1)/(2*.3**2)))
            evidence += likelihood[:,None]*influence
            after = np.log(prior)+evidence[:,i]
            np.testing.assert_allclose(q['component_weights_after_query'],np.exp(after-logsumexp(after)),atol=2e-11)
        if output['cost']['new_physical_calls'] != 0:
            raise ValueError('Joint replay conceals new physical calls')
        return {'path':row['path'],'prequery_gp_checks':numerical,'reliability_updates':128}


def audit():
    index = read_json(REPORT/'run_index.json')
    if (index['protocol'] != protocol() or index['protocol']['training_protocol']['contract'] != training_contract()
            or gate(index['runs']) != read_json(REPORT/'gate.json')):
        raise ValueError('Joint frozen protocol or complete admission result differs')
    cache_checks = audit_sources()['sources']
    raw = {}
    for row in index['runs']:
        key = row['target_id'],row['scenario']
        if key not in raw:
            bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
            raw[key] = bank.candidates,[bank(s).feedback.to_dict() for s in bank.candidates]
    checked = audit_rows(index['runs'],raw)
    checked.sort(key=lambda r:r['path'])
    result = {'status':'passed','runs':280,'queries':35840,'raw_banks':len(raw),
              'raw_pool_executions':512*len(raw),'joint_likelihood_and_reliability_updates_checked':35840,
              'independent_numpy_gp_prefixes':840,'source_cache_checks':cache_checks,
              'new_physical_calls':0,'previous_gates_changed':False,'checks':checked}
    write_json(REPORT/'audit/report.json',result)
    return result


def audit_rows(rows,raw):
    audit_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    feedback_hashes = {key:hashlib.sha256(json.dumps(value[1],sort_keys=True).encode()).hexdigest()
                       for key,value in raw.items()}
    checked = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {}
        for row in rows:
            key = row['target_id'],row['scenario']
            identity = {'audit_source':audit_source,'run_sha256':hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),
                        'raw_feedback_sha256':feedback_hashes[key]}
            destination = REPORT/'audit/runs'/row['method']/row['target_id']/row['scenario']/f"{row['algorithm_seed']}.json"
            if destination.exists():
                saved = read_json(destination)
                if saved['identity'] == identity:
                    checked.append(saved['result'])
                    continue
            future = pool.submit(check_run,row,*raw[key])
            futures[future] = destination,identity
        for future in as_completed(futures):
            destination,identity = futures[future]
            result = future.result()
            write_json(destination,{'status':'passed','identity':identity,'result':result})
            checked.append(result)
            if len(checked) % 8 == 0:
                print('joint_response_audit',len(checked),'/',len(rows),flush=True)
    return checked


def audit_available():
    if (read_json(REPORT/'protocol.json') != protocol()
            or read_json(REPORT/'training_protocol.json')['contract'] != training_contract()):
        raise ValueError('Current source differs from the frozen joint response protocol')
    rows,raw = [],{}
    for method in METHODS:
        for task in development_tasks():
            path = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
            if not path.exists():
                continue
            payload = read_json(path)
            rows.append({**task,'method':method,'path':str(path.relative_to(ROOT)),**payload['metrics']})
            key = task['target_id'],task['scenario']
            if key not in raw:
                bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
                raw[key] = bank.candidates,[bank(s).feedback.to_dict() for s in bank.candidates]
    checked = audit_rows(rows,raw)
    result = {'status':'available_completed_runs_checked','runs':len(checked),'expected_runs':280,
              'entire_matrix_pass_claimed':False,'new_physical_calls':0}
    write_json(REPORT/'audit/available.json',result)
    return result


def audit_sources():
    frozen = read_json(REPORT/'protocol.json')
    if frozen != protocol() or frozen['training_protocol']['contract'] != training_contract():
        raise ValueError('Source posterior audit differs from the frozen construction')
    identity = {'audit_source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'training_contract':training_contract(),'checkpoints':frozen['checkpoints'],
                'source_posteriors':frozen['source_posteriors'],
                'history_fingerprints':{kind:archive_fingerprints(load_construction(kind)[0])
                                        for kind in ('cut_in','cut_out')}}
    destination = REPORT/'audit/source_cache.json'
    if destination.exists():
        result = read_json(destination)
        if result['identity'] == identity and result['status'] == 'passed':
            return result
    result = {'status':'passed','identity':identity,
              'sources':[audit_source_cache(kind) for kind in ('cut_in','cut_out')]}
    write_json(destination,result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scope',nargs='?',default='complete',choices=('complete','available','sources'))
    scope = parser.parse_args().scope
    if scope == 'sources':
        audit_sources()
    else:
        {'complete':audit,'available':audit_available}[scope]()
